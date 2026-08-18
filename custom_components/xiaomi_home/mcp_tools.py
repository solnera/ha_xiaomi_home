"""Generate MCP tools from MIoT device specifications.

This module reads a MIoTDevice's spec and produces llm.Tool instances
that can be registered with the mcp_gateway integration, exposing each
device's readable/writable properties and actions as individual MCP tools.

Every tool built here MUST declare a response schema (``response_schema``)
next to its input schema (``parameters``). The response schema describes the
JSON object returned by ``async_call`` so that MCP clients can be served an
``outputSchema`` for the tool, and it is validated at runtime. Tools without
a response schema are rejected by :func:`build_mcp_tools`.
"""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType

from .miot.miot_client import MIoTClient
from .miot.miot_device import MIoTDevice
from .miot.miot_spec import MIoTSpecAction, MIoTSpecProperty, MIoTSpecService

_LOGGER = logging.getLogger(__name__)

_TYPE_NAMES: dict[type, str] = {
    bool: "bool",
    str: "string",
    int: "int",
    float: "float",
}

# Response schema of every set-property tool.
_SET_RESPONSE_SCHEMA = vol.Schema({vol.Required("success"): bool})


def _value_validator(prop: MIoTSpecProperty) -> Any:
    """Build a voluptuous validator for a value sent to the device."""
    if prop.format_ is bool:
        return bool
    if prop.format_ is str:
        return str

    # Numeric types (int / float)
    validators: list = [vol.Coerce(prop.format_)]
    if prop.value_range:
        validators.append(
            vol.Range(min=prop.value_range.min_, max=prop.value_range.max_)
        )
    if prop.value_list:
        allowed = [item.value for item in prop.value_list.items]
        validators.append(vol.In(allowed))
    return vol.All(*validators)


def _response_value_validator(prop: MIoTSpecProperty) -> Any:
    """Build a voluptuous validator for a value reported by the device.

    Only the type is checked, and the value is nullable: a device may fail
    to report a value or report one the spec does not allow, and an
    expression may rewrite the raw value. The value-range / value-list
    limits are spelled out in the tool description instead, so that an
    off-spec reading is not turned into a tool call error by an MCP client
    validating the response against the schema. A value of another type is
    coerced for the same reason; a bool is not, because value_format()
    already normalizes it.
    """
    if prop.format_ is bool:
        return vol.Maybe(bool)
    return vol.Maybe(vol.Coerce(prop.format_))


def _param_keys(
    params: list[MIoTSpecProperty],
) -> list[tuple[MIoTSpecProperty, str]]:
    """Pair every action parameter with a unique JSON key."""
    result: list[tuple[MIoTSpecProperty, str]] = []
    used: set[str] = set()
    for param in params:
        key = param.name
        if key in used:
            key = f"{key}_{param.iid}"
        used.add(key)
        result.append((param, key))
    return result


def _prop_schema(prop: MIoTSpecProperty) -> vol.Schema:
    """Build a voluptuous schema for a property's value based on its spec."""
    return vol.Schema({vol.Required("value"): _value_validator(prop)})


def _prop_response_schema(prop: MIoTSpecProperty) -> vol.Schema:
    """Build the response schema of a get-property tool."""
    return vol.Schema(
        {vol.Required("value"): _response_value_validator(prop)})


def _action_schema(action: MIoTSpecAction) -> vol.Schema:
    """Build a voluptuous schema for an action's input parameters."""
    if not action.in_:
        return vol.Schema({})

    schema_dict: dict = {}
    for param, key in _param_keys(action.in_):
        schema_dict[vol.Required(key)] = _value_validator(param)
    return vol.Schema(schema_dict)


def _action_response_schema(
    out_params: list[tuple[MIoTSpecProperty, str]],
) -> vol.Schema:
    """Build the response schema of an action tool.

    The result object always carries every declared out parameter; a value
    the device did not report is returned as null.
    """
    result_dict: dict = {
        vol.Required(key): _response_value_validator(param)
        for param, key in out_params
    }
    return vol.Schema({vol.Required("result"): vol.Schema(result_dict)})


def _normalize_action_out(
    out_params: list[tuple[MIoTSpecProperty, str]], out: Any
) -> dict[str, Any]:
    """Map a raw action output payload onto the declared out parameters.

    The device may report the output either as a list of
    {"piid": x, "value": y} items or as a bare list of values in the order
    declared by the spec. Both forms are accepted.
    """
    values: dict[str, Any] = {key: None for _, key in out_params}
    if not out_params or not out:
        return values
    if not isinstance(out, list):
        _LOGGER.debug("unexpected action output payload, %s", out)
        return values

    if all(isinstance(item, dict) for item in out):
        by_piid = {param.iid: (param, key) for param, key in out_params}
        for item in out:
            pair = by_piid.get(item.get("piid"))
            if pair is None:
                continue
            param, key = pair
            values[key] = param.value_format(item.get("value"))
        return values

    for (param, key), value in zip(out_params, out):
        values[key] = param.value_format(value)
    return values


def _describe_prop(prop: MIoTSpecProperty) -> str:
    """Build a human-readable description for a property."""
    desc = prop.description_trans or prop.description or prop.name
    parts = [desc]
    if prop.unit:
        parts.append(f"unit: {prop.unit}")
    if prop.value_range:
        parts.append(
            f"range: [{prop.value_range.min_}, {prop.value_range.max_}] "
            f"step {prop.value_range.step}"
        )
    if prop.value_list:
        options = ", ".join(
            f"{item.value}={item.description}"
            for item in prop.value_list.items
        )
        parts.append(f"options: {options}")
    return " | ".join(parts)


def _type_hint(prop: MIoTSpecProperty, nullable: bool = True) -> str:
    """Return the response type of a value as a short readable hint."""
    name = _TYPE_NAMES.get(prop.format_, "int")
    return f"{name}|null" if nullable else name


def _action_result_hint(
    out_params: list[tuple[MIoTSpecProperty, str]],
) -> str:
    """Describe the response of an action tool."""
    if not out_params:
        return 'returns {"result": {}}'
    fields = ", ".join(
        f'"{key}": <{_type_hint(param)}>' for param, key in out_params
    )
    return f'returns {{"result": {{{fields}}}}}'


def _tool_name(service: MIoTSpecService, name: str, prefix: str) -> str:
    """Generate a tool name: {prefix}_{service_name}_{name}."""
    svc = service.name.replace("-", "_").replace(" ", "_")
    n = name.replace("-", "_").replace(" ", "_")
    return f"{prefix}_{svc}_{n}"


def has_response_schema(tool: llm.Tool) -> bool:
    """Check that a tool declares a usable response schema."""
    schema = getattr(tool, "response_schema", None)
    return isinstance(schema, vol.Schema) and bool(schema.schema)


class MIoTMcpTool(llm.Tool):
    """Base class of the Xiaomi Home MCP tools.

    Subclasses MUST set response_schema to a non-empty vol.Schema describing
    the object they return, and implement _async_execute() instead of
    async_call().
    """

    response_schema: vol.Schema = vol.Schema({})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Run the tool and validate its response against the schema."""
        response = await self._async_execute(hass, tool_input, llm_context)
        try:
            return self.response_schema(response)
        except vol.Invalid as err:
            # A device may report a value the spec does not allow. Keep the
            # payload, the mismatch is a device or spec problem.
            _LOGGER.warning(
                "tool response mismatches its schema, %s, %s, %s",
                self.name, response, err)
            return response

    async def _async_execute(
        self, hass: HomeAssistant, tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Execute the tool and return its unvalidated response."""
        raise NotImplementedError


class MIoTGetPropertyTool(MIoTMcpTool):
    """MCP tool to read a device property."""

    def __init__(
        self,
        miot_client: MIoTClient,
        did: str,
        prop: MIoTSpecProperty,
    ) -> None:
        """Initialize the get-property tool."""
        self.name = _tool_name(prop.service, prop.name, "get")
        self.description = (
            f"Read: {_describe_prop(prop)} | "
            f'returns {{"value": <{_type_hint(prop)}>}}'
        )
        self.parameters = vol.Schema({})
        self.response_schema = _prop_response_schema(prop)
        self._miot_client = miot_client
        self._did = did
        self._siid = prop.service.iid
        self._piid = prop.iid
        self._prop = prop

    async def _async_execute(
        self, hass: HomeAssistant, tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Read the property value from the device."""
        value = await self._miot_client.get_prop_async(
            did=self._did, siid=self._siid, piid=self._piid
        )
        if value is not None:
            value = self._prop.value_format(value)
            value = self._prop.eval_expr(value)
        return {"value": value}


class MIoTSetPropertyTool(MIoTMcpTool):
    """MCP tool to write a device property."""

    def __init__(
        self,
        miot_client: MIoTClient,
        did: str,
        prop: MIoTSpecProperty,
    ) -> None:
        """Initialize the set-property tool."""
        self.name = _tool_name(prop.service, prop.name, "set")
        self.description = (
            f"Write: {_describe_prop(prop)} | "
            'returns {"success": <bool>}'
        )
        self.parameters = _prop_schema(prop)
        self.response_schema = _SET_RESPONSE_SCHEMA
        self._miot_client = miot_client
        self._did = did
        self._siid = prop.service.iid
        self._piid = prop.iid
        self._prop = prop

    async def _async_execute(
        self, hass: HomeAssistant, tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Write a value to the device property."""
        value = tool_input.tool_args["value"]
        value = self._prop.value_format(value)
        if self._prop.value_range:
            value = self._prop.value_precision(value)
        success = await self._miot_client.set_prop_async(
            did=self._did, siid=self._siid, piid=self._piid, value=value
        )
        return {"success": bool(success)}


class MIoTActionTool(MIoTMcpTool):
    """MCP tool to invoke a device action."""

    def __init__(
        self,
        miot_client: MIoTClient,
        did: str,
        action: MIoTSpecAction,
    ) -> None:
        """Initialize the action tool."""
        self.name = _tool_name(action.service, action.name, "action")
        self._in_params = _param_keys(action.in_)
        self._out_params = _param_keys(action.out)
        desc = (
            action.description_trans or action.description or action.name
        )
        self.description = (
            f"{desc} | {_action_result_hint(self._out_params)}"
        )
        self.parameters = _action_schema(action)
        self.response_schema = _action_response_schema(self._out_params)
        self._miot_client = miot_client
        self._did = did
        self._siid = action.service.iid
        self._aiid = action.iid
        self._action = action

    async def _async_execute(
        self, hass: HomeAssistant, tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Execute the action on the device."""
        in_list = [
            {
                "piid": param.iid,
                "value": param.value_format(tool_input.tool_args[key]),
            }
            for param, key in self._in_params
            if key in tool_input.tool_args
        ]
        out = await self._miot_client.action_async(
            did=self._did, siid=self._siid, aiid=self._aiid, in_list=in_list
        )
        return {"result": _normalize_action_out(self._out_params, out)}


def build_mcp_tools(
    device: MIoTDevice,
) -> tuple[list[llm.Tool], str]:
    """Build MCP tools and prompt for a single MIoT device.

    Iterates ALL services in the device's MIoT spec to cover every
    capability: service-level entities (light, climate, cover, fan,
    vacuum, etc.), individual properties, and standalone actions.

    Tools that do not declare a response schema are dropped.

    Returns (tools, prompt).
    """
    tools: list[llm.Tool] = []
    miot_client = device.miot_client
    did = device.did
    seen_props: set[int] = set()   # Track by spec_id to avoid duplicates
    seen_actions: set[int] = set()

    prompt_sections: list[str] = [
        f"Xiaomi device: {device.name} (model: {device.model})",
    ]

    def add_tool(tool_list: list[llm.Tool], tool: llm.Tool) -> None:
        """Append a tool once its response schema has been checked."""
        if not has_response_schema(tool):
            _LOGGER.error(
                "tool without response schema is not exposed, %s, %s",
                device.did_tag, tool.name)
            return
        tool_list.append(tool)

    for service in device.spec_instance.services:
        svc_desc = (
            service.description_trans or service.description or service.name
        )
        svc_tools: list[llm.Tool] = []

        # Properties
        for prop in service.properties:
            if prop.spec_id in seen_props:
                continue
            if not prop.access:
                continue
            if prop.need_filter:
                continue
            seen_props.add(prop.spec_id)

            if prop.readable:
                add_tool(svc_tools, MIoTGetPropertyTool(miot_client, did, prop))
            if prop.writable:
                add_tool(svc_tools, MIoTSetPropertyTool(miot_client, did, prop))

        # Actions
        for action in service.actions:
            if action.spec_id in seen_actions:
                continue
            if action.need_filter:
                continue
            seen_actions.add(action.spec_id)
            add_tool(svc_tools, MIoTActionTool(miot_client, did, action))

        if svc_tools:
            tools.extend(svc_tools)
            prompt_sections.append(f"\n[{svc_desc}]")
            prompt_sections.extend(
                f"  - {tool.name}: {tool.description}" for tool in svc_tools
            )

    prompt_sections.insert(1, f"Total tools: {len(tools)}")

    return tools, "\n".join(prompt_sections)
