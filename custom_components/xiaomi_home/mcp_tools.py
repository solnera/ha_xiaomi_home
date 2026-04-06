"""Generate MCP tools from MIoT device specifications.

This module reads a MIoTDevice's spec and produces llm.Tool instances
that can be registered with the mcp_gateway integration, exposing each
device's readable/writable properties and actions as individual MCP tools.
"""
from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType

from .miot.miot_client import MIoTClient
from .miot.miot_device import MIoTDevice
from .miot.miot_spec import MIoTSpecAction, MIoTSpecProperty, MIoTSpecService

_LOGGER = logging.getLogger(__name__)


def _prop_schema(prop: MIoTSpecProperty) -> vol.Schema:
    """Build a voluptuous schema for a property's value based on its spec."""
    if prop.format_ is bool:
        return vol.Schema({vol.Required("value"): bool})

    if prop.format_ is str:
        return vol.Schema({vol.Required("value"): str})

    # Numeric types (int / float)
    coerce = vol.Coerce(prop.format_)
    validators: list = [coerce]

    if prop.value_range:
        validators.append(
            vol.Range(min=prop.value_range.min_, max=prop.value_range.max_)
        )
    if prop.value_list:
        allowed = [item.value for item in prop.value_list.items]
        validators.append(vol.In(allowed))

    return vol.Schema({vol.Required("value"): vol.All(*validators)})


def _action_schema(action: MIoTSpecAction) -> vol.Schema:
    """Build a voluptuous schema for an action's input parameters."""
    if not action.in_:
        return vol.Schema({})

    schema_dict: dict = {}
    for param in action.in_:
        key = vol.Required(param.name)
        if param.format_ is bool:
            schema_dict[key] = bool
        elif param.format_ is str:
            schema_dict[key] = str
        else:
            coerce = vol.Coerce(param.format_)
            validators: list = [coerce]
            if param.value_range:
                validators.append(
                    vol.Range(
                        min=param.value_range.min_, max=param.value_range.max_
                    )
                )
            if param.value_list:
                allowed = [item.value for item in param.value_list.items]
                validators.append(vol.In(allowed))
            schema_dict[key] = vol.All(*validators)
    return vol.Schema(schema_dict)


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


def _tool_name(service: MIoTSpecService, name: str, prefix: str) -> str:
    """Generate a tool name: {prefix}_{service_name}_{name}."""
    svc = service.name.replace("-", "_").replace(" ", "_")
    n = name.replace("-", "_").replace(" ", "_")
    return f"{prefix}_{svc}_{n}"


class MIoTGetPropertyTool(llm.Tool):
    """MCP tool to read a device property."""

    def __init__(
        self,
        miot_client: MIoTClient,
        did: str,
        prop: MIoTSpecProperty,
    ) -> None:
        """Initialize the get-property tool."""
        self.name = _tool_name(prop.service, prop.name, "get")
        self.description = f"Read: {_describe_prop(prop)}"
        self.parameters = vol.Schema({})
        self._miot_client = miot_client
        self._did = did
        self._siid = prop.service.iid
        self._piid = prop.iid
        self._prop = prop

    async def async_call(
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


class MIoTSetPropertyTool(llm.Tool):
    """MCP tool to write a device property."""

    def __init__(
        self,
        miot_client: MIoTClient,
        did: str,
        prop: MIoTSpecProperty,
    ) -> None:
        """Initialize the set-property tool."""
        self.name = _tool_name(prop.service, prop.name, "set")
        self.description = f"Write: {_describe_prop(prop)}"
        self.parameters = _prop_schema(prop)
        self._miot_client = miot_client
        self._did = did
        self._siid = prop.service.iid
        self._piid = prop.iid
        self._prop = prop

    async def async_call(
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
        return {"success": success}


class MIoTActionTool(llm.Tool):
    """MCP tool to invoke a device action."""

    def __init__(
        self,
        miot_client: MIoTClient,
        did: str,
        action: MIoTSpecAction,
    ) -> None:
        """Initialize the action tool."""
        self.name = _tool_name(action.service, action.name, "action")
        self.description = (
            action.description_trans or action.description or action.name
        )
        self.parameters = _action_schema(action)
        self._miot_client = miot_client
        self._did = did
        self._siid = action.service.iid
        self._aiid = action.iid
        self._action = action

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Execute the action on the device."""
        in_list = [
            {
                "piid": param.iid,
                "value": param.value_format(tool_input.tool_args[param.name]),
            }
            for param in self._action.in_
            if param.name in tool_input.tool_args
        ]
        out = await self._miot_client.action_async(
            did=self._did, siid=self._siid, aiid=self._aiid, in_list=in_list
        )
        return {"result": out}


def build_mcp_tools(
    device: MIoTDevice,
) -> tuple[list[llm.Tool], str]:
    """Build MCP tools and prompt for a single MIoT device.

    Iterates ALL services in the device's MIoT spec to cover every
    capability: service-level entities (light, climate, cover, fan,
    vacuum, etc.), individual properties, and standalone actions.

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

    for service in device.spec_instance.services:
        svc_desc = service.description_trans or service.description or service.name
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
                tool = MIoTGetPropertyTool(miot_client, did, prop)
                svc_tools.append(tool)
            if prop.writable:
                tool = MIoTSetPropertyTool(miot_client, did, prop)
                svc_tools.append(tool)

        # Actions
        for action in service.actions:
            if action.spec_id in seen_actions:
                continue
            if action.need_filter:
                continue
            seen_actions.add(action.spec_id)
            tool = MIoTActionTool(miot_client, did, action)
            svc_tools.append(tool)

        if svc_tools:
            tools.extend(svc_tools)
            prompt_sections.append(f"\n[{svc_desc}]")
            prompt_sections.extend(
                f"  - {tool.name}: {tool.description}" for tool in svc_tools
            )

    prompt_sections.insert(1, f"Total tools: {len(tools)}")

    return tools, "\n".join(prompt_sections)
