"""MCP platform for Xiaomi Home integration.

Discovered automatically by mcp_gateway. Exposes per-device MIoT properties
and actions as MCP tools.
"""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, llm

from .mcp_tools import build_mcp_tools
from .miot.const import DOMAIN
from .miot.miot_device import MIoTDevice

_LOGGER = logging.getLogger(__name__)

try:
    from custom_components.mcp_gateway.types import DeviceDescription
except ImportError:
    DeviceDescription = None  # type: ignore[assignment,misc]


async def async_get_device_tools(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    device_entry: dr.DeviceEntry,
) -> tuple[list[llm.Tool], str, Any] | None:
    """Return MCP tools, prompt, and device description for a Xiaomi device.

    Called by mcp_gateway when a xiaomi_home device is discovered.
    Returns None if the device cannot be found or has no tools.
    """
    if DeviceDescription is None:
        return None
    # Look up the MIoTDevice from hass.data
    devices: list[MIoTDevice] | None = (
        hass.data.get(DOMAIN, {})
        .get("devices", {})
        .get(config_entry.entry_id)
    )
    if not devices:
        return None

    # Match device_entry to MIoTDevice via identifiers
    miot_device = _find_miot_device(device_entry, devices)
    if miot_device is None:
        return None

    tools, prompt = build_mcp_tools(miot_device)
    if not tools:
        return None

    description = DeviceDescription(
        brand=device_entry.manufacturer,
        model=miot_device.model,
        alias=miot_device.name,
        description=f"Xiaomi MIoT device: {miot_device.name} "
        f"(model: {miot_device.model})",
        device_id=miot_device.did,
    )

    return tools, prompt, description


def _find_miot_device(
    device_entry: dr.DeviceEntry,
    devices: list[MIoTDevice],
) -> MIoTDevice | None:
    """Find the MIoTDevice matching a device registry entry."""
    # xiaomi_home uses identifiers={(DOMAIN, did_tag)}
    target_tags = {
        ident[1] for ident in device_entry.identifiers if ident[0] == DOMAIN
    }
    for device in devices:
        if device.did_tag in target_tags:
            return device
    return None
