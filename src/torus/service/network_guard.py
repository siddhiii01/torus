"""In-process outbound network guard for workers (P11.5).

Workers may only connect to loopback or private addresses (local model
servers). This is defence in depth: deployments must also block egress at the
container or firewall level (P11.6). The guard cannot stop native code that
opens sockets itself.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

from torus.service.errors import ServiceError


class OutboundNetworkBlockedError(ServiceError):
    """A worker tried to connect outside the allowed (internal) address range."""


_installed = False
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex


def _check(address: Any) -> None:
    if not isinstance(address, tuple) or not address:
        return  # AF_UNIX and other non-IP families stay local
    host = str(address[0])
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        # A hostname reached connect(): refuse rather than resolve it here.
        raise OutboundNetworkBlockedError("outbound connection blocked") from None
    if not (ip.is_loopback or ip.is_private):
        raise OutboundNetworkBlockedError("outbound connection blocked")


def _guarded_connect(self: socket.socket, address: Any) -> None:
    _check(address)
    _original_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    _check(address)
    return _original_connect_ex(self, address)


def install() -> None:
    global _installed
    if _installed:
        return
    setattr(socket.socket, "connect", _guarded_connect)  # noqa: B010
    setattr(socket.socket, "connect_ex", _guarded_connect_ex)  # noqa: B010
    _installed = True


def uninstall() -> None:
    global _installed
    setattr(socket.socket, "connect", _original_connect)  # noqa: B010
    setattr(socket.socket, "connect_ex", _original_connect_ex)  # noqa: B010
    _installed = False
