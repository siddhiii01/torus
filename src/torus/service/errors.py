"""Typed errors for the service layer."""

from __future__ import annotations

from torus.core.errors import TorusError


class ServiceError(TorusError):
    """Base class for service failures (bad request, bad config, bad state)."""


class AuthNotConfiguredError(ServiceError):
    """No API token is configured; the API refuses to start (fail closed)."""


class UnsafePathError(ServiceError):
    """A submitted path escapes the allowed data directory."""


class StageTimeoutError(ServiceError):
    """A pipeline stage exceeded its wall-clock budget."""


class ModelServerError(ServiceError):
    """A model server process failed to start or to become healthy."""
