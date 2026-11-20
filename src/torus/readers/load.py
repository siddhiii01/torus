"""Lazy import of reader plugins from ``ReaderSpec.module`` paths."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

from torus.config.schemas import DeviceKind, ReaderSpec
from torus.core.errors import ReaderError
from torus.core.protocols import Reader

CreateReader = Callable[..., Reader]


def import_reader_module(module_path: str) -> Any:
    """Import ``module_path``. Heavy deps must live inside the reader, not here."""
    try:
        return importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 — missing optional stack = skip
        raise ReaderError(
            f"failed to import reader module {module_path!r}: {exc}"
        ) from exc


def resolve_factory(module: Any, module_path: str) -> CreateReader:
    """Return ``create_reader`` or a constructible Reader class from ``module``."""
    factory = getattr(module, "create_reader", None)
    if callable(factory):
        return cast(CreateReader, factory)

    for attr in ("Reader", "EchoReader"):
        cls = getattr(module, attr, None)
        if isinstance(cls, type):
            return _class_factory(cls)

    # Last segment of the module path as a PascalCase class name.
    leaf = module_path.rsplit(".", 1)[-1]
    guess = "".join(part.capitalize() for part in leaf.split("_")) + "Reader"
    cls = getattr(module, guess, None)
    if isinstance(cls, type):
        return _class_factory(cls)

    raise ReaderError(
        f"reader module {module_path!r} must export create_reader() or a Reader class"
    )


def _class_factory(cls: type) -> CreateReader:
    def factory(*, device: str, spec: ReaderSpec, **kwargs: Any) -> Reader:
        return cast(Reader, cls(device=device, spec=spec, **kwargs))

    return factory


def instantiate_reader(
    spec: ReaderSpec,
    *,
    device: DeviceKind,
) -> Reader:
    """Import and construct one reader. Raises ``ReaderError`` on hard failure."""
    if not spec.module:
        raise ReaderError(f"reader {spec.name!r} has no module path")
    module = import_reader_module(spec.module)
    factory = resolve_factory(module, spec.module)
    try:
        reader = factory(device=device.value, spec=spec)
    except TypeError:
        # Older / minimal stubs may only take device=.
        reader = factory(device=device.value)
    except Exception as exc:  # noqa: BLE001
        raise ReaderError(f"failed to construct reader {spec.name!r}: {exc}") from exc

    name = getattr(reader, "name", None)
    if not isinstance(name, str) or not name:
        raise ReaderError(f"reader from {spec.module!r} has no usable name attribute")
    caps = getattr(reader, "capabilities", None)
    read = getattr(reader, "read", None)
    if not callable(caps) or not callable(read):
        raise ReaderError(
            f"object from {spec.module!r} does not satisfy the Reader protocol"
        )
    return reader
