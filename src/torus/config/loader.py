"""Load and validate ``config/*.yaml``; compute a stable config hash."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from torus.config.schemas import (
    FormFamilySpec,
    LocalOverrides,
    PageTypesConfig,
    PipelineConfig,
    ReadersConfig,
    ReaderSpec,
    TemplateConfig,
    TorusConfig,
)
from torus.core.errors import ConfigError

_ENV_CONFIG_DIR = "TORUS_CONFIG_DIR"

_PIPELINE_FILE = "pipeline.yaml"
_READERS_FILE = "readers.yaml"
_PAGE_TYPES_FILE = "page_types.yaml"
_LOCAL_FILE = "local.yaml"
_TEMPLATES_DIR = "templates"
_FAMILIES_DIR = "families"


def default_config_dir() -> Path:
    """Resolve the config root: ``TORUS_CONFIG_DIR`` or repo ``config/``."""
    override = os.environ.get(_ENV_CONFIG_DIR)
    if override:
        return Path(override).expanduser().resolve()
    # src/torus/config/loader.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "config"


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {path}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def _validate[TModel: BaseModel](
    model_type: type[TModel], data: Any, *, source: Path | str
) -> TModel:
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {source}:\n{exc}") from exc


def load_pipeline(path: Path) -> PipelineConfig:
    """Load and validate a pipeline.yaml file."""
    return _validate(PipelineConfig, _read_yaml(path), source=path)


def load_readers(path: Path) -> ReadersConfig:
    """Load and validate a readers.yaml file."""
    return _validate(ReadersConfig, _read_yaml(path), source=path)


def load_page_types(path: Path) -> PageTypesConfig:
    """Load and validate a page_types.yaml file."""
    return _validate(PageTypesConfig, _read_yaml(path), source=path)


def load_template(path: Path) -> TemplateConfig:
    """Load and validate one template YAML under config/templates/."""
    template = _validate(TemplateConfig, _read_yaml(path), source=path)
    stem = path.stem
    if template.id != stem:
        raise ConfigError(
            f"template id {template.id!r} must match file stem {stem!r} ({path})"
        )
    return template


def load_family(path: Path) -> FormFamilySpec:
    """Load and validate one form-family hint YAML under config/families/."""
    family = _validate(FormFamilySpec, _read_yaml(path), source=path)
    stem = path.stem
    if family.id != stem:
        raise ConfigError(
            f"family id {family.id!r} must match file stem {stem!r} ({path})"
        )
    return family


def load_families(families_dir: Path | str) -> dict[str, FormFamilySpec]:
    """Load all family YAMLs. Missing directory is empty (hints are optional)."""
    root = Path(families_dir)
    if not root.is_dir():
        return {}
    families: dict[str, FormFamilySpec] = {}
    paths = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    seen_stems: set[str] = set()
    for path in paths:
        if path.stem in seen_stems:
            continue
        seen_stems.add(path.stem)
        family = load_family(path)
        families[family.id] = family
    return families


def default_families_dir(config_dir: Path | str | None = None) -> Path:
    """Resolve ``config/families`` from a config root (or the default root)."""
    root = Path(config_dir) if config_dir is not None else default_config_dir()
    return root.expanduser().resolve() / _FAMILIES_DIR


def _load_templates(templates_dir: Path) -> dict[str, TemplateConfig]:
    if not templates_dir.is_dir():
        raise ConfigError(f"templates directory not found: {templates_dir}")
    templates: dict[str, TemplateConfig] = {}
    paths = sorted(templates_dir.glob("*.yaml")) + sorted(templates_dir.glob("*.yml"))
    # Deduplicate if both extensions somehow collide on the same stem.
    seen_stems: set[str] = set()
    for path in paths:
        if path.stem in seen_stems:
            continue
        seen_stems.add(path.stem)
        template = load_template(path)
        templates[template.id] = template
    return templates


def _load_local_overrides(path: Path) -> LocalOverrides | None:
    if not path.is_file():
        return None
    return _validate(LocalOverrides, _read_yaml(path), source=path)


def _apply_local_overrides(
    readers: ReadersConfig, overrides: LocalOverrides | None
) -> ReadersConfig:
    if overrides is None or not overrides.readers:
        return readers
    unknown = sorted(set(overrides.readers) - {r.name for r in readers.readers})
    if unknown:
        raise ConfigError(
            "local.yaml references unknown readers: " + ", ".join(unknown)
        )
    updated: list[ReaderSpec] = []
    for reader in readers.readers:
        if reader.name in overrides.readers:
            updated.append(
                reader.model_copy(update={"enabled": overrides.readers[reader.name]})
            )
        else:
            updated.append(reader)
    return ReadersConfig(readers=updated)


def _cross_check(
    page_types: PageTypesConfig, templates: dict[str, TemplateConfig]
) -> None:
    known = page_types.ids()
    for template in templates.values():
        for field in template.fields:
            missing = [pt for pt in field.look_in if pt not in known]
            if missing:
                raise ConfigError(
                    f"template {template.id!r} field {field.key!r}: "
                    f"look_in references unknown page types: {', '.join(missing)}"
                )


def _canonical_payload(
    pipeline: PipelineConfig,
    readers: ReadersConfig,
    page_types: PageTypesConfig,
    templates: dict[str, TemplateConfig],
) -> dict[str, Any]:
    return {
        "pipeline": pipeline.model_dump(mode="json"),
        "readers": readers.model_dump(mode="json"),
        "page_types": page_types.model_dump(mode="json"),
        "templates": {
            tid: templates[tid].model_dump(mode="json") for tid in sorted(templates)
        },
    }


def compute_config_hash(
    pipeline: PipelineConfig,
    readers: ReadersConfig,
    page_types: PageTypesConfig,
    templates: dict[str, TemplateConfig],
) -> str:
    """SHA-256 of the canonical JSON of the validated config tree.

    The hash covers effective config after local overrides, so results record
    exactly what ran. ``config_hash`` itself is not part of the payload.
    """
    payload = _canonical_payload(pipeline, readers, page_types, templates)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(config_dir: Path | str | None = None) -> TorusConfig:
    """Load pipeline, readers, page types and templates; validate; hash.

    Raises ``ConfigError`` on missing files, invalid YAML, schema failures,
    or cross-file inconsistencies (e.g. unknown ``look_in`` page types).
    """
    root = Path(config_dir) if config_dir is not None else default_config_dir()
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ConfigError(f"config directory not found: {root}")

    pipeline = load_pipeline(root / _PIPELINE_FILE)
    readers = load_readers(root / _READERS_FILE)
    page_types = load_page_types(root / _PAGE_TYPES_FILE)
    templates = _load_templates(root / _TEMPLATES_DIR)

    overrides = _load_local_overrides(root / _LOCAL_FILE)
    readers = _apply_local_overrides(readers, overrides)
    _cross_check(page_types, templates)

    config_hash = compute_config_hash(pipeline, readers, page_types, templates)
    return TorusConfig(
        pipeline=pipeline,
        readers=readers,
        page_types=page_types,
        templates=templates,
        config_hash=config_hash,
    )
