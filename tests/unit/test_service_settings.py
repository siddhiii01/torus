from __future__ import annotations

from pathlib import Path

import pytest

from torus.core.errors import ConfigError
from torus.service.settings import (
    ModelServerConfig,
    load_service_config,
    load_settings,
)

from .service_helpers import make_settings


def test_repo_config_is_valid() -> None:
    cfg = load_service_config(Path("config/service.yaml"))
    assert cfg.default_template_id


def test_stage_override_merges_with_default(tmp_path: Path) -> None:
    s = make_settings(
        tmp_path,
        retry={"max_attempts": 3, "timeout_s": 100.0},
        stages={"read": {"timeout_s": 5.0}},
    )
    assert s.config.policy_for("read").timeout_s == 5.0
    assert s.config.policy_for("read").max_attempts == 3
    assert s.config.policy_for("write").timeout_s == 100.0


def test_model_server_rejects_external_host() -> None:
    with pytest.raises(ValueError, match="internal"):
        ModelServerConfig(name="x", command=["a"], health_url="http://example.com/h")
    ModelServerConfig(name="x", command=["a"], health_url="http://127.0.0.1:1/h")
    ModelServerConfig(name="x", command=["a"], health_url="http://10.0.0.5:1/h")


def test_data_dir_required(tmp_path: Path) -> None:
    cfg = Path("config/service.yaml")
    with pytest.raises(ConfigError):
        load_settings(cfg, env={})
    ok = load_settings(cfg, env={"TORUS_DATA_DIR": str(tmp_path)})
    assert ok.db_path.is_relative_to(tmp_path.resolve())


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "s.yaml"
    bad.write_text("default_template_id: a\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_service_config(bad)
