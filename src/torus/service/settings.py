"""Typed service settings loaded from ``config/service.yaml``."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from torus.core.errors import ConfigError

DATA_DIR_ENV = "TORUS_DATA_DIR"
TOKEN_ENV = "TORUS_API_TOKEN"
DEFAULT_CONFIG = Path("config") / "service.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(_Frozen):
    inbox: str = "inbox"
    uploads: str = "uploads"
    jobs: str = "jobs"
    db: str = "service/jobs.sqlite3"


class WatcherConfig(_Frozen):
    enabled: bool = True
    poll_interval_s: float = Field(default=5.0, gt=0)
    stable_for_s: float = Field(default=2.0, ge=0)


class ApiConfig(_Frozen):
    max_upload_bytes: int = Field(default=200 * 1024 * 1024, gt=0)


class QueueConfig(_Frozen):
    lease_s: float = Field(default=600.0, gt=0)
    heartbeat_s: float = Field(default=30.0, gt=0)
    poll_interval_s: float = Field(default=1.0, gt=0)


class RetryPolicy(_Frozen):
    max_attempts: int = Field(default=3, ge=1)
    backoff_s: float = Field(default=2.0, ge=0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    timeout_s: float = Field(default=600.0, gt=0)


class StageOverride(_Frozen):
    max_attempts: int | None = Field(default=None, ge=1)
    backoff_s: float | None = Field(default=None, ge=0)
    backoff_factor: float | None = Field(default=None, ge=1.0)
    timeout_s: float | None = Field(default=None, gt=0)


class RetentionConfig(_Frozen):
    """How long job data (input copy, work files, Excel) is kept after it ends."""

    job_days: float = Field(default=30.0, gt=0)


class MonitoringConfig(_Frozen):
    window_s: float = Field(default=3600.0, gt=0)
    # Share of succeeded jobs sent to a person to re-check confirmed values.
    spot_check_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class ModelServerConfig(_Frozen):
    name: str = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    health_url: str
    startup_timeout_s: float = Field(default=120.0, gt=0)

    @field_validator("health_url")
    @classmethod
    def _local_only(cls, value: str) -> str:
        host = urlparse(value).hostname or ""
        if not is_internal_host(host):
            msg = "model server health_url must point at a loopback/internal host"
            raise ValueError(msg)
        return value


def is_internal_host(host: str) -> bool:
    """True for localhost and loopback/private literal addresses only."""
    if host == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


class ServiceConfig(_Frozen):
    default_template_id: str = Field(min_length=1)
    paths: PathsConfig = PathsConfig()
    watcher: WatcherConfig = WatcherConfig()
    api: ApiConfig = ApiConfig()
    queue: QueueConfig = QueueConfig()
    retry: RetryPolicy = RetryPolicy()
    retention: RetentionConfig = RetentionConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    stages: dict[str, StageOverride] = Field(default_factory=dict)
    model_servers: list[ModelServerConfig] = Field(default_factory=list)

    def policy_for(self, stage: str) -> RetryPolicy:
        """Default retry policy with the stage's overrides applied."""
        override = self.stages.get(stage)
        if override is None:
            return self.retry
        patch = override.model_dump(exclude_none=True)
        return self.retry.model_copy(update=patch)


class ServiceSettings(_Frozen):
    """Config plus resolved runtime locations."""

    config: ServiceConfig
    data_dir: Path

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / self.config.paths.inbox

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / self.config.paths.uploads

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / self.config.paths.jobs

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.config.paths.db


def load_service_config(path: Path = DEFAULT_CONFIG) -> ServiceConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return ServiceConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ConfigError(
            f"invalid service config {path}: {type(exc).__name__}"
        ) from exc


def resolve_data_dir(env: dict[str, str] | None = None) -> Path:
    raw = (env if env is not None else os.environ).get(DATA_DIR_ENV, "").strip()
    if not raw:
        raise ConfigError(f"{DATA_DIR_ENV} must be set to run the service")
    path = Path(raw).resolve()
    if not path.is_dir():
        raise ConfigError(f"{DATA_DIR_ENV} is not a directory")
    return path


def load_settings(
    config_path: Path | None = None, env: dict[str, str] | None = None
) -> ServiceSettings:
    """Load ``service.yaml`` from ``TORUS_CONFIG_DIR`` (or repo config/) by default."""
    if config_path is None:
        from torus.config.loader import default_config_dir

        config_path = default_config_dir() / "service.yaml"
    return ServiceSettings(
        config=load_service_config(config_path), data_dir=resolve_data_dir(env)
    )
