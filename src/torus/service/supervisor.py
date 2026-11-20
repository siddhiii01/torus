"""Run model servers as separate processes and keep them healthy.

Each server is a configured command (for example a local vLLM or Ollama
wrapper). Processes are isolated from workers so each can be scaled and
restarted independently, one per GPU. Health checks only ever target internal
hosts (validated in config), so no document data or traffic leaves the network.
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from torus.logging import get_logger
from torus.service.errors import ModelServerError
from torus.service.settings import ModelServerConfig

_log = get_logger(__name__)


def http_health(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            return bool(200 <= resp.status < 300)
    except (urllib.error.URLError, OSError, ValueError):
        return False


class ModelServerSupervisor:
    def __init__(
        self,
        servers: list[ModelServerConfig],
        *,
        health: Callable[[str], bool] = http_health,
        sleep: Callable[[float], None] = time.sleep,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self._servers = {s.name: s for s in servers}
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._health = health
        self._sleep = sleep
        self._popen = popen

    def start_all(self) -> None:
        for name in self._servers:
            self.start(name)

    def start(self, name: str) -> None:
        spec = self._servers[name]
        proc = self._popen(
            spec.command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._procs[name] = proc
        deadline = time.monotonic() + spec.startup_timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise ModelServerError(f"model server {name!r} exited during startup")
            if self._health(spec.health_url):
                _log.info("model_server_ready", reader_name=name)
                return
            self._sleep(0.5)
        self.stop(name)
        raise ModelServerError(f"model server {name!r} not healthy in time")

    def ensure_running(self) -> list[str]:
        """Restart servers that died or fail health; return restarted names."""
        restarted: list[str] = []
        for name, spec in self._servers.items():
            proc = self._procs.get(name)
            alive = proc is not None and proc.poll() is None
            if alive and self._health(spec.health_url):
                continue
            self.stop(name)
            self.start(name)
            restarted.append(name)
        return restarted

    def stop(self, name: str) -> None:
        proc = self._procs.pop(name, None)
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def stop_all(self) -> None:
        for name in list(self._procs):
            self.stop(name)
