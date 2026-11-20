from __future__ import annotations

import pytest

from torus.service.errors import ModelServerError
from torus.service.settings import ModelServerConfig
from torus.service.supervisor import ModelServerSupervisor


class FakeProc:
    def __init__(self, code: int | None = None) -> None:
        self.code = code
        self.terminated = False

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = 0

    def kill(self) -> None:
        self.code = -9

    def wait(self, timeout: float) -> int:
        return self.code or 0


def _sup(
    healthy: bool, procs: list[FakeProc], *, timeout: float = 5.0, dead: bool = False
) -> ModelServerSupervisor:
    spec = ModelServerConfig(
        name="m",
        command=["x"],
        health_url="http://127.0.0.1:1/h",
        startup_timeout_s=timeout,
    )

    def popen(*_a: object, **_k: object) -> FakeProc:
        p = FakeProc(code=2 if dead else None)
        procs.append(p)
        return p

    return ModelServerSupervisor(
        [spec],
        health=lambda _u: healthy,
        sleep=lambda _s: None,
        popen=popen,  # type: ignore[arg-type]
    )


def test_start_and_restart_on_death() -> None:
    procs: list[FakeProc] = []
    sup = _sup(True, procs)
    sup.start_all()
    assert sup.ensure_running() == []
    procs[0].code = 1  # crashed
    assert sup.ensure_running() == ["m"]
    assert len(procs) == 2
    sup.stop_all()
    assert procs[1].terminated


def test_unhealthy_start_raises_and_cleans_up() -> None:
    procs: list[FakeProc] = []
    sup = _sup(False, procs, timeout=0.05)
    with pytest.raises(ModelServerError, match="healthy"):
        sup.start("m")
    assert procs[0].terminated


def test_exit_during_startup() -> None:
    sup = _sup(False, [], dead=True)
    with pytest.raises(ModelServerError, match="exited"):
        sup.start("m")
