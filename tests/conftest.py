"""Fixtures every test gets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


@pytest.fixture(autouse=True)
def no_real_scheduler(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """No test reaches slurm's real ``submit.py``, even with a guard broken on purpose.

    Test runs queued real GPU jobs twice on 2026-10-22 (448 and 450): the default path exists on this
    machine, so a lookup that slipped past a guard submitted for real. The default becomes a tripwire that
    fails the test if it is run.
    """
    folder = tmp_path_factory.mktemp("tripwire")
    ran = folder / "ran"
    script = folder / "submit.py"
    script.write_text(
        f"import pathlib, sys\npathlib.Path({str(ran)!r}).write_text(' '.join(sys.argv))\nsys.exit(1)\n"
    )
    monkeypatch.setattr("localmodels.retune.DEFAULT_SUBMIT", script)
    yield
    assert not ran.exists(), f"a test ran the default submit.py: {ran.read_text()}"


@pytest.fixture(autouse=True)
def gpu_with_its_own_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test's server sees a GPU with its own memory unless the test says otherwise.

    The launcher sizes llama.cpp's prompt cache only on a GPU that shares the host's RAM, and this machine's
    does, so without this every test's server arguments would depend on the host the tests run on.
    """
    monkeypatch.setattr("localmodels.launcher._gpu_shares_host_ram", lambda: False)


@pytest.fixture(autouse=True)
def container_engines_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The container engine walk of ``refresh run`` reaches no network and no Docker in a test.

    Every tag listing fails (so nothing is submitted), sparkrun lists no tags, and the reaper removes
    nothing. A test of the walk passes its own fakes.
    """
    from localmodels import engine_refresh  # noqa: PLC0415 - loaded here, not for every importer of conftest
    from localmodels.engines import TagListing  # noqa: PLC0415

    def no_network(*_args: object, **_kwargs: object) -> None:
        msg = "network in a test"
        raise OSError(msg)

    monkeypatch.setattr("localmodels.engines.urllib_fetch", no_network)
    monkeypatch.setattr(engine_refresh, "list_sparkrun_tags", TagListing)
    monkeypatch.setattr(engine_refresh, "reap_leftovers", list)


class LoginDouble:
    """Stands in for the loopback login: changes nothing, counts the sparkrun calls that asked for it."""

    def __init__(self) -> None:
        self.calls = 0

    def env(self, base: Mapping[str, str]) -> dict[str, str]:
        self.calls += 1
        return dict(base)


@pytest.fixture(autouse=True)
def no_real_loopback_login(monkeypatch: pytest.MonkeyPatch) -> LoginDouble:
    """No test makes a key or changes the real ``~/.ssh/authorized_keys``; a test may read the double's count.

    A test of ``LoopbackLogin`` itself builds one on temporary paths with a fake runner.
    """
    double = LoginDouble()
    monkeypatch.setattr("localmodels.loopback.loopback_login", lambda: double)
    return double
