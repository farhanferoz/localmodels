"""The launcher's memory guards, with fake memory readings and a fake server; no GPU, no model."""

from __future__ import annotations

import asyncio
import os
import stat
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from localmodels.launcher import LocalServerManager, MemoryGuards, ServerStartError
from localmodels.lease import GpuLease
from localmodels.pinned import PinnedFile
from localmodels.process import ServerSpec, build_args
from localmodels.stack import Engine, Settings, SpecType
from tests.test_launcher import _answer, _gpu_params, _make_fake_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

PORT = 8187
PROJECTED_GIB = 20  # what the fake llama-fit-params prints (20480 MiB)


def _fit_params(bin_path: Path, output: str = "projected to use 20480 MiB of device memory") -> None:
    script = bin_path.with_name("llama-fit-params")
    script.write_text(f"#!/bin/sh\necho '{output}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _manager(
    tmp_path: Path,
    http: httpx.AsyncClient,
    guards: MemoryGuards,
    readings: Iterator[float],
    *,
    health: int = 3,
    **extra: Any,
) -> LocalServerManager:
    model = tmp_path / "test.gguf"
    model.touch()
    bin_path = _make_fake_server(tmp_path, min_health_attempts=health)
    _fit_params(bin_path)
    last = [0.0]

    def available(_on_gpu: bool) -> float:
        last[0] = next(readings, last[0])
        return last[0]

    return LocalServerManager(
        ServerSpec(host="127.0.0.1", port=PORT, model_path=model),
        _gpu_params(),
        GpuLease(tmp_path / "lease", project="localmodels-test"),
        http=http,
        pid_dir=tmp_path / "pids",
        binary=bin_path,
        stop_grace_s=1.0,
        ready_timeout_s=10.0,
        guards=guards,
        available_gib=available,
        **extra,
    )


def _steady(gib: float) -> Iterator[float]:
    while True:
        yield gib


async def test_load_refused_when_projection_plus_reserve_exceeds_available(tmp_path: Path) -> None:
    guards = MemoryGuards(host_reserve_gib=12)
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, _steady(PROJECTED_GIB + 11))
        with pytest.raises(ServerStartError, match=r"needs 32\.0 GiB .* 31\.0 GiB is available"):
            await manager.start()
        assert not (tmp_path / "pids" / f"{PORT}.pid").exists()


async def test_load_refused_when_there_is_no_projection(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, MemoryGuards(host_reserve_gib=12), _steady(500))
        _fit_params(tmp_path / "bin" / "llama-server", "nothing useful")
        with pytest.raises(ServerStartError, match="printed no projection"):
            await manager.start()


async def test_server_that_leaves_less_than_the_reserve_is_stopped(tmp_path: Path) -> None:
    readings = iter([100.0, 5.0])  # before the load it fits; after, 5 GiB is left
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, MemoryGuards(host_reserve_gib=12), readings)
        with pytest.raises(ServerStartError, match=r"left 5\.0 GiB .* under the 12 GiB reserve"):
            await manager.start()
        assert not (tmp_path / "pids" / f"{PORT}.pid").exists()


async def test_kill_floor_during_the_load(tmp_path: Path) -> None:
    guards = MemoryGuards(kill_below_gib=10, load_poll_s=0.05)
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, _steady(4), health=10_000)
        with pytest.raises(
            ServerStartError, match=r"killed while loading: available memory fell to 4\.0 GiB"
        ):
            await manager.start()


async def test_kill_floor_watchdog_while_running(tmp_path: Path) -> None:
    guards = MemoryGuards(kill_below_gib=10, load_poll_s=0.05, watch_poll_s=0.05)
    level = [50.0]

    def readings() -> Iterator[float]:
        while True:
            yield level[0]

    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, readings())
        await manager.start()
        assert (await manager.status()).alive
        level[0] = 3.0
        for _ in range(100):
            await asyncio.sleep(0.05)
            if not (await manager.status()).alive:
                break
        assert not (await manager.status()).alive
        assert manager.killed_reason is not None


async def test_server_oom_score_adj_is_set_on_the_server_alone(tmp_path: Path) -> None:
    guards = MemoryGuards(server_oom_score_adj=700)
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, _steady(500))
        handle = await manager.start()
        try:
            with open(f"/proc/{handle.pid}/oom_score_adj") as fh:  # noqa: ASYNC230,PTH123
                assert fh.read().strip() == "700"
            with open("/proc/self/oom_score_adj") as fh:  # noqa: ASYNC230,PTH123
                assert fh.read().strip() != "700"
            assert os.getpgid(handle.pid) != os.getpgid(0)
        finally:
            await manager.stop()


async def test_a_settings_answer_reaches_the_server_flags_the_fit_check_and_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []  # every process started: llama-fit-params runs, then the llama-server
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        calls.append([str(a) for a in args])
        return await real(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    draft = tmp_path / "draft.gguf"
    with draft.open("wb") as fh:
        fh.truncate(30 * 1024**3)  # sparse: 30 GiB by its size, none on disk
    chosen = Settings(n_ubatch=2048, spec=SpecType.DRAFT_MTP, spec_draft_file=str(draft))
    guards, model = MemoryGuards(host_reserve_gib=12), str(tmp_path / "test.gguf")
    async with httpx.AsyncClient() as http:
        # no answer: the argument lists it always had
        plain = _manager(tmp_path, http, guards, _steady(100), health=1)
        await plain.start()
        await plain.stop()
        fit, server = calls
        assert fit[1:] == ["-m", model, "-c", "8192", "-np", "1", "-lv", "4"]
        assert server == build_args(plain.binary, plain.spec, plain.params)
        assert plain.stamp().settings is None

        # no answer, but its own `-ub` and draft: still read by nothing, so the lists stay as they were
        calls.clear()
        own = ("-ub", "512", "-md", "/x.gguf")
        owner = _manager(tmp_path, http, guards, _steady(100), health=1, extra_flags=own)
        await owner.start()
        await owner.stop()
        assert [c[1:] for c in calls[:-1]] == [fit[1:]]
        assert calls[-1] == build_args(owner.binary, owner.spec, owner.params, own)

        # an answer: its flags; an MTP head, which llama-fit-params cannot load alone, counts as its size
        calls.clear()
        asked = _manager(
            tmp_path, http, guards, _steady(100), health=1, settings=_answer(**chosen.model_dump())
        )
        await asked.start()
        await asked.stop()
        added = ("-ub", "2048", "--spec-type", "draft-mtp", "--spec-draft-model", str(draft))
        fit, server = calls
        assert (fit[2], fit[-4:]) == (model, ["-ub", "2048", "-lv", "4"])
        assert server == build_args(asked.binary, asked.spec, asked.params, added)
        stamped = asked.stamp()
        assert stamped.layout is not None
        assert stamped.layout.extra_flags == added
        assert stamped.settings is not None
        assert (stamped.settings.applied, stamped.settings.overridden) == (chosen, [])

        # the caller's own flags and environment win, group by group, and the fit check follows them
        calls.clear()
        theirs = ("--spec-type", "ngram-mod")
        kept = _manager(
            tmp_path, http, guards, _steady(100), health=1, settings=_answer(**chosen.model_dump()),
            extra_flags=theirs, extra_env={"LLAMA_ARG_UBATCH": "256"},
        )  # fmt: skip
        await kept.start()
        await kept.stop()
        assert len(calls) == 2
        assert calls[0][-4:] == ["-ub", "256", "-lv", "4"]
        assert calls[1] == build_args(kept.binary, kept.spec, kept.params, theirs)
        stamped = kept.stamp()
        assert stamped.settings is not None
        assert (stamped.settings.applied, stamped.settings.overridden) == (Settings(), ["n_ubatch", "spec"])


async def test_a_draft_model_is_projected_on_its_own_and_an_mtp_head_counts_as_its_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        calls.append([str(a) for a in args])
        return await real(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    draft = tmp_path / "draft.gguf"
    with draft.open("wb") as fh:
        fh.truncate(30 * 1024**3)  # sparse: 30 GiB by its size, none on disk
    head = Settings(n_ubatch=2048, spec=SpecType.DRAFT_MTP, spec_draft_file=str(draft))
    simple = head.model_copy(update={"spec": SpecType.DRAFT_SIMPLE})
    guards, model = MemoryGuards(host_reserve_gib=12), str(tmp_path / "test.gguf")
    async with httpx.AsyncClient() as http:
        # an MTP head: llama-fit-params projects the model alone, and the head's 30 GiB are added to it
        short = _manager(
            tmp_path, http, guards, _steady(PROJECTED_GIB + 30 + 11), settings=_answer(**head.model_dump())
        )
        with pytest.raises(ServerStartError, match=r"needs 62\.0 GiB"):
            await short.start()
        assert [c[2] for c in calls] == [model]

        # the type from the environment and from the flags together, as llama.cpp keeps every value given
        calls.clear()
        mixed = _manager(
            tmp_path, http, guards, _steady(PROJECTED_GIB + 30 + 11), settings=_answer(n_ubatch=2048),
            extra_flags=("--spec-type", "ngram-mod", "-md", str(draft)),
            extra_env={"LLAMA_ARG_SPEC_TYPE": "draft-mtp"},
        )  # fmt: skip
        with pytest.raises(ServerStartError, match=r"needs 62\.0 GiB"):
            await mixed.start()
        assert [c[2] for c in calls] == [model]

        # a head that is not there gives no projection
        gone = head.model_copy(update={"spec_draft_file": str(tmp_path / "gone.gguf")})
        missing = _manager(tmp_path, http, guards, _steady(100), settings=_answer(**gone.model_dump()))
        with pytest.raises(ServerStartError, match="the draft file cannot be read"):
            await missing.start()

        # a draft model of its own gets its own projection beside the model's, at the same `-ub`
        calls.clear()
        alone = _manager(
            tmp_path, http, guards, _steady(100), health=1, settings=_answer(**simple.model_dump())
        )
        await alone.start()
        await alone.stop()
        assert [c[2] for c in calls[:2]] == [model, str(draft)]
        assert all(c[-4:] == ["-ub", "2048", "-lv", "4"] for c in calls[:2])


async def test_a_stamp_says_what_the_answer_gave_and_what_the_callers_flags_or_the_served_file_set_aside(
    tmp_path: Path,
) -> None:
    def applied_and_overridden(manager: LocalServerManager) -> tuple[Settings, list[str]]:
        settings = manager.stamp().settings
        assert settings is not None
        return settings.applied, settings.overridden

    def with_answer(chosen: dict[str, Any], **extra: Any) -> LocalServerManager:
        return _manager(tmp_path, http, MemoryGuards(), _steady(100), settings=_answer(**chosen), **extra)

    async with httpx.AsyncClient() as http:
        both = {"n_ubatch": 2048, "spec": SpecType.DRAFT_MTP}
        assert applied_and_overridden(with_answer(both)) == (Settings(**both), [])
        # a smaller batch size of the caller's caps `-ub`: not passed, not stamped
        capped = with_answer({"n_ubatch": 2048}, extra_flags=("-b", "512"))
        assert applied_and_overridden(capped) == (Settings(), ["n_ubatch"])
        # `none` is a choice: applied when it took effect, overridden when the caller's flags replaced it
        none = {"spec": SpecType.NONE}
        assert applied_and_overridden(with_answer(none)) == (Settings(spec=SpecType.NONE), [])
        theirs = with_answer(none, extra_flags=("--spec-type", "ngram-mod"))
        assert applied_and_overridden(theirs) == (Settings(), ["spec"])
        # a quantised file the manager does not serve is set aside; the one it serves is not
        other, same = (PinnedFile(path=name, sha256="ab" * 32) for name in ("other.gguf", "test.gguf"))
        assert applied_and_overridden(with_answer({"model_file": (other,)})) == (Settings(), ["model_file"])
        assert applied_and_overridden(with_answer({"model_file": (same,)})) == (Settings(), [])
        # this server runs one answer: it is not stamped with another, and takes no other engine's
        manager = with_answer(both)
        manager.stamp(settings=_answer(**both))
        with pytest.raises(ValueError, match="built with"):
            manager.stamp(settings=_answer(n_ubatch=1))
        python = _answer(**both).model_copy(update={"engine": Engine.PYTORCH})
        with pytest.raises(ValueError, match=r"llama\.cpp"):
            _manager(tmp_path, http, MemoryGuards(), _steady(100), settings=python)
