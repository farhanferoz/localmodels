"""The launcher's memory guards, with fake memory readings and a fake server; no GPU, no model."""

from __future__ import annotations

import asyncio
import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from localmodels import launcher
from localmodels.launcher import LocalServerManager, MemoryGuards, ServerStartError
from localmodels.lease import GpuLease
from localmodels.pinned import PinnedFile
from localmodels.process import ServerSpec, build_args
from localmodels.settings import MeasuredPeak, RecordRef
from localmodels.stack import Engine, LlamaLayout, Settings, SpecType
from tests.test_launcher import _answer, _cpu_params, _gpu_params, _make_fake_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from localmodels.params import ServingParams

PORT = 8187
PROJECTED_GIB = 20  # what the fake llama-fit-params prints (20480 MiB, on the GPU)


def _fit_params(bin_path: Path, output: str = "CUDA0 20000 400 80\nHost 0 0 0") -> None:
    script = bin_path.with_name("llama-fit-params")
    script.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _manager(
    tmp_path: Path,
    http: httpx.AsyncClient,
    guards: MemoryGuards,
    readings: Iterator[float],
    *,
    health: int = 3,
    params: ServingParams | None = None,
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
        params or _gpu_params(),
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


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every process the code under test starts, as argument lists: llama-fit-params runs, then the server."""
    started: list[list[str]] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        started.append([str(a) for a in args])
        return await real(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    return started


async def test_load_refused_when_projection_plus_reserve_exceeds_available(tmp_path: Path) -> None:
    guards = MemoryGuards(host_reserve_gib=12)
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, _steady(PROJECTED_GIB + 11))
        with pytest.raises(ServerStartError, match=r"needs 32\.0 GiB .* 31\.0 GiB is available"):
            await manager.start()
        assert not (tmp_path / "pids" / f"{PORT}.pid").exists()


@pytest.mark.parametrize(
    ("shares", "cpu", "counted"),
    [
        pytest.param(True, False, 30, id="shared-memory-gpu-counts-gpu-and-host"),
        pytest.param(False, False, 20, id="gpu-with-own-memory-counts-the-gpu"),
        pytest.param(False, True, 10, id="no-gpu-counts-the-host"),
    ],
)
async def test_the_projection_counts_the_memory_the_server_draws_from(
    tmp_path: Path,
    calls: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    shares: bool,
    cpu: bool,
    counted: int,
) -> None:
    # llama.cpp put Flash-Next 62 GiB on the GPU and 28 GiB on the host; the GPU line alone let it start
    monkeypatch.setattr(launcher, "_gpu_shares_host_ram", lambda: shares)
    # a caller's own load mode keeps the host-side weights counted (D30 adds `-lm mmap` only without one)
    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path, http, MemoryGuards(host_reserve_gib=12), _steady(counted + 11),
            params=_cpu_params() if cpu else None, extra_flags=("-lm", "none"),
        )  # fmt: skip
        _fit_params(Path(manager.binary), "CUDA0 20000 400 80\nHost 10000 200 40")
        with pytest.raises(ServerStartError, match=rf"needs {counted + 12}\.0 GiB"):
            await manager.start()
        assert calls[0][3:5] == ["-ngl", "0" if cpu else "99"]
        # a GPU server whose projection has no GPU line was projected for another device: refused
        _fit_params(Path(manager.binary), "Host 10000 200 40")
        if not shares and not cpu:
            with pytest.raises(ServerStartError, match=r"projected test\.gguf for no GPU"):
                await manager.start()


async def test_a_peak_its_tune_measured_at_the_same_layout_raises_the_need(tmp_path: Path) -> None:
    def tuned(n_ctx: int) -> Any:
        answer = _answer(spec=SpecType.NONE)
        answer.record = RecordRef(
            path="r.json", measured_at=datetime.now(UTC), hardware="h", candidate="reference", quality="",
            stack=None, build=None, recipe_project=None,
            peak=MeasuredPeak(gib=PROJECTED_GIB + 9, layout=LlamaLayout(n_ctx=n_ctx, n_parallel=1)),
        )  # fmt: skip
        return answer

    guards = MemoryGuards(host_reserve_gib=12)
    async with httpx.AsyncClient() as http:
        same = _manager(tmp_path, http, guards, _steady(PROJECTED_GIB + 20), settings=tuned(8192))
        with pytest.raises(
            ServerStartError, match=r"needs 41\.0 GiB"
        ):  # the 29 GiB peak, not the 20 projected
            await same.start()
        # another context: the peak is not this server's, so the projection decides and it starts
        other = _manager(tmp_path, http, guards, _steady(PROJECTED_GIB + 20), health=1, settings=tuned(4096))
        await other.start()
        await other.stop()


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
    tmp_path: Path, calls: list[list[str]]
) -> None:
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
        assert fit[1:] == ["-m", model, "-ngl", "99", "-c", "8192", "-np", "1", "--fit-print", "on"]
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
        assert (fit[2], fit[-4:]) == (model, ["-ub", "2048", "--fit-print", "on"])
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
        assert calls[0][-4:] == ["-ub", "256", "--fit-print", "on"]
        assert calls[1] == build_args(kept.binary, kept.spec, kept.params, theirs)
        stamped = kept.stamp()
        assert stamped.settings is not None
        assert (stamped.settings.applied, stamped.settings.overridden) == (Settings(), ["n_ubatch", "spec"])


async def test_a_draft_model_is_projected_on_its_own_and_an_mtp_head_counts_as_its_size(
    tmp_path: Path, calls: list[list[str]]
) -> None:
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
        assert all(c[-4:] == ["-ub", "2048", "--fit-print", "on"] for c in calls[:2])


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


@dataclass(frozen=True)
class _CacheCase:
    """One server start: what it runs on, what the caller set, and the cap llama-fit-params' figure gives."""

    shares: bool | None = True  # the GPU's memory is the host's RAM; None: cannot be told
    cpu: bool = False
    guards: MemoryGuards = field(default_factory=MemoryGuards)
    own: dict[str, Any] = field(default_factory=dict)  # the caller's `extra_flags` / `extra_env`
    available: float = 100  # GiB, as the server starts
    fit_output: str | None = None  # what llama-fit-params prints, if not a 20 GiB projection
    cap: tuple[str, ...] = ()  # the `--cache-ram` flags expected
    fit_runs: int = 0
    refuses: str | None = None  # a pattern the refusal's message must match, when the start is refused
    peak: tuple[float, str] | None = None  # a tune's measured peak (GiB, load mode), when the answer has one
    ask_mib: int | None = None  # the slurm job's --mem (SLURM_MEM_PER_NODE), when the start runs in one


# D30: 60 GiB projected, 20 GiB of it host weights (the Host row's model column); the reserve is 10 GiB, so
# the model needs 70 GiB with them in memory and 50 GiB without, and the cap is available - need - 10 GiB
_HOST_HEAVY = "CUDA0 40960 0 0\nHost 20480 0 0"
_RESERVE = MemoryGuards(host_reserve_gib=10)
_MMAP = ("-lm", "mmap")
_COUNTED = r"needs 70\.0 GiB .* 55\.0 GiB is available; not loading it"

# the projection is 20 GiB, so a cap is (available - 20 - margin) GiB, between 0 and llama.cpp's 8,192 MiB
_CACHE_CASES = [
    pytest.param(
        _CacheCase(guards=MemoryGuards(host_reserve_gib=10), cap=("--cache-ram", "8192"), fit_runs=1),
        id="clamped-to-8192",
    ),
    pytest.param(
        _CacheCase(
            guards=MemoryGuards(kill_below_gib=10), available=34, cap=("--cache-ram", "4096"), fit_runs=1
        ),
        id="computed",
    ),
    pytest.param(
        _CacheCase(
            guards=MemoryGuards(kill_below_gib=10), available=25, cap=("--cache-ram", "0"), fit_runs=1
        ),
        id="clamped-to-0",
    ),
    # both guards: the reserve stays free above the kill floor, for what the projection misses (D22)
    pytest.param(
        _CacheCase(
            guards=MemoryGuards(host_reserve_gib=10, kill_below_gib=10),
            available=44,
            cap=("--cache-ram", "4096"),
            fit_runs=1,
        ),
        id="reserve-above-the-floor",
    ),
    # the memory kind cannot be read: capped like a shared one, never left at llama.cpp's 8 GiB
    pytest.param(_CacheCase(shares=None, cap=("--cache-ram", "0")), id="kind-unknown"),
    pytest.param(
        _CacheCase(fit_output="nothing useful", cap=("--cache-ram", "0"), fit_runs=1), id="no-projection"
    ),
    # the caller's own flag or environment: nothing is added, and nothing is projected for it
    pytest.param(_CacheCase(own={"extra_flags": ("-cram", "1")}), id="own-flag"),
    pytest.param(_CacheCase(own={"extra_env": {"LLAMA_ARG_CACHE_RAM": "1"}}), id="own-env"),
    # a GPU with its own memory, or no GPU at all: llama.cpp's default stays
    pytest.param(_CacheCase(shares=False), id="gpu-with-own-memory"),
    pytest.param(_CacheCase(cpu=True), id="cpu"),
    # D30: with the host weights counted it fits, so nothing changes (cap clamped to 8,192 MiB)
    pytest.param(
        _CacheCase(
            guards=_RESERVE, fit_output=_HOST_HEAVY, available=80, cap=("--cache-ram", "8192"), fit_runs=1
        ),
        id="fits-with-host-weights-no-mmap",
    ),
    # the machine has room (70 <= 80) but the slurm job asked for 55 GiB: over its ask with the host weights
    # counted, so `-lm mmap` (preflight's guard stops a job over its ask, not one over the machine); the cap
    # comes from the ask too: 55 - 40 - 10 = 5 GiB, not the machine's 80 - 40 - 10 = 30 (clamped to 8 GiB)
    pytest.param(
        _CacheCase(
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=80,
            ask_mib=55 * 1024,
            cap=(*_MMAP, "--cache-ram", "5120"),
            fit_runs=1,
        ),
        id="over-the-slurm-ask-mmap",
    ),
    # it does not fit with them (70 > 55) but fits without (50 <= 55): `-lm mmap`, cap from the 40 GiB need
    pytest.param(
        _CacheCase(
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=55,
            cap=(*_MMAP, "--cache-ram", "5120"),
            fit_runs=1,
        ),
        id="fits-only-without-host-weights-mmap",
    ),
    # not even without them (50 > 45): refused, and the message gives both figures
    pytest.param(
        _CacheCase(
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=45,
            fit_runs=1,
            refuses=(
                r"needs 70\.0 GiB .* 45\.0 GiB is available \(with -lm mmap, which leaves out 20\.0 GiB "
                r"of host weights, it would need 50\.0 GiB\)"
            ),
        ),
        id="does-not-fit-even-with-mmap",
    ),
    # a caller's own load mode wins: `mmap` leaves the host weights out of the need, any other counts them
    pytest.param(
        _CacheCase(
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=55,
            own={"extra_flags": _MMAP},
            cap=("--cache-ram", "5120"),
            fit_runs=1,
        ),
        id="caller-mmap",
    ),
    pytest.param(
        _CacheCase(
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=55,
            own={"extra_env": {"LLAMA_ARG_LOAD_MODE": "none"}},
            fit_runs=1,
            refuses=_COUNTED,
        ),
        id="caller-none-env",
    ),
    pytest.param(
        _CacheCase(
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=55,
            own={"extra_flags": ("--load-mode=none",)},
            fit_runs=1,
            refuses=_COUNTED,
        ),
        id="caller-none-flag",
    ),
    # a GPU with its own memory counts the GPU rows (50 GiB with the reserve) and never gets `-lm mmap`
    pytest.param(
        _CacheCase(shares=False, guards=_RESERVE, fit_output=_HOST_HEAVY, available=55, fit_runs=1),
        id="own-memory-fits",
    ),
    pytest.param(
        _CacheCase(
            shares=False,
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=45,
            fit_runs=1,
            refuses=r"^(?!.*-lm mmap).*needs 50\.0 GiB .* 45\.0 GiB is available",
        ),
        id="own-memory-refused-without-mmap-note",
    ),
    # a peak measured at the default load mode (65 GiB) is not this start's: mmap goes by the projection
    pytest.param(
        _CacheCase(
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=55,
            peak=(65, ""),
            cap=(*_MMAP, "--cache-ram", "5120"),
            fit_runs=1,
        ),
        id="default-mode-peak-not-used-for-mmap",
    ),
    # one measured at mmap is (52 GiB, over the 40 projected): 62 > 55, refused with that figure
    pytest.param(
        _CacheCase(
            guards=_RESERVE,
            fit_output=_HOST_HEAVY,
            available=55,
            peak=(52, "mmap"),
            fit_runs=1,
            refuses=r"needs 70\.0 GiB .* it would need 62\.0 GiB\)",
        ),
        id="mmap-peak-used-for-mmap",
    ),
]


@pytest.mark.parametrize("case", _CACHE_CASES)
async def test_the_prompt_cache_and_the_load_mode_follow_the_memory_the_model_leaves(
    tmp_path: Path, calls: list[list[str]], monkeypatch: pytest.MonkeyPatch, case: _CacheCase
) -> None:
    monkeypatch.setattr(launcher, "_gpu_shares_host_ram", lambda: case.shares)
    monkeypatch.delenv(launcher.SLURM_MEM_ENV, raising=False)  # a test run inside a slurm job keeps its own
    if case.ask_mib is not None:
        monkeypatch.setenv(launcher.SLURM_MEM_ENV, str(case.ask_mib))
    answer = _answer(
        spec=SpecType.NONE
    )  # an answer that adds no flag: adopting a server compares its arguments
    if case.peak is not None:
        gib, load_mode = case.peak
        answer.record = RecordRef(
            path="r.json", measured_at=datetime.now(UTC), hardware="h", candidate="reference", quality="",
            stack=None, build=None, recipe_project=None,
            peak=MeasuredPeak(gib=gib, layout=LlamaLayout(n_ctx=8192, n_parallel=1, load_mode=load_mode)),
        )  # fmt: skip

    def manager(available: float, **own: Any) -> LocalServerManager:
        made = _manager(
            tmp_path, http, case.guards, _steady(available), health=1, settings=answer,
            params=_cpu_params() if case.cpu else None, **own,
        )  # fmt: skip
        if case.fit_output is not None:
            _fit_params(Path(made.binary), case.fit_output)
        return made

    async with httpx.AsyncClient() as http:
        first = manager(case.available, **case.own)
        if case.refuses is not None:
            with pytest.raises(ServerStartError, match=case.refuses):
                await first.start()
            return
        try:
            started = await first.start()
            flags = (*case.own.get("extra_flags", ()), *case.cap)
            assert [c[0].endswith("llama-fit-params") for c in calls].count(True) == case.fit_runs
            assert calls[-1] == build_args(first.binary, first.spec, first.params, flags)

            # the cap follows the memory at the time, so it is no part of a server's identity: a manager
            # whose own cap would differ adopts the running server, stamps its flags (`-lm mmap` too: D30)
            # and projects nothing
            calls.clear()
            later = manager(case.available + 50, **case.own)
            assert (await later.start()).pid == started.pid
            assert not calls
            for stamped in (first.stamp().layout, later.stamp().layout):
                assert stamped is not None
                assert stamped.extra_flags == flags

            # a caller's own value is identity: another one restarts the server
            if "extra_flags" in case.own:
                other = manager(case.available, extra_flags=("-cram", "2"))
                assert (await other.start()).pid != started.pid
                await other.stop()
        finally:
            await first.stop()
