"""`container_run`: start refusals, the recipe and stop paths, the guard and the reaper. No Docker or GPU."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from localmodels import container_run
from localmodels.container_run import ContainerRun, ContainerStartError, MemoryNeed, Reaper, RunKind
from localmodels.engines import GateVerdict, Publisher, VersionRecord, write_record
from localmodels.launcher import MemoryGuards, NotEnoughMemoryError, ServerStartError, kill_below_floor
from localmodels.stack import Engine

if TYPE_CHECKING:
    from collections.abc import Sequence

IMAGE = "lmsysorg/sglang@sha256:" + "cd" * 32
MODEL = "nvidia/Qwen3-30B-A3B-NVFP4"


@pytest.fixture(autouse=True)
def fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALMODELS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setattr(container_run, "POLL_S", 0.01)
    monkeypatch.setattr(container_run, "SETTLE_LIMIT_S", 0.05)


def record(*, pulled: bool = True, measured: float | None = 30.0) -> VersionRecord:
    results = [{"model": MODEL, "memory": {"fall_gib": measured}}] if measured is not None else []
    made = VersionRecord(
        engine=Engine.SGLANG,
        tag="v0.5.23-cu130",
        image=IMAGE,
        publisher=Publisher.OFFICIAL,
        gate=GateVerdict.PASSED,
        pulled=pulled,
        results=results,
    )
    write_record(made)
    return made


class FakeDocker:
    """Containers by run label, removed by ``rm -f``; every command kept."""

    def __init__(self, images: frozenset[str] = frozenset({IMAGE})) -> None:
        self.calls: list[list[str]] = []
        self.running: set[str] = set()
        self.images = images

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        cmd = list(command)
        self.calls.append(cmd)
        if cmd[1] == "ps":
            label = cmd[cmd.index("--filter") + 1]
            ids = [cid for cid in self.running if label.endswith(cid)]
            return subprocess.CompletedProcess(cmd, 0, "\n".join(ids), "")
        if cmd[1] == "rm":
            self.running.discard(cmd[-1])
        if cmd[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0 if cmd[3] in self.images else 1, "", "No such image")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def fake_sparkrun(tmp_path: Path, docker: FakeDocker, run_id: str, *, fail: bool = False) -> Path:
    """A sparkrun that logs its argv and HF_HOME; ``run`` starts the run's container in ``docker``."""
    log = tmp_path / "sparkrun.log"
    script = tmp_path / "sparkrun"
    script.write_text(
        f'#!/bin/sh\necho "$HF_HOME $SPARKRUN_NO_TELEMETRY $*" >> {log}\nexit {1 if fail else 0}\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    docker.running.add(run_id)  # the container a successful `run` makes
    return script


def hub(tmp_path: Path) -> Path:
    """A Hub root whose model links, as on this machine, into a blob store shared by several models."""
    root = tmp_path / "hf"
    model = root / "hub" / "models--nvidia--Qwen3-30B-A3B-NVFP4"
    shared = root / "hub" / "blobs" / "1d" / "1d20"
    for folder in (model / "snapshots" / "rev", model / "blobs", shared.parent):
        folder.mkdir(parents=True, exist_ok=True)
    shared.write_text("weights", encoding="utf-8")
    for link, target in (
        (model / "blobs" / "00fe", "../../blobs/1d/1d20"),
        (model / "snapshots" / "rev" / "w", "../../blobs/00fe"),
    ):
        if not link.is_symlink():
            link.symlink_to(target)
    return root


def make_run(tmp_path: Path, rec: VersionRecord, docker: FakeDocker, **kw: object) -> ContainerRun:
    run_id = "run0001"
    defaults: dict[str, object] = {
        "record": rec,
        "model": MODEL,
        "recipe": {"runtime": "sglang", "defaults": {"port": 8199}},
        "sparkrun": fake_sparkrun(tmp_path, docker, run_id),
        "guards": MemoryGuards(host_reserve_gib=10.0),
        "hub_root": hub(tmp_path),
        "work_dir": tmp_path,
        "docker": docker,
        "free_gib": lambda: 100.0,
        "run_id": run_id,
        "reaper": Reaper(docker=docker, squeue=lambda c: subprocess.CompletedProcess(c, 0, "", "")),
    }
    return ContainerRun(**{**defaults, **kw})  # type: ignore[arg-type]


def test_a_start_is_refused_with_its_remedy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docker = FakeDocker()
    cases = [
        (record(pulled=False), {}, ContainerStartError, "engines pull"),
        (record(), {"docker": FakeDocker(images=frozenset())}, ContainerStartError, "engines pull"),
        (record(), {"recipe": {"pre_exec": ["curl x | sh"]}}, ContainerStartError, "pre_exec"),
        (record(measured=None), {}, ContainerStartError, "no measured run"),
        (record(measured=95.0), {}, NotEnoughMemoryError, "needs 105.0 GiB"),
        (record(), {"free_gib": lambda: None}, ServerStartError, "cannot be read"),
    ]
    for rec, kw, error, words in cases:
        with pytest.raises(error, match=words), make_run(tmp_path, rec, kw.pop("docker", docker), **kw):
            pass
    monkeypatch.delenv("SLURM_JOB_ID")  # a container outside a job would be the reaper's at once
    with (
        pytest.raises(ContainerStartError, match="only inside a slurm job"),
        make_run(tmp_path, record(), docker),
    ):
        pass
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    extra = record().model_copy(
        update={"results": [{"model": MODEL, "memory": {"fall_gib": 30.0, "new": 1}}]}
    )
    assert MemoryNeed.recorded(extra, MODEL) == 30.0  # a field written by a later version is ignored
    # a bootstrap run of a never-measured engine uses D16's seed instead of refusing
    seeded = make_run(tmp_path, record(measured=None), docker, kind=RunKind.BOOTSTRAP)
    assert seeded.need_gib() == container_run.SEED_NEED_GIB[Engine.SGLANG]


def test_a_run_starts_offline_read_only_labelled_and_always_stops(  # noqa: PLR0915 - every stop path
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="localmodels.container_run")
    docker = FakeDocker()
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.peak").write_text(str(2 * container_run.GIB), encoding="utf-8")
    inside = subprocess.Popen(["sleep", "30"])  # stands in for the container's process
    (cgroup / "cgroup.procs").write_text(f"{inside.pid}\n", encoding="utf-8")
    monkeypatch.setattr(ContainerRun, "_cgroup", lambda _self, _cid: cgroup)
    guards = MemoryGuards(host_reserve_gib=10.0, server_oom_score_adj=900)  # a score, no kill floor
    with make_run(tmp_path, record(), docker, guards=guards) as run:
        for _ in range(200):
            if Path(f"/proc/{inside.pid}/oom_score_adj").read_text(encoding="utf-8").strip() == "900":
                break
            run._stopped.wait(0.01)
    inside.kill()
    assert Path(f"/proc/{inside.pid}/oom_score_adj").read_text(encoding="utf-8").strip() == "900"
    assert "OOM score 900 set on the container's 1 processes" in caplog.text  # the allow path is logged
    inside.wait()
    with make_run(tmp_path, record(), docker) as run:
        body = json.loads(next(tmp_path.glob("localmodels-recipe-*.json")).read_text(encoding="utf-8"))
        real = tmp_path / "hf" / "hub" / "models--nvidia--Qwen3-30B-A3B-NVFP4"
        assert body["container"] == IMAGE
        assert body["model"] == MODEL
        shared = tmp_path / "hf" / "hub" / "blobs" / "1d" / "1d20"
        assert body["executor_config"]["volumes"] == [
            f"{real}:{real}:ro",
            f"{shared}:{shared}:ro",
        ]  # links resolve
        assert set(body["executor_config"]["labels"]) == {
            "lifecycle=ephemeral",
            "slurm-job=4242",
            "localmodels.run=run0001",
        }
        root = tmp_path / "localmodels-hub-run0001"
        assert (root / "hub" / real.name).resolve() == real  # the per-run Hub root only links the model
        assert run.stamp()["image"] == IMAGE
    log = (tmp_path / "sparkrun.log").read_text(encoding="utf-8").splitlines()
    assert log[0].startswith(f"{root} 1 run ")  # the per-run Hub root and telemetry off
    assert "--offline" in log[0]
    assert "--memory-limit 30G" in log[0]  # the whole need caps the container
    assert " stop " in log[1]
    assert ["docker", "rm", "-f", "run0001"] in docker.calls
    assert not root.exists()
    # sampled without a kill floor, and the cgroup's peak kept from before the container went
    assert run.memory() == MemoryNeed(fall_gib=0.0, host_peak_gib=2.0)

    docker = FakeDocker()  # an interrupt during the start still stops what it began
    real_run = container_run.run_sparkrun

    def interrupted(binary: Path, args: Sequence[str], **kw: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "run":
            raise KeyboardInterrupt
        return real_run(binary, args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(container_run, "run_sparkrun", interrupted)
    with pytest.raises(KeyboardInterrupt), make_run(tmp_path, record(), docker):
        pass
    monkeypatch.setattr(container_run, "run_sparkrun", real_run)
    assert ["docker", "rm", "-f", "run0001"] in docker.calls
    assert not root.exists()
    blocked = tmp_path / "not-executable"  # a binary that cannot run is an exit status: the stop completes
    blocked.write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(ContainerStartError, match="exited 126"), make_run(tmp_path, record(), FakeDocker(),
                                                                          sparkrun=blocked):  # fmt: skip
        pass

    docker = FakeDocker()  # an exception in the block still stops it
    with pytest.raises(RuntimeError), make_run(tmp_path, record(), docker):
        raise RuntimeError
    assert ["docker", "rm", "-f", "run0001"] in docker.calls

    docker, seen = FakeDocker(), []  # SIGTERM stops it, then the caller's own handler runs
    previous = signal.signal(signal.SIGTERM, lambda *_: seen.append("caller"))
    try:
        with make_run(tmp_path, record(), docker):
            os.kill(os.getpid(), signal.SIGTERM)
            assert ["docker", "rm", "-f", "run0001"] in docker.calls
        assert seen == ["caller"]
    finally:
        signal.signal(signal.SIGTERM, previous)

    ended: list[int] = []  # a handler not set from Python (None): SIGTERM still ends the process
    monkeypatch.setattr(container_run.signal, "getsignal", lambda _s: None)
    monkeypatch.setattr(container_run.os, "kill", lambda _pid, sig: ended.append(sig))
    try:
        with make_run(tmp_path, record(), FakeDocker()):
            signal.raise_signal(signal.SIGTERM)
        assert ended == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_the_guard_kills_the_container_below_the_floor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    docker = FakeDocker()
    readings = iter([100.0] * 50)  # the settle wait, the start and a while serving; then the fall
    guards = MemoryGuards(kill_below_gib=8.0, server_oom_score_adj=900)
    run = make_run(tmp_path, record(), docker, guards=guards, free_gib=lambda: next(readings, 5.0))
    caplog.set_level(logging.INFO, logger="localmodels.container_run")
    with run:
        for _ in range(200):
            if docker.calls.count(["docker", "kill", "run0001"]) >= 2:  # and again after it
                break
            run._stopped.wait(0.01)
        assert "below the 8 GiB kill floor" in (run._killed or "")
        # it keeps killing what the run makes until it is stopped (sparkrun may still be starting one)
        assert docker.calls.count(["docker", "kill", "run0001"]) >= 2
    assert "OOM score 900 not set" in caplog.text  # the cgroup was unreadable: said, not swallowed
    polls: list[int] = []
    asyncio.run(
        kill_below_floor(
            0, 8.0, lambda: 0, lambda: 5.0, "x", alive=lambda: True, kill=lambda: None,
            on_poll=lambda: polls.append(1),
        )
    )  # fmt: skip
    assert polls == []  # the floor is decided before the poll's own (possibly slow) work


def test_the_reaper_removes_only_containers_of_jobs_no_longer_running() -> None:
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    boxes = {"a" * 12: "10", "b" * 12: "11", "c" * 12: "12", "d" * 12: "13", "e" * 12: "none"}
    ps = "\n".join(f"{cid}|{job}|{created:%Y-%m-%d %H:%M:%S} +0000 UTC" for cid, job in boxes.items())
    before, after = created - timedelta(minutes=5), created + timedelta(minutes=5)
    listing = f"10|RUNNING|{before:%Y-%m-%dT%H:%M:%S}\n12|PENDING|N/A\n13|RUNNING|{after:%Y-%m-%dT%H:%M:%S}\n"
    removed: list[str] = []

    def docker(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "rm":
            removed.append(command[-1][0])
        return subprocess.CompletedProcess(command, 0, ps if command[1] == "ps" else "", "")

    asked: list[list[str]] = []

    def squeue(answer: int) -> container_run.Runner:
        def run(c: Sequence[str]) -> subprocess.CompletedProcess[str]:
            asked.append(list(c))
            return subprocess.CompletedProcess(c, answer, listing if answer == 0 else "", "timeout")

        return run

    report = Reaper(docker=docker, squeue=squeue(0)).run()
    assert sorted(removed) == [
        "b",
        "c",
        "d",
    ]  # ended (forgotten), requeued and waiting, started again after it
    assert [line[:12] for line in report.kept] == ["a" * 12, "e" * 12]  # e: not a job id, never guessed
    assert asked[0][-1] == "%A|%T|%S"  # %A: an array task's own id, as its label carries it
    removed.clear()
    failed = Reaper(docker=docker, squeue=squeue(1)).run()
    assert (removed, failed.reason) == ([], "squeue did not answer: nothing removed")
