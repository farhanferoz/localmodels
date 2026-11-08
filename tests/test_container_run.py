"""`container_run`: start refusals, the recipe and stop paths, the guard and the reaper. No Docker or GPU."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from localmodels import container_run
from localmodels.container_run import ContainerRun, ContainerStartError, Reaper, RunKind
from localmodels.engines import GateVerdict, Publisher, VersionRecord, write_record
from localmodels.launcher import MemoryGuards, NotEnoughMemoryError, ServerStartError
from localmodels.stack import Engine

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.running: set[str] = set()

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        cmd = list(command)
        self.calls.append(cmd)
        if cmd[1] == "ps":
            label = cmd[cmd.index("--filter") + 1]
            ids = [cid for cid in self.running if label.endswith(cid)]
            return subprocess.CompletedProcess(cmd, 0, "\n".join(ids), "")
        if cmd[1] == "rm":
            self.running.discard(cmd[-1])
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
    root = tmp_path / "hf"
    (root / "hub" / "models--nvidia--Qwen3-30B-A3B-NVFP4" / "snapshots").mkdir(parents=True, exist_ok=True)
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


def test_a_start_is_refused_with_its_remedy(tmp_path: Path) -> None:
    docker = FakeDocker()
    cases = [
        (record(pulled=False), {}, ContainerStartError, "engines pull"),
        (record(), {"recipe": {"pre_exec": ["curl x | sh"]}}, ContainerStartError, "pre_exec"),
        (record(measured=None), {}, ContainerStartError, "no measured run"),
        (record(measured=95.0), {}, NotEnoughMemoryError, "needs 105.0 GiB"),
        (record(), {"free_gib": lambda: None}, ServerStartError, "cannot be read"),
    ]
    for rec, kw, error, words in cases:
        with pytest.raises(error, match=words), make_run(tmp_path, rec, docker, **kw):
            pass
    # a bootstrap run of a never-measured engine uses D16's seed instead of refusing
    seeded = make_run(tmp_path, record(measured=None), docker, kind=RunKind.BOOTSTRAP)
    assert seeded.need_gib() == container_run.SEED_NEED_GIB[Engine.SGLANG]


def test_a_run_starts_offline_read_only_labelled_and_always_stops(tmp_path: Path) -> None:
    docker = FakeDocker()
    with make_run(tmp_path, record(), docker) as run:
        body = json.loads(next(tmp_path.glob("localmodels-recipe-*.json")).read_text(encoding="utf-8"))
        real = tmp_path / "hf" / "hub" / "models--nvidia--Qwen3-30B-A3B-NVFP4"
        assert body["container"] == IMAGE
        assert body["model"] == MODEL
        assert f"{real}:{real}:ro" in body["executor_config"]["volumes"]
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
    assert " stop " in log[1]
    assert ["docker", "rm", "-f", "run0001"] in docker.calls
    assert not root.exists()

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


def test_the_guard_kills_the_container_below_the_floor(tmp_path: Path) -> None:
    docker = FakeDocker()
    readings = iter([100.0] * 50)  # the settle wait, the start and a while serving; then the fall
    guards = MemoryGuards(kill_below_gib=8.0, server_oom_score_adj=900)
    run = make_run(tmp_path, record(), docker, guards=guards, free_gib=lambda: next(readings, 5.0))
    with run:
        run._guard.join(timeout=2)  # type: ignore[union-attr]
        assert ["docker", "kill", "run0001"] in docker.calls
        assert "below the 8 GiB kill floor" in (run._killed or "")


def test_the_reaper_removes_only_containers_of_jobs_no_longer_running() -> None:
    created = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    boxes = {"a" * 12: "10", "b" * 12: "11", "c" * 12: "12", "d" * 12: "13"}
    ps = "\n".join(f"{cid}|{job}|{created:%Y-%m-%d %H:%M:%S} +0000 UTC" for cid, job in boxes.items())
    before, after = created - timedelta(minutes=5), created + timedelta(minutes=5)
    listing = f"10|RUNNING|{before:%Y-%m-%dT%H:%M:%S}\n12|PENDING|N/A\n13|RUNNING|{after:%Y-%m-%dT%H:%M:%S}\n"
    removed: list[str] = []

    def docker(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "rm":
            removed.append(command[-1][0])
        return subprocess.CompletedProcess(command, 0, ps if command[1] == "ps" else "", "")

    def squeue(answer: int) -> container_run.Runner:
        return lambda c: subprocess.CompletedProcess(c, answer, listing if answer == 0 else "", "timeout")

    report = Reaper(docker=docker, squeue=squeue(0)).run()
    assert sorted(removed) == [
        "b",
        "c",
        "d",
    ]  # ended (forgotten), requeued and waiting, started again after it
    assert [line[:12] for line in report.kept] == ["a" * 12]
    removed.clear()
    failed = Reaper(docker=docker, squeue=squeue(1)).run()
    assert (removed, failed.reason) == ([], "squeue did not answer: nothing removed")
