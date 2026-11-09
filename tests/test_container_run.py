"""`container_run`: start refusals, the recipe and stop paths, the guard and the reaper. No Docker or GPU."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import pytest

from localmodels import container_run, engine_refresh, loopback
from localmodels.container_run import ContainerRun, ContainerStartError, MemoryNeed, Reaper, RunKind
from localmodels.engine_choice import ChoiceRule, EngineChoice
from localmodels.engines import RUNNER_NOT_FOUND, GateVerdict, Publisher, VersionRecord, write_record
from localmodels.enums import DeviceType, ServingRole, WorkloadKind
from localmodels.launcher import MemoryGuards, NotEnoughMemoryError, ServerStartError, kill_below_floor
from localmodels.settings import ModelIdentity, Settings, SettingsAnswer, SettingsSource, SettingsStatus
from localmodels.stack import Engine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from conftest import LoginDouble

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
        if cmd[1] == "inspect":
            return subprocess.CompletedProcess(
                cmd, 0, '{"Status":"exited","ExitCode":137,"OOMKilled":true}', ""
            )
        if cmd[1] == "cp":  # the engine's serve log, copied out of the container
            Path(cmd[3]).write_text("engine line 1\nCUDA error: out of memory\n", encoding="utf-8")
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

    def no_login() -> None:
        msg = "a login with localmodels' key failed (255: Permission denied)"
        raise loopback.LoopbackLoginError(msg)

    def weighed() -> float:
        pytest.fail("memory was weighed before the login was checked")

    monkeypatch.setattr(loopback, "loopback_login", no_login)
    with (
        pytest.raises(ContainerStartError, match="Permission denied"),
        make_run(tmp_path, record(), docker, free_gib=weighed),
    ):
        pass
    refused = container_run.run_sparkrun(tmp_path / "sparkrun", ["stop"], timeout=5, logs_in=True)
    assert (refused.returncode, refused.stderr) == (
        1,
        "a login with localmodels' key failed (255: Permission denied)",
    )
    plain = container_run.run_sparkrun(
        tmp_path / "absent", ["setup"], timeout=5
    )  # a call that needs no login
    assert plain.returncode == RUNNER_NOT_FOUND  # it ran (and found no binary) without asking for the login
    extra = record().model_copy(
        update={"results": [{"model": MODEL, "memory": {"fall_gib": 30.0, "new": 1}}]}
    )
    assert MemoryNeed.recorded(extra, MODEL) == 30.0  # a field written by a later version is ignored
    # a bootstrap run of a never-measured engine uses D16's seed instead of refusing
    seeded = make_run(tmp_path, record(measured=None), docker, kind=RunKind.BOOTSTRAP)
    assert seeded.need_gib() == container_run.SEED_NEED_GIB[Engine.SGLANG]


def test_the_loopback_login_is_made_once_allowed_from_this_machine_only_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    answer = {"ssh": subprocess.CompletedProcess([], 0, "", "")}

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[0] == "ssh-keygen":
            return subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603 - a key in tmp
        return answer["ssh"]

    client = tmp_path / "client" / "ssh"  # stands in for OpenSSH's client: prints what it was given
    client.parent.mkdir()
    client.write_text('#!/bin/sh\necho "$@"\n', encoding="utf-8")
    client.chmod(0o755)
    monkeypatch.setenv("PATH", f"{client.parent}{os.pathsep}{os.environ['PATH']}")
    authorized = tmp_path / "home" / ".ssh" / "authorized_keys"
    authorized.parent.mkdir(parents=True)
    authorized.write_bytes(
        b"ssh-ed25519 AAAAlaptop caf\xe9"
    )  # someone's line: Latin-1, no newline at the end
    key = tmp_path / "state" / "ssh" / loopback.KEY_NAME
    login = loopback.LoopbackLogin(key=key, authorized_keys=authorized, runner=runner)
    login.ensure()
    login.ensure()  # the second changes nothing and checks again
    public = login.public.read_text(encoding="utf-8").strip()
    line = f'from="127.0.0.1,::1" {public}\n'.encode()
    assert authorized.read_bytes() == b"ssh-ed25519 AAAAlaptop caf\xe9\n" + line
    modes = [p.stat().st_mode & 0o777 for p in (authorized, key, key.parent, login.launcher)]
    assert modes == [0o600, 0o600, 0o700, 0o700]
    assert [c[0] for c in calls] == ["ssh-keygen", str(login.launcher), str(login.launcher)]
    # the check and sparkrun's own ``ssh`` take the same launcher: this key alone, ahead of the user's config
    given = subprocess.run(  # noqa: S603 - the launcher just written, in tmp_path
        [str(login.launcher), "localhost", "true"], check=True, capture_output=True, text=True
    )
    assert given.stdout.split() == [
        "-o",
        "IdentityAgent=none",
        "-o",
        "IdentitiesOnly=yes",
        "-i",
        str(key),
        "localhost",
        "true",
    ]
    assert login.env({"PATH": "/usr/bin"})["PATH"] == f"{login.launcher.parent}{os.pathsep}/usr/bin"
    # a public half lost in a copy is rebuilt; a commented-out copy of the line does not count as allowed
    login.public.unlink()
    authorized.write_bytes(b"# " + line)
    login.ensure()
    assert login.public.read_text(encoding="utf-8").split()[1] == public.split()[1]
    assert authorized.read_bytes() == b"# " + line + line
    # a key that is not this user's own private file, a public half that holds no key and a login that does
    # not work are each refused with their remedy
    key.chmod(0o644)
    with pytest.raises(loopback.LoopbackLoginError, match="not this user's own private file"):
        login.ensure()
    key.chmod(0o600)
    login.public.write_text("", encoding="utf-8")
    with pytest.raises(loopback.LoopbackLoginError, match="holds no public key"):
        login.ensure()
    login.public.unlink()
    answer["ssh"] = subprocess.CompletedProcess([], 255, "", "Permission denied (publickey)")
    with pytest.raises(loopback.LoopbackLoginError, match=r"Permission denied.*authorized_keys"):
        login.ensure()


def test_a_run_starts_offline_read_only_labelled_and_always_stops(  # noqa: PLR0915 - every stop path
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    no_real_loopback_login: LoginDouble,
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
    with make_run(tmp_path, record(), docker, recipe=engine_refresh.BASE_RECIPES[Engine.VLLM]) as run:
        body = json.loads(next(tmp_path.glob("localmodels-recipe-*.json")).read_text(encoding="utf-8"))
        assert body["executor_config"]["entrypoint"] == ""  # vLLM's image would eat sparkrun's command
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
    assert "--no-rm" in log[0]  # a container that dies at start stays to be read
    assert " stop " in log[1]
    assert ["docker", "rm", "-f", "run0001"] in docker.calls
    assert not root.exists()
    assert no_real_loopback_login.calls == 4  # two runs, each start and stop with the loopback login
    # sampled without a kill floor, and the cgroup's peak kept from before the container went
    assert run.memory() == MemoryNeed(fall_gib=0.0, host_peak_gib=2.0)
    # a dry run (the sparkrun move's check) launches nothing, leaves no file, and asks for -vvv when told:
    # only that level prints the docker run line the check compares
    make_run(tmp_path, record(), docker).dry_run(verbose=True)
    dry = (tmp_path / "sparkrun.log").read_text(encoding="utf-8").splitlines()[-1]
    assert " -vvv run " in dry
    assert dry.endswith("--dry-run")
    assert not root.exists()
    assert no_real_loopback_login.calls == 5  # the dry run too

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
    failed = FakeDocker()
    with pytest.raises(ContainerStartError, match="exited 126"), make_run(tmp_path, record(), failed,
                                                                          sparkrun=blocked):  # fmt: skip
        pass
    # a failed start logs why its container failed, before the stop removes it
    kinds = [c[1] for c in failed.calls if c[1] in {"inspect", "cp", "rm"}]
    assert kinds[:3] == ["inspect", "cp", "rm"]
    assert '"OOMKilled":true' in caplog.text
    assert "CUDA error: out of memory" in caplog.text
    assert not list(tmp_path.glob("localmodels-serve-*"))

    class BrokenCopy(FakeDocker):
        def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            if command[1] == "cp":
                msg = "denied"
                raise PermissionError(msg)
            return super().__call__(command)

    broken = BrokenCopy()  # reading the output fails: the start's own error stands, and the stop still runs
    with pytest.raises(ContainerStartError, match="exited 126"), make_run(tmp_path, record(), broken,
                                                                          sparkrun=blocked):  # fmt: skip
        pass
    assert ["docker", "rm", "-f", "run0001"] in broken.calls

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


def test_gpu_peak_is_sampled_from_the_containers_processes_and_a_failed_call_is_no_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = FakeDocker()
    docker.running.add("run0001")
    procs = tmp_path / "cgroup"
    procs.mkdir()
    (procs / "cgroup.procs").write_text("111\n112\n", encoding="utf-8")
    readings: list[dict[int, int] | None] = [
        {111: 3327, 999: 40000},  # another process's memory is not ours
        None,  # nvidia-smi failed: no reading, the peak stays
        {111: 1024, 112: 1024},
        {111: 500},
    ]
    run = make_run(tmp_path, record(), docker, gpu_memory=lambda: readings.pop(0))
    monkeypatch.setattr(ContainerRun, "_cgroup", lambda _self, _cid: procs)
    assert run.memory().gpu_peak_gib is None  # nothing read yet: not zero
    for _ in range(4):
        run._sample_gpu()
    assert run.memory().gpu_peak_gib == pytest.approx(3327 / 1024)  # the largest, not the last


def test_the_stamp_carries_the_recipe_the_settings_and_the_cache_and_tuning_listings(tmp_path: Path) -> None:
    cache = tmp_path / "spark-cache"
    leaf = cache / "runtime-cache" / "sglang" / "img-1" / "model-2"
    (leaf / "inductor").mkdir(parents=True)
    (leaf / "inductor" / "k.bin").write_text("12345", encoding="utf-8")
    (leaf / ".sparkrun-last-used").touch()
    stale = cache / "runtime-cache" / "sglang" / "old" / ".sparkrun-last-used"
    stale.parent.mkdir(parents=True)
    stale.touch()
    os.utime(stale, (1, 1))  # an old tree of another image: not this run's
    (cache / "tuning" / "sglang").mkdir(parents=True)
    (cache / "tuning" / "sglang" / "t.json").write_text("{}", encoding="utf-8")
    recipe = {
        "runtime": "sglang",
        "model_revision": "abc",
        "defaults": {"port": 1, "gpu_memory_utilization": 0.3, "max_num_seqs": 16},
    }
    run = make_run(tmp_path, record(), FakeDocker(), recipe=recipe, sparkrun_cache=cache)
    run._launched_at = time.time()
    first = run.stamp()
    assert first["model_revision"] == "abc"
    assert (first["memory_fraction"], first["max_sequences"]) == ("0.3", "16")
    assert first["autotune_cache_key"] == "sglang/img-1/model-2"
    assert len(first["recipe_sha256"]) == 64
    assert first["tuning_hash"] != "absent"
    (leaf / "inductor" / "k2.bin").write_text("1", encoding="utf-8")  # a new kernel changes the listing hash
    assert run.stamp()["autotune_cache_hash"] != first["autotune_cache_hash"]
    cacheless = make_run(
        tmp_path,
        record(),
        FakeDocker(),
        recipe={**recipe, "runtime_cache": False},
        sparkrun_cache=cache / "none",
    )
    stamp = cacheless.stamp()
    assert (stamp["autotune_cache_key"], stamp["autotune_cache_hash"], stamp["tuning_hash"]) == (
        "off",
        "absent",
        "absent",
    )


def test_record_result_adds_the_runs_memory_and_what_it_measured(tmp_path: Path) -> None:
    rec = record(measured=None)
    run = make_run(tmp_path, rec, FakeDocker())
    run._gpu_peak = 20.0
    run._free_before, run._lowest_free = 90.0, 60.0
    container_run.record_result(rec, MODEL, run, extra={"kind": "warm", "complete": True})
    entry = container_run.read_record(Engine.SGLANG, rec.tag).results[-1]  # type: ignore[union-attr]
    assert entry["memory"]["fall_gib"] == 30.0
    assert (entry["gpu_peak_gib"], entry["kind"], entry["complete"], entry["run"]) == (
        20.0,
        "warm",
        True,
        "run0001",
    )
    assert MemoryNeed.recorded(container_run.read_record(Engine.SGLANG, rec.tag), MODEL) == 30.0  # type: ignore[arg-type]
    run._free_before, run._lowest_free = 90.0, 80.0  # a guard-killed run measured only part of its need
    container_run.record_result(rec, MODEL, run, extra={"kind": "candidate", "complete": False})
    assert MemoryNeed.recorded(container_run.read_record(Engine.SGLANG, rec.tag), MODEL) == 30.0  # type: ignore[arg-type]
    run._gpu_peak = 40.0  # larger, but incomplete: still not the need
    container_run.record_result(rec, MODEL, run, extra={"kind": "candidate", "complete": False})
    assert MemoryNeed.recorded(container_run.read_record(Engine.SGLANG, rec.tag), MODEL) == 30.0  # type: ignore[arg-type]


# --- bulk_server (design 3.8) ----------------------------------------------------------------------


class FakeRun:
    """A ContainerRun that starts, or is refused with ``fail``; the last one made is kept."""

    made: ClassVar[list[FakeRun]] = []
    fail: ClassVar[Exception | None] = None

    def __init__(
        self, record: VersionRecord, model: str, recipe: dict[str, object], *args: object, **kw: object
    ) -> None:
        self.model, self.recipe, self.kind = model, recipe, kw["kind"]
        self.stopped = False
        FakeRun.made.append(self)

    def __enter__(self) -> Self:
        if FakeRun.fail is not None:
            raise FakeRun.fail
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stopped = True

    def stamp(self) -> dict[str, str]:
        return {"engine": "sglang", "tag": "v0.5.23-cu130"}


class FakeManager:
    """A LocalServerManager that records how it was built, started and stopped."""

    made: ClassVar[list[FakeManager]] = []

    def __init__(self, spec: object, params: object, lease: object, **kw: object) -> None:
        self.spec, self.params, self.kw = spec, params, kw
        self.events: list[str] = []
        FakeManager.made.append(self)

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> bool:
        self.events.append("stop")
        return True

    def stamp(self, settings: SettingsAnswer) -> object:
        assert settings is self.kw["settings"]
        return type("Stamp", (), {"model_dump": lambda _s, mode: {"engine": "llama.cpp", "settings": {}}})()


def bulk_answer(engine: Engine, tmp_path: Path) -> tuple[SettingsAnswer, container_run.LlamaFallback]:
    choice = EngineChoice(
        engine=engine,
        rule=ChoiceRule.CHOSEN,
        version="v0.5.23-cu130",
        file=f"{MODEL}@{'b' * 40}",
    )
    answer = SettingsAnswer(
        model=ModelIdentity(repository="Qwen/Qwen3-30B-A3B-GGUF", revision="c" * 40),
        engine=engine,
        kind=WorkloadKind.K2,
        settings=Settings(n_ubatch=512),
        status=SettingsStatus.UNTUNED,
        source=SettingsSource.DEFAULT,
        choice=choice,
    )
    fallback = container_run.LlamaFallback(
        model_path=tmp_path / "q4.gguf",
        lease=object(),  # type: ignore[arg-type]
        http=object(),  # type: ignore[arg-type]
        pid_dir=tmp_path / "pid",
        port=18123,
        binary=tmp_path / "llama-server",
        guards=MemoryGuards(host_reserve_gib=10.0),
        params=container_run.derive_serving_params(DeviceType.CUDA, 122.0, role=ServingRole.BULK),
    )
    return answer, fallback


def test_bulk_server_serves_the_container_and_falls_back_to_llama_cpp_when_it_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = record()
    recipe = {"runtime": "sglang", "defaults": {"port": 18102}}
    write_record(rec.model_copy(update={"recipes": {MODEL: json.dumps(recipe)}}))
    FakeRun.made, FakeManager.made, FakeRun.fail = [], [], None
    monkeypatch.setattr(container_run, "ContainerRun", FakeRun)
    monkeypatch.setattr(container_run, "LocalServerManager", FakeManager)
    monkeypatch.setattr(container_run, "ensure_sparkrun", lambda _tag: tmp_path / "sparkrun")
    answer, fallback = bulk_answer(Engine.SGLANG, tmp_path)

    async def serve() -> container_run.BulkServer:
        async with container_run.bulk_server(answer, fallback=fallback) as served:
            return served

    started = asyncio.run(serve())  # the container starts: a project run on the recorded recipe's port
    (run,) = FakeRun.made
    assert (started.url, started.engine, started.fallback) == ("http://127.0.0.1:18102", Engine.SGLANG, None)
    assert (run.kind, run.model, run.recipe, run.stopped) == (RunKind.PROJECT, MODEL, recipe, True)
    assert started.stamp["tag"] == "v0.5.23-cu130"
    assert started.stamp["settings"]["engine_choice"]["rule"] == "chosen"
    assert FakeManager.made == []

    refusals = (
        ContainerStartError("its image is not on disk"),
        NotEnoughMemoryError("needs 105 GiB"),
        OSError("No space left on device"),  # preparing the start: still the fallback, not the job's end
    )
    for refusal in refusals:
        FakeRun.made, FakeManager.made, FakeRun.fail = [], [], refusal
        served = asyncio.run(serve())
        (manager,) = FakeManager.made
        assert manager.events == ["start", "stop"]  # started and stopped by the context
        assert (served.url, served.engine) == ("http://127.0.0.1:18123", Engine.LLAMA_CPP)
        assert served.fallback is not None
        assert str(refusal) in served.fallback
        assert served.stamp["fallback"] == served.fallback  # the result says it was not the engine chosen
        assert manager.kw["settings"].engine is Engine.LLAMA_CPP  # the answer's settings, as llama.cpp's
        assert manager.params.n_parallel == 16  # the bulk layout
        assert manager.spec.model_path == tmp_path / "q4.gguf"
    FakeRun.fail = None

    llama_answer, _ = bulk_answer(Engine.LLAMA_CPP, tmp_path)
    FakeRun.made, FakeManager.made = [], []
    plain = asyncio.run(_serve_answer(llama_answer, fallback))
    assert (plain.engine, plain.fallback, FakeRun.made) == (Engine.LLAMA_CPP, None, [])

    pytorch_answer, _ = bulk_answer(Engine.PYTORCH, tmp_path)
    with pytest.raises(ValueError, match="not a server"):
        asyncio.run(_serve_answer(pytorch_answer, fallback))
    named = answer.model_copy(update={"choice": None})  # a container engine named, not chosen: no version
    with pytest.raises(ValueError, match="named, not chosen"):
        asyncio.run(_serve_answer(named, fallback))


async def _serve_answer(
    answer: SettingsAnswer, fallback: container_run.LlamaFallback
) -> container_run.BulkServer:
    async with container_run.bulk_server(answer, fallback=fallback) as served:
        return served
