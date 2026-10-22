"""Unit tests for the local llama-server launcher and manager."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import stat
import sys
import time
from typing import TYPE_CHECKING

import httpx
import pytest

from localmodels.enums import ModelSizeClass, QuantLevel, WorkloadKind
from localmodels.launcher import (
    Contention,
    LeaseConflictError,
    LeaseWaitTimeoutError,
    LocalServerManager,
    MemoryGuards,
    PortHeldError,
    ReservedPortError,
    ServerStartError,
    StayUpConfigError,
    SupervisorExit,
    supervise,
)
from localmodels.lease import GpuLease, LeaseRecord, QuietRequest
from localmodels.params import ServingParams
from localmodels.pinned import PinnedFile
from localmodels.process import ApiKeyRequiredError, ServerSpec, _is_pid_alive, _process_args
from localmodels.settings import ModelIdentity, SettingsAnswer
from localmodels.stack import Engine, Settings, SettingsSource, SettingsStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

FAKE_SERVER_CODE = """#!__PYTHON__
import os
import sys
import signal
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

health_attempts = 0

with open(sys.argv[0] + ".cuda", "w") as env_out:
    env_out.write(repr(os.environ.get("CUDA_VISIBLE_DEVICES")))

model_path = ""
n_ctx = 8192
n_parallel = 1
port = 8091
override_model = None

host = "127.0.0.1"
for i, arg in enumerate(sys.argv):
    if arg == "-m" and i + 1 < len(sys.argv):
        model_path = sys.argv[i + 1]
    elif arg == "-c" and i + 1 < len(sys.argv):
        n_ctx = int(sys.argv[i + 1])
    elif arg == "-np" and i + 1 < len(sys.argv):
        n_parallel = int(sys.argv[i + 1])
    elif arg == "--port" and i + 1 < len(sys.argv):
        port = int(sys.argv[i + 1])
    elif arg == "--host" and i + 1 < len(sys.argv):
        host = sys.argv[i + 1]
    elif arg == "--override-model" and i + 1 < len(sys.argv):
        override_model = sys.argv[i + 1]

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global health_attempts
        if self.path == "/health":
            health_attempts += 1
            if health_attempts < __MIN_HEALTH_ATTEMPTS__:
                body = b"loading"
                self.send_response(503)
            else:
                body = b"OK"
                self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        elif self.path == "/props":
            reported_model = override_model if override_model else model_path
            per_slot_ctx = n_ctx // n_parallel
            payload = {
                "model_path": reported_model,
                "default_generation_settings": {"n_ctx": per_slot_ctx}
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, format, *args):
        pass

server = HTTPServer((host, port), Handler)
server.serve_forever()
"""


def _make_fake_server(tmp_path: Path, min_health_attempts: int = 3) -> Path:
    bin_path = tmp_path / "bin" / "llama-server"
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    code = FAKE_SERVER_CODE.replace("__PYTHON__", sys.executable).replace(
        "__MIN_HEALTH_ATTEMPTS__", str(min_health_attempts)
    )
    bin_path.write_text(code)
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def _manager(
    *,
    tmp_path: Path,
    http: httpx.AsyncClient,
    bin_path: Path,
    port: int,
    model_file: Path,
    params: ServingParams,
    reserved_ports: frozenset[int] = frozenset(),
    contention: Contention = Contention.REFUSE,
    stop_grace_s: float = 1.0,
    ready_timeout_s: float = 10.0,
    wait_poll_s: float = 0.05,
    wait_timeout_s: float | None = None,
    min_slot_tokens: int = 0,
    lease: GpuLease | None = None,
    host: str = "127.0.0.1",
    extra_env: dict[str, str] | None = None,
    guards: MemoryGuards | None = None,
    resident_allowance_gib: float | None = None,
    available_gib: Callable[[bool], float | None] | None = None,
    settings: SettingsAnswer | None = None,
) -> LocalServerManager:
    spec = ServerSpec(host=host, port=port, model_path=model_file)
    return LocalServerManager(
        spec,
        params,
        lease or GpuLease(tmp_path / "lease", project="localmodels-test"),
        reserved_ports=reserved_ports,
        contention=contention,
        ready_timeout_s=ready_timeout_s,
        http=http,
        pid_dir=tmp_path / "pids",
        binary=bin_path,
        stop_grace_s=stop_grace_s,
        wait_poll_s=wait_poll_s,
        wait_timeout_s=wait_timeout_s,
        min_slot_tokens=min_slot_tokens,
        extra_env=extra_env,
        guards=guards,
        resident_allowance_gib=resident_allowance_gib,
        settings=settings,
        **({} if available_gib is None else {"available_gib": available_gib}),
    )


def _answer(**chosen: object) -> SettingsAnswer:
    """A tuned settings answer holding ``chosen`` (``Settings`` fields), for a manager's ``settings=``."""
    files = (PinnedFile(path="m.gguf", sha256="cd" * 32),)
    return SettingsAnswer(
        model=ModelIdentity(repository="org/model", files=files),
        engine=Engine.LLAMA_CPP,
        kind=WorkloadKind.K5,
        settings=Settings(**chosen),  # type: ignore[arg-type]
        status=SettingsStatus.TUNED,
        source=SettingsSource.RECORD,
    )


def _gpu_params(**overrides: object) -> ServingParams:
    base = {
        "n_gpu_layers": 99,
        "n_ctx": 8192,
        "n_parallel": 1,
        "recommended_quant": QuantLevel.Q4_K_M,
        "recommended_size_class": ModelSizeClass.LARGE,
        "assumptions": [],
    }
    base.update(overrides)
    return ServingParams(**base)


def _cpu_params(**overrides: object) -> ServingParams:
    base = {
        "n_gpu_layers": 0,
        "recommended_quant": None,
        "recommended_size_class": ModelSizeClass.SMALL,
    }
    base.update(overrides)
    return _gpu_params(**base)


async def _child_cuda_env(tmp_path: Path, extra_env: dict[str, str] | None) -> str:
    """Start the fake server and return the CUDA_VISIBLE_DEVICES repr it saw in its environment."""
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8198,
            model_file=model_file,
            params=_cpu_params(),
            extra_env=extra_env,
        )
        await manager.start()
        try:
            return (bin_path.parent / "llama-server.cuda").read_text()
        finally:
            await manager.stop()


async def test_extra_env_overrides_the_child_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert await _child_cuda_env(tmp_path, {"CUDA_VISIBLE_DEVICES": ""}) == "''"


async def test_no_extra_env_child_inherits_the_parent_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert await _child_cuda_env(tmp_path, None) == "'0'"


async def test_start_and_stop_fake_server(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease = GpuLease(tmp_path / "lease", project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8199,
            model_file=model_file,
            params=_gpu_params(),
            lease=lease,
        )
        assert (await manager.status()).alive is False

        handle = await manager.start()
        assert handle.pid > 0
        assert (tmp_path / "pids" / "8199.pid").exists()
        assert lease.lease_path().exists()
        status = await manager.status()
        assert (status.port, status.pid, status.alive, status.healthy) == (8199, handle.pid, True, True)

        stopped = await manager.stop()
        assert stopped is True
        assert not (tmp_path / "pids" / "8199.pid").exists()
        assert not lease.lease_path().exists()
        assert (await manager.status()).alive is False


async def test_reserved_port_refuses_start_and_stop(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8080,
            model_file=model_file,
            params=_gpu_params(),
            reserved_ports=frozenset({8080}),
        )
        with pytest.raises(ReservedPortError, match="reserved"):
            await manager.start()
        with pytest.raises(ReservedPortError, match="reserved"):
            await manager.stop()


async def test_contention_refuse_raises_when_another_lease_is_live(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease_dir = tmp_path / "lease"

    sub = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(10)")
    try:
        lease_dir.mkdir(parents=True, exist_ok=True)
        (lease_dir / "other.json").write_text(json.dumps({"pid": sub.pid, "project": "other"}))

        lease = GpuLease(lease_dir, project="localmodels-test")
        async with httpx.AsyncClient() as http:
            manager = _manager(
                tmp_path=tmp_path,
                http=http,
                bin_path=bin_path,
                port=8198,
                model_file=model_file,
                params=_gpu_params(),
                contention=Contention.REFUSE,
                lease=lease,
            )
            with pytest.raises(LeaseConflictError, match="currently held by: other"):
                await manager.start()
    finally:
        sub.terminate()
        await sub.wait()


async def test_contention_wait_blocks_then_proceeds_once_the_other_lease_dies(tmp_path: Path) -> None:
    """Contention.WAIT does not raise; it waits for the other live lease to disappear."""
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease_dir = tmp_path / "lease"

    sub = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(0.3)")
    lease_dir.mkdir(parents=True, exist_ok=True)
    (lease_dir / "other.json").write_text(json.dumps({"pid": sub.pid, "project": "other"}))

    lease = GpuLease(lease_dir, project="localmodels-test")
    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8197,
            model_file=model_file,
            params=_gpu_params(),
            contention=Contention.WAIT,
            wait_poll_s=0.05,
            lease=lease,
        )
        start = time.time()
        handle = await manager.start()
        elapsed = time.time() - start
        assert handle.pid > 0
        assert elapsed >= 0.2  # actually waited for the sleeping process to die
        await manager.stop()
    await sub.wait()


async def test_contention_wait_times_out(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease_dir = tmp_path / "lease"

    sub = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(10)")
    try:
        lease_dir.mkdir(parents=True, exist_ok=True)
        (lease_dir / "other.json").write_text(json.dumps({"pid": sub.pid, "project": "other"}))

        lease = GpuLease(lease_dir, project="localmodels-test")
        async with httpx.AsyncClient() as http:
            manager = _manager(
                tmp_path=tmp_path,
                http=http,
                bin_path=bin_path,
                port=8196,
                model_file=model_file,
                params=_gpu_params(),
                contention=Contention.WAIT,
                wait_poll_s=0.05,
                wait_timeout_s=0.2,
                lease=lease,
            )
            with pytest.raises(LeaseWaitTimeoutError, match="timed out waiting"):
                await manager.start()
            assert not lease.lease_path().exists()  # the timed-out request is withdrawn
    finally:
        sub.terminate()
        await sub.wait()


async def test_port_already_held_by_external_socket_refuses_start(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 8195))
    sock.listen(1)

    try:
        async with httpx.AsyncClient() as http:
            manager = _manager(
                tmp_path=tmp_path,
                http=http,
                bin_path=bin_path,
                port=8195,
                model_file=model_file,
                params=_cpu_params(),
            )
            with pytest.raises(PortHeldError, match="held by a process this manager did not start"):
                await manager.start()
    finally:
        sock.close()


async def test_props_context_slot_mismatch_stops_server_and_refuses(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8194,
            model_file=model_file,
            params=_cpu_params(n_ctx=4096),
            min_slot_tokens=8192,
        )
        with pytest.raises(ServerStartError, match="is smaller than required min_slot_tokens"):
            await manager.start()
        assert not (tmp_path / "pids" / "8194.pid").exists()


async def test_unified_pool_needs_n_ctx_for_every_slot(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()

    async with httpx.AsyncClient() as http:

        def manager(n_ctx: int, *, kv_unified: bool = True) -> LocalServerManager:
            return _manager(
                tmp_path=tmp_path,
                http=http,
                bin_path=bin_path,
                port=8195,
                model_file=model_file,
                params=_gpu_params(n_ctx=n_ctx, n_parallel=4, kv_unified=kv_unified),
                min_slot_tokens=8192,
            )

        with pytest.raises(ServerStartError, match=r"n_ctx 16384.*4 slots.*min_slot_tokens 8192"):
            await manager(16384).start()
        assert not (tmp_path / "pids" / "8195.pid").exists()
        manager(32768)._check_slot_guarantee()
        manager(16384, kv_unified=False)._check_slot_guarantee()
        assert manager(32768).stamp().layout.kv_unified  # type: ignore[union-attr]


async def test_status_probes_the_configured_host(tmp_path: Path) -> None:
    """Deployed, a server may listen on a non-loopback host: status must probe spec.host, never
    a hardcoded 127.0.0.1, or it reports a healthy server as unhealthy."""
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    spec = ServerSpec(host="127.0.0.2", port=8192, model_path=model_file)

    async with httpx.AsyncClient() as http:
        manager = LocalServerManager(
            spec,
            _cpu_params(),
            GpuLease(tmp_path / "lease", project="localmodels-test"),
            http=http,
            pid_dir=tmp_path / "pids",
            binary=bin_path,
            ready_timeout_s=10.0,
        )
        await manager.start()
        try:
            status = await manager.status()
            assert status.healthy is True
        finally:
            await manager.stop()


async def test_supervise_server_killed_exits_and_cleans_lease(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease = GpuLease(tmp_path / "lease", project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8191,
            model_file=model_file,
            params=_gpu_params(),
            lease=lease,
        )
        handle = await manager.start()
        assert lease.lease_path().exists()

        stop_event = asyncio.Event()

        async def _kill_soon() -> None:
            await asyncio.sleep(0.1)
            try:
                os.killpg(os.getpgid(handle.pid), signal.SIGKILL)
            except OSError:
                os.kill(handle.pid, signal.SIGKILL)

        kill_task = asyncio.create_task(_kill_soon())
        exit_code = await supervise(manager, handle, poll_s=0.05, stop=stop_event)
        await kill_task

        assert exit_code == SupervisorExit.SERVER_EXITED
        assert not lease.lease_path().exists()


async def test_supervise_yields_when_another_lease_requests(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease_dir = tmp_path / "lease"
    lease = GpuLease(lease_dir, project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8190,
            model_file=model_file,
            params=_gpu_params(),
            lease=lease,
        )
        handle = await manager.start()
        assert lease.lease_path().exists()

        stop_event = asyncio.Event()
        sup_task = asyncio.create_task(supervise(manager, handle, poll_s=0.05, stop=stop_event))

        await asyncio.sleep(0.1)
        (lease_dir / "service_c.json").write_text(
            json.dumps({"pid": os.getpid(), "project": "service_c", "requested_at": int(time.time())})
        )

        result = await sup_task
        assert result == SupervisorExit.YIELDED
        assert not lease.lease_path().exists()
        assert not (tmp_path / "pids" / "8190.pid").exists()


async def test_adopted_gpu_server_writes_a_lease(tmp_path: Path) -> None:
    """A manager that adopts a running GPU server owns the GPU claim, so it writes the lease."""
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease = GpuLease(tmp_path / "lease", project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8189,
            model_file=model_file,
            params=_gpu_params(),
            lease=lease,
        )
        handle = await manager.start()
        lease.remove()  # the previous owner of this server is gone, the server is not

        adopted = await manager.start()
        assert adopted.pid == handle.pid
        assert lease.lease_path().exists()
        await manager.stop()


async def test_a_running_server_with_other_flags_is_restarted_for_a_settings_answer(tmp_path: Path) -> None:
    """Adopting by model name alone would run `-ub` 512 under a stamp that says 2048."""
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    (tmp_path / "linked").symlink_to(tmp_path)
    bin_path = _make_fake_server(tmp_path, 1)
    async with httpx.AsyncClient() as http:

        def manager(ubatch: int | None, model: Path = model_file) -> LocalServerManager:
            return _manager(
                tmp_path=tmp_path, http=http, bin_path=bin_path, port=8184, model_file=model,
                params=_gpu_params(), settings=None if ubatch is None else _answer(n_ubatch=ubatch),
            )  # fmt: skip

        old = await manager(512).start()
        assert (await manager(512).start()).pid == old.pid  # the same flags: adopted
        assert (await manager(512, tmp_path / "linked" / "test.gguf").start()).pid == old.pid  # the same file
        asked = manager(2048)
        # another project's lease on the port: its server is not ours to stop
        theirs = GpuLease(tmp_path / "lease", project="other-project")
        theirs.write(LeaseRecord(project="other-project", pid=os.getpid(), port=8184))
        with pytest.raises(LeaseConflictError, match="other-project"):
            await asked.start()
        assert _is_pid_alive(old.pid)
        theirs.remove()
        fresh = await asked.start()
        assert fresh.pid != old.pid
        assert not _is_pid_alive(old.pid)
        assert _process_args(fresh.pid)[-2:] == ["-ub", "2048"]
        assert (await manager(None).start()).pid == fresh.pid  # no answer: adopted as it always was
        await asked.stop()


async def test_adopted_cpu_server_writes_no_lease(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease = GpuLease(tmp_path / "lease", project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8188,
            model_file=model_file,
            params=_cpu_params(),
            lease=lease,
        )
        await manager.start()
        adopted = await manager.start()
        assert adopted.pid > 0
        assert not lease.lease_path().exists()
        await manager.stop()


async def test_start_refused_for_a_missing_api_key_leaves_no_lease(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease = GpuLease(tmp_path / "lease", project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8187,
            model_file=model_file,
            params=_gpu_params(),
            lease=lease,
            host="0.0.0.0",  # noqa: S104
        )
        with pytest.raises(ApiKeyRequiredError):
            await manager.start()
    assert not lease.lease_path().exists()


async def test_start_that_cannot_spawn_leaves_no_lease(tmp_path: Path) -> None:
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease = GpuLease(tmp_path / "lease", project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=tmp_path / "no-such-llama-server",
            port=8186,
            model_file=model_file,
            params=_gpu_params(),
            lease=lease,
        )
        with pytest.raises(OSError):  # noqa: PT011 -- FileNotFoundError from the spawn
            await manager.start()
    assert not lease.lease_path().exists()


async def test_supervise_does_not_yield_a_cpu_server_to_a_gpu_request(tmp_path: Path) -> None:
    """A server with no GPU layers holds nothing the other project needs, so it keeps serving."""
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease_dir = tmp_path / "lease"
    lease = GpuLease(lease_dir, project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8185,
            model_file=model_file,
            params=_cpu_params(),
            lease=lease,
        )
        handle = await manager.start()

        stop_event = asyncio.Event()
        sup_task = asyncio.create_task(supervise(manager, handle, poll_s=0.05, stop=stop_event))
        await asyncio.sleep(0.1)
        lease_dir.mkdir(parents=True, exist_ok=True)  # a CPU server wrote no lease, so no directory
        (lease_dir / "service_c.json").write_text(
            json.dumps({"pid": os.getpid(), "project": "service_c", "requested_at": int(time.time())})
        )
        await asyncio.sleep(0.4)
        assert not sup_task.done()

        stop_event.set()
        assert await sup_task == SupervisorExit.STOPPED


_STAY_UP_GUARDS = MemoryGuards(kill_below_gib=10, server_oom_score_adj=500, watch_poll_s=0.05)


def _resident(lease_dir: Path, project: str) -> dict[str, object]:
    (path,) = (lease_dir / "resident").glob(f"{project}-*.json")
    return json.loads(path.read_text())


async def test_stay_up_is_refused_without_its_guards(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        common = {"tmp_path": tmp_path, "http": http, "bin_path": tmp_path / "bin" / "x", "port": 8189}
        common |= {
            "model_file": tmp_path / "m.gguf",
            "params": _gpu_params(),
            "contention": Contention.STAY_UP,
        }
        with pytest.raises(StayUpConfigError, match=r"kill_below_gib, .*server_oom_score_adj"):
            _manager(**common, resident_allowance_gib=20)
        with pytest.raises(StayUpConfigError, match="resident_allowance_gib"):
            _manager(**common, guards=_STAY_UP_GUARDS)


async def test_stay_up_ignores_ordinary_requests_and_yields_to_a_quiet_card(tmp_path: Path) -> None:
    bin_path = _make_fake_server(tmp_path, min_health_attempts=1)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease_dir = tmp_path / "lease"
    lease = GpuLease(lease_dir, project="localmodels-test")

    async with httpx.AsyncClient() as http:
        manager = _manager(
            tmp_path=tmp_path,
            http=http,
            bin_path=bin_path,
            port=8189,
            model_file=model_file,
            params=_gpu_params(),
            lease=lease,
            contention=Contention.STAY_UP,
            guards=_STAY_UP_GUARDS,
            resident_allowance_gib=20,
            available_gib=lambda _on_gpu: 100.0,
        )
        (lease_dir / "service_c.json").parent.mkdir(parents=True, exist_ok=True)
        (lease_dir / "service_c.json").write_text(json.dumps({"pid": os.getpid(), "project": "service_c"}))
        handle = await manager.start()  # another project holds the lease and it still starts
        assert not lease.lease_path().exists()
        assert _resident(lease_dir, "localmodels-test")["loaded"] is True

        sup_task = asyncio.create_task(supervise(manager, handle, poll_s=0.05, stop=asyncio.Event()))
        (lease_dir / "service_c.json").write_text(
            json.dumps({"pid": os.getpid(), "project": "service_c", "requested_at": int(time.time())})
        )
        await asyncio.sleep(0.3)
        assert not sup_task.done()  # an ordinary request is not a reason to unload

        GpuLease(lease_dir, project="bench").write_quiet(
            QuietRequest(pid=os.getpid(), project="bench", requested_at=int(time.time()), reason="benchmark")
        )
        assert await sup_task == SupervisorExit.YIELDED
        assert _resident(lease_dir, "localmodels-test")["loaded"] is False  # still registered, not loaded


async def test_stay_up_floor_kill_marks_the_resident_unloaded(tmp_path: Path) -> None:
    """Also for a server adopted from a serve process that is gone: it is watched as if started here."""
    bin_path = _make_fake_server(tmp_path, min_health_attempts=1)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    level = [100.0]

    async with httpx.AsyncClient() as http:
        common = {
            "tmp_path": tmp_path, "http": http, "bin_path": bin_path, "port": 8188, "model_file": model_file,
            "params": _gpu_params(), "contention": Contention.STAY_UP, "guards": _STAY_UP_GUARDS,
            "resident_allowance_gib": 20, "available_gib": lambda _on_gpu: level[0],
        }  # fmt: skip
        first = _manager(**common)
        handle = await first.start()
        first._cancel_watchdog()  # the serve process that started it is gone; its server is not
        manager = _manager(**common)
        assert (await manager.start()).pid == handle.pid  # adopted
        assert _resident(tmp_path / "lease", "localmodels-test")["loaded"] is True
        level[0] = 3.0
        for _ in range(100):
            await asyncio.sleep(0.05)
            if not _resident(tmp_path / "lease", "localmodels-test")["loaded"]:
                break
        assert _resident(tmp_path / "lease", "localmodels-test")["loaded"] is False
        assert manager.killed_reason is not None
