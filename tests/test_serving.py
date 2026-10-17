"""The model-serving pattern: wire protocol, server loop, server process, lazy registry and the warm host.

The protocol and server-loop tests moved here from service_b with the code (its unit test_protocol.py).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from localmodels.lease import GpuLease, GpuLeaseClient
from localmodels.serving import (
    ErrorMessage,
    InferMessage,
    LazyRegistry,
    LoadedMessage,
    LoadMessage,
    Message,
    ModelHost,
    ModelServerProcess,
    ProgressMessage,
    ProgressStage,
    ResultMessage,
    ServingCancelled,
    ServingError,
    ServingErrorCode,
    ShutdownMessage,
    decode,
    encode,
    serve,
)
from tests.echo_server import EchoServer

ROOT = Path(__file__).resolve().parents[1]
ECHO_ARGV = [sys.executable, "-m", "tests.echo_server"]
ECHO_ENV = {"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
ECHO = ("echo", "r1")

MESSAGES: list[Message] = [
    LoadMessage(model_dir="/m", revision="abc", options={"extra": "/e"}),
    LoadedMessage(seconds=1.5, torch_allocated_gib=2.0, torch_reserved_gib=3.0),
    InferMessage(request_id="r1", input_dir="/in", params={"sleep_s": 1}),
    ProgressMessage(request_id="r1", stage=ProgressStage.DECODE, new_tokens=4),
    ResultMessage(request_id="r1", raw_text="x", native_outputs={"a": 1}, timings_s={"t": 0.1}),
    ErrorMessage(request_id=None, code=ServingErrorCode.CUDA_OOM, message="oom"),
    ShutdownMessage(),
]


@pytest.mark.parametrize("msg", MESSAGES, ids=lambda m: m.op.value)
def test_round_trip_is_one_line(msg: Message) -> None:
    line = encode(msg)
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert decode(line) == msg


@pytest.mark.parametrize(
    "line",
    [
        b'{"op": "nope"}',
        b'{"op": "shutdown", "extra": 1}',
        b'{"op": "error", "code": "other", "message": "m"}',
        b'{"op": "error", "code": "timeout", "message": "m"}',
        b"not json",
    ],
    ids=["unknown-op", "extra-field", "unknown-code", "manager-only-code", "not-json"],
)
def test_decode_rejects(line: bytes) -> None:
    with pytest.raises(ValidationError):
        decode(line)


def test_an_error_carries_one_of_the_three_server_codes_by_its_wire_value() -> None:
    for code in (ServingErrorCode.CUDA_OOM, ServingErrorCode.BAD_INPUT, ServingErrorCode.INTERNAL):
        reply = decode(f'{{"op":"error","code":"{code.value}","message":"m"}}'.encode())
        assert isinstance(reply, ErrorMessage)
        assert reply.code is code
        assert f'"code":"{code.value}"'.encode() in encode(reply)


def _serve(*lines: bytes) -> list[Message]:
    out = io.BytesIO()
    serve(EchoServer(), io.BytesIO(b"".join(lines)), out)
    return [decode(line) for line in out.getvalue().splitlines()]


def test_the_server_loop_answers_each_request_until_shutdown() -> None:
    replies = _serve(
        encode(LoadMessage(model_dir="", revision="r")),
        encode(InferMessage(request_id="r1", input_dir="/in", params={"progress": 1})),
        encode(InferMessage(request_id="r2", input_dir="/", params={"fail": "oom"})),
        encode(ShutdownMessage()),
        encode(LoadMessage(model_dir="", revision="after-shutdown")),
    )
    assert [r.op.value for r in replies] == ["loaded", "progress", "result", "error"]
    assert isinstance(replies[2], ResultMessage)
    assert replies[2].raw_text == "/in"
    assert isinstance(replies[3], ErrorMessage)
    assert (replies[3].request_id, replies[3].code) == ("r2", ServingErrorCode.CUDA_OOM)


def test_server_errors_carry_no_exception_text() -> None:
    replies = _serve(
        encode(LoadMessage(model_dir="", revision="x", options={"fail": True})),
        encode(InferMessage(request_id="r1", input_dir="/", params={"fail": "leak"})),
        b'{"op": "infer"}\n',
    )
    assert [(r.code, r.message) for r in replies if isinstance(r, ErrorMessage)] == [
        (ServingErrorCode.INTERNAL, "RuntimeError"),
        (ServingErrorCode.INTERNAL, "ValueError"),
        (ServingErrorCode.BAD_INPUT, "malformed message"),
    ]


def _started() -> ModelServerProcess:
    server = ModelServerProcess(ECHO_ARGV, ECHO_ENV, ECHO)
    server.start()
    server.load(LoadMessage(model_dir="/m", revision="r1"), lambda: False)
    return server


def test_a_server_process_infers_reports_progress_survives_a_reported_error_and_stops() -> None:
    server = _started()
    try:
        assert server.serves(ECHO)
        assert not server.serves(("echo", "r2"))
        progress: list[ProgressMessage] = []
        request = InferMessage(request_id="j", input_dir="/in", params={"progress": 2})
        result = server.infer(request, lambda: False, progress.append)
        assert (result.raw_text, [p.new_tokens for p in progress]) == ("/in", [1, 2])
        with pytest.raises(ServingError) as oom:
            server.infer(
                InferMessage(request_id="j", input_dir="/", params={"fail": "oom"}), lambda: False, print
            )
        assert oom.value.code == ServingErrorCode.CUDA_OOM
        assert server.alive  # a reported failure keeps the model loaded
    finally:
        server.stop()
    assert not server.alive


@pytest.mark.parametrize(
    ("params", "timeout_s", "cancelled", "code"),
    [
        ({"sleep_s": 30}, 0.5, False, ServingErrorCode.TIMEOUT),
        ({"exit": True}, 30.0, False, ServingErrorCode.INTERNAL),
        ({"sleep_s": 30}, 30.0, True, None),
    ],
    ids=["stuck", "crashed", "cancelled"],
)
def test_a_stuck_crashed_or_cancelled_server_is_stopped(
    params: dict[str, object], timeout_s: float, cancelled: bool, code: ServingErrorCode | None
) -> None:
    server = _started()
    request = InferMessage(request_id="j", input_dir="/", params=params)
    with pytest.raises(ServingError if code else ServingCancelled) as raised:
        server.infer(request, lambda: cancelled, lambda _progress: None, timeout_s)
    if code is not None:
        assert isinstance(raised.value, ServingError)
        assert raised.value.code == code
    assert not server.alive


def test_the_registry_imports_on_first_use_keeps_one_instance_and_refuses_an_unknown_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "lazy_probe.py").write_text("class Probe:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry: LazyRegistry[str, object] = LazyRegistry({"probe": "lazy_probe:Probe"})
    assert "lazy_probe" not in sys.modules
    first = registry.get("probe")
    assert "lazy_probe" in sys.modules
    assert registry.get("probe") is first
    with pytest.raises(ServingError) as unknown:
        registry.get("other")
    assert unknown.value.code == ServingErrorCode.NOT_IMPLEMENTED
    monkeypatch.delitem(sys.modules, "lazy_probe")


def test_the_host_waits_its_turn_keeps_the_model_warm_unloads_when_idle_and_yields(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="service_b")
    now = [0.0]
    host = ModelHost(GpuLeaseClient(lease), idle_unload_s=300, poll_s=0.01, clock=lambda: now[0])
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        holder = tmp_path / "other.json"
        holder.write_text(json.dumps({"pid": other.pid, "project": "other"}))
        waits: list[list[str]] = []
        gave_up = host.acquire(
            job_id="j1",
            model_path="/m",
            should_stop=lambda: bool(waits),
            on_wait=lambda blocking: waits.append([r.project for r in blocking]),
        )
        assert (gave_up, waits, host.holding, lease.read_own()) == (False, [["other"]], False, None)

        holder.unlink()
        assert host.acquire(job_id="j1", model_path="/m", should_stop=lambda: False, on_wait=print)
        own = lease.read_own()
        assert own is not None
        assert (own.model_path, own.requested_at, host.holding) == ("/m", None, True)
        load = LoadMessage(model_dir="/m", revision="r1")
        warm = host.server(ECHO, ECHO_ARGV, ECHO_ENV, load, lambda: False)
        assert host.server(ECHO, ECHO_ARGV, ECHO_ENV, load, lambda: False) is warm
        host.finished()
        now[0] = 299.0
        host.unload_if_idle()
        assert warm.alive
        now[0] = 300.0
        host.unload_if_idle()
        assert (warm.alive, host.holding, lease.read_own()) == (False, False, None)

        assert host.acquire(job_id="j2", model_path="/m", should_stop=lambda: False, on_wait=print)
        (tmp_path / "waiter.json").write_text(
            json.dumps({"pid": other.pid, "project": "waiter", "requested_at": 1})
        )
        assert host.yield_if_requested() == "waiter"
        assert (host.holding, lease.read_own()) == (False, None)
    finally:
        other.kill()
        other.wait()
