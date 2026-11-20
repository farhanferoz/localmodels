"""The model-serving pattern: wire protocol, server loop, server process, lazy registry and the warm host.

The protocol and server-loop tests moved here from service_b with the code (its unit test_protocol.py).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from localmodels.enums import DeviceTier, DeviceType, Precision
from localmodels.hardware import HardwareReport
from localmodels.lease import GpuLease, GpuLeaseClient
from localmodels.serving import (
    ErrorMessage,
    InferMessage,
    LazyRegistry,
    LoadedMessage,
    LoaderDeclaration,
    LoaderServer,
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
    TooManyCandidates,
    candidates,
    decode,
    encode,
    require_device,
    resolve_factory,
    serve,
)
from localmodels.stack import Settings
from tests.echo_server import EchoServer

ROOT = Path(__file__).resolve().parents[1]
ECHO_ARGV = [sys.executable, "-m", "tests.echo_server"]
ECHO_ENV = {"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
ECHO = ("echo", "r1")

MESSAGES: list[Message] = [
    LoadMessage(model_dir="/m", revision="abc", options={"extra": "/e"}),
    LoadedMessage(seconds=1.5, torch_allocated_gib=2.0, torch_reserved_gib=3.0, libraries={"torch": "2.9"}),
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


def test_the_server_loop_answers_each_request_until_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("localmodels.serving.server.LIBRARIES", ("pytest", "no-such-distribution"))
    replies = _serve(
        encode(LoadMessage(model_dir="", revision="r")),
        encode(InferMessage(request_id="r1", input_dir="/in", params={"progress": 1})),
        encode(InferMessage(request_id="r2", input_dir="/", params={"fail": "oom"})),
        encode(ShutdownMessage()),
        encode(LoadMessage(model_dir="", revision="after-shutdown")),
    )
    assert [r.op.value for r in replies] == ["loaded", "progress", "result", "error"]
    assert isinstance(replies[0], LoadedMessage)
    assert replies[0].libraries == {"pytest": pytest.__version__}  # installed ones only, read from metadata
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


def test_time_this_process_spent_frozen_does_not_count_against_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _started()
    jump = [0.0]
    real = time.monotonic
    monkeypatch.setattr("localmodels.serving.client.time.monotonic", lambda: real() + jump[0])

    def frozen_once() -> bool:  # the first poll comes back after a 100-s freeze (D10), as the clock sees it
        jump[0] = 100.0
        return False

    try:
        request = InferMessage(request_id="j", input_dir="/", params={"sleep_s": 0.5})
        server.infer(request, frozen_once, lambda _progress: None, 3.0)  # answers; does not time out
    finally:
        server.stop()


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
        assert host.loaded is warm
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


OLDER_SERVER = (  # answers a load without first saying which version it runs, as servers before 0.3.1 did
    "import sys; sys.stdin.readline(); "
    'print(\'{"op":"loaded","seconds":0,"torch_allocated_gib":0,"torch_reserved_gib":0}\', flush=True); '
    "sys.stdin.readline()"
)
OTHER_SERVER = 'import sys; print(\'{"op":"ready","version":"0.0.1"}\', flush=True); sys.stdin.readline()'


@pytest.mark.parametrize(
    ("script", "says"),
    [(OLDER_SERVER, "an older localmodels"), (OTHER_SERVER, "localmodels 0.0.1")],
    ids=["older", "other-version"],
)
def test_a_server_on_another_localmodels_version_is_refused_with_a_plain_message(
    script: str, says: str
) -> None:
    server = ModelServerProcess([sys.executable, "-c", script])
    server.start()
    with pytest.raises(ServingError) as refused:
        server.load(LoadMessage(model_dir="/m", revision="r"), lambda: False, timeout_s=10)
    assert refused.value.code == ServingErrorCode.NOT_IMPLEMENTED
    assert says in refused.value.message
    assert "install the same version" in refused.value.message
    assert not server.alive


def test_a_missing_server_program_is_a_serving_error_in_plain_words(tmp_path: Path) -> None:
    server = ModelServerProcess([str(tmp_path / "no-such-python")])
    with pytest.raises(ServingError) as refused:
        server.start()
    assert refused.value.code == ServingErrorCode.INTERNAL
    assert refused.value.message.startswith(f"The model server could not start: {tmp_path}/no-such-python: ")
    assert not server.alive


def test_unsupported_hardware_is_refused_in_plain_words_before_anything_starts() -> None:
    cpu = HardwareReport(
        platform="linux", arch="x86_64", device=DeviceType.CPU, device_name="CPU", total_memory_gb=8.0,
        tier=DeviceTier.SLOW_CPU,
    )  # fmt: skip
    assert require_device({DeviceType.CUDA, DeviceType.CPU}, cpu) is DeviceType.CPU
    with pytest.raises(ServingError) as refused:
        require_device({DeviceType.CUDA}, cpu)
    assert refused.value.code == ServingErrorCode.NOT_IMPLEMENTED
    assert refused.value.message == "This model needs a CUDA or ROCm GPU; this machine has only a CPU (CPU)."


def test_a_model_server_loads_neither_the_launcher_nor_its_http_client() -> None:
    probe = (
        "import sys, localmodels.serving as _; print({'httpx', 'localmodels.launcher'} & set(sys.modules))"
    )
    env = {**os.environ, **ECHO_ENV}
    out = subprocess.run(  # noqa: S603 - this interpreter and a fixed probe
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "set()"


def test_network_access_is_refused_and_counted_while_local_sockets_still_work() -> None:
    probe = (
        "import socket\n"
        "from localmodels.serving import block_network\n"
        "attempts = block_network()\n"
        "for call in (lambda: socket.socket().connect(('192.0.2.1', 80)), "
        "lambda: socket.getaddrinfo('example.com', 443)):\n"
        "    try:\n"
        "        call()\n"
        "    except OSError as exc:\n"
        "        assert 'blocked' in str(exc)\n"
        "a, b = socket.socketpair(socket.AF_UNIX)\n"
        "print(attempts)\n"
    )
    env = {**os.environ, **ECHO_ENV}
    out = subprocess.run(  # noqa: S603 - this interpreter and a fixed probe
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "['connect', 'lookup']"


# ---- the loader server and the candidates taken from a loader's declaration ----------------------

GIB = 1024**3
BF16, FP16 = Precision.BFLOAT16, Precision.FLOAT16
REFERENCE_OPTIONS = {"int8": False, "tokens": 1000}
DECLARATION = LoaderDeclaration(
    dtypes=(BF16, FP16),
    options={"int8": (False, True), "tokens": (1000, 2000, 4000)},
    reference=Settings(dtype=BF16, options=REFERENCE_OPTIONS),
)


class _FakeLoader:
    """A project's loader; the entry name picks what its scorer does."""

    def __init__(self, entry: str) -> None:
        self.entry = entry

    def declare(self) -> LoaderDeclaration:
        return DECLARATION

    def load(self, model_dir: Path, settings: Settings, device: str) -> Any:
        if self.entry == "missing":
            raise FileNotFoundError(2, "no such file", str(model_dir / "weights.bin"))
        assert (settings.dtype, device) == (BF16, "cuda")
        return {
            "short": lambda texts: [1.0] * (len(texts) - 1),
            "nan": lambda texts: [float("nan")] * len(texts),
        }.get(self.entry, lambda texts: [len(t) / 10 for t in texts])


@pytest.fixture
def fake_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A project module with a loader factory, a stub PyTorch and the texts of one request in ``tmp_path``."""
    project = types.ModuleType("fake_project_loaders")
    project.model = _FakeLoader  # type: ignore[attr-defined]
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: 2 * GIB,
        memory_reserved=lambda: 3 * GIB,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 5 * GIB,
    )
    monkeypatch.setitem(sys.modules, "fake_project_loaders", project)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr("localmodels.serving.server.block_network", list)  # it would patch this process
    rows = [{"id": "a", "text": "four"}, {"id": "b", "text": "twelve chars"}]
    (tmp_path / "texts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def _loader_replies(entry: str, folder: Path, *, dtype: Precision = BF16, **options: object) -> list[Message]:
    load = LoadMessage(model_dir=str(folder), revision="r", dtype=dtype, options=options)
    out = io.BytesIO()
    lines = [encode(load), encode(InferMessage(request_id="q", input_dir=str(folder)))]
    serve(LoaderServer("fake_project_loaders:model", entry), io.BytesIO(b"".join(lines)), out)
    return [decode(line) for line in out.getvalue().splitlines()]


def test_the_loader_server_scores_the_texts_by_id_and_reports_memory(fake_project: Path) -> None:
    loaded, result = _loader_replies("ok", fake_project, int8=False, tokens=2000)
    assert isinstance(loaded, LoadedMessage)
    assert (loaded.torch_allocated_gib, loaded.torch_reserved_gib) == (2.0, 3.0)
    assert isinstance(result, ResultMessage)
    assert result.native_outputs == {"scores": {"a": 0.4, "b": 1.2}}
    assert (result.request_id, result.peak_allocated_gib) == ("q", 5.0)


@pytest.mark.parametrize(
    ("dtype", "options", "named"),
    [
        (Precision.FLOAT32, {}, "float32"),
        (BF16, {"int8": "maybe"}, "'maybe'"),
        (BF16, {"int8": 1}, "1"),  # True is declared, 1 is not
        (BF16, {"cuda_graphs": True}, "'cuda_graphs'"),
    ],
    ids=["dtype", "value", "value-of-another-type", "option"],
)
def test_the_loader_server_refuses_what_the_loader_did_not_declare(
    fake_project: Path, dtype: Precision, options: dict[str, object], named: str
) -> None:
    error = _loader_replies("ok", fake_project, dtype=dtype, **options)[0]
    assert isinstance(error, ErrorMessage)
    assert error.code == ServingErrorCode.BAD_INPUT
    assert named in error.message
    assert "declared" in error.message


def test_a_weight_file_the_snapshot_lacks_is_named_by_its_path(fake_project: Path) -> None:
    error = _loader_replies("missing", fake_project)[0]
    assert isinstance(error, ErrorMessage)
    assert error.code == ServingErrorCode.BAD_INPUT
    assert (
        error.message
        == f"{fake_project / 'weights.bin'} is missing from the snapshot; download it before tuning"
    )


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("short", "the scorer returned 1 values for 2 texts"),
        ("nan", "the scorer returned a non-finite score for a"),
    ],
)
def test_a_scorer_that_returns_too_few_or_a_non_finite_score_is_an_internal_error(
    fake_project: Path, entry: str, message: str
) -> None:
    error = _loader_replies(entry, fake_project)[1]
    assert isinstance(error, ErrorMessage)
    assert (error.code, error.message) == (ServingErrorCode.INTERNAL, message)


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"dtypes": ()}, "at least one dtype"),
        (
            {"reference": Settings(dtype=Precision.FLOAT32, options={"int8": False, "tokens": 1000})},
            "float32",
        ),
        (
            {"reference": Settings(dtype=BF16, options={"int8": False})},
            "exactly one value per declared option",
        ),
        ({"reference": Settings(dtype=BF16, options={"int8": False, "tokens": 3})}, "3 of option 'tokens'"),
        ({"options": {"int8": (), "tokens": (1000,)}}, "Option 'int8' declares no values"),
        ({"reference": Settings(dtype=BF16, options=REFERENCE_OPTIONS, n_ubatch=512)}, "llama.cpp fields"),
    ],
    ids=[
        "no-dtype",
        "reference-dtype",
        "reference-lacks-option",
        "reference-value",
        "option-without-values",
        "llama-field",
    ],
)
def test_a_declaration_whose_reference_is_not_among_its_declared_values_is_refused(
    change: dict[str, Any], refusal: str
) -> None:
    with pytest.raises(ValidationError, match=refusal):
        LoaderDeclaration(**{**DECLARATION.model_dump(), **change})


def test_an_answer_with_missing_fields_is_completed_from_the_reference() -> None:
    rule_answer = Settings(dtype=BF16)  # what lookup answers from K6's rule: a dtype and no options
    assert DECLARATION.complete(rule_answer) == Settings(dtype=BF16, options=REFERENCE_OPTIONS)
    tuned = Settings(dtype=FP16, options={"tokens": 4000})
    assert DECLARATION.complete(tuned) == Settings(dtype=FP16, options={"int8": False, "tokens": 4000})
    assert DECLARATION.complete(Settings()) == DECLARATION.reference_settings()


def test_candidates_are_the_product_with_the_reference_first_then_single_changes_then_pairs() -> None:
    names = [c.name for c in candidates(DECLARATION, 12)]
    assert names == [
        "bfloat16,int8=false,tokens=1000",  # the reference
        "float16,int8=false,tokens=1000",  # one change each: dtype, then int8, then tokens in declared order
        "bfloat16,int8=true,tokens=1000",
        "bfloat16,int8=false,tokens=2000",
        "bfloat16,int8=false,tokens=4000",
        "float16,int8=true,tokens=1000",  # two changes
        "float16,int8=false,tokens=2000",
        "float16,int8=false,tokens=4000",
        "bfloat16,int8=true,tokens=2000",
        "bfloat16,int8=true,tokens=4000",
        "float16,int8=true,tokens=2000",  # three
        "float16,int8=true,tokens=4000",
    ]
    assert candidates(DECLARATION, 12)[2].options == {"int8": True, "tokens": 1000}
    with pytest.raises(TooManyCandidates, match=r"12 candidates.*cap of 11.*declare fewer values"):
        candidates(DECLARATION, 11)
    alone = LoaderDeclaration(dtypes=(BF16,), options={}, reference=Settings(dtype=BF16))
    assert [c.name for c in candidates(alone, 1)] == ["bfloat16"]


def test_a_loader_factory_is_found_by_module_and_attribute_or_refused_naming_the_spec(
    fake_project: Path,
) -> None:
    assert resolve_factory("fake_project_loaders:model") is _FakeLoader
    for spec in (
        "fake_project_loaders",
        "fake_project_loaders:nothing",
        "no_such_module_here:model",
        ":model",
    ):
        with pytest.raises(ValueError, match=f"Loader '{spec}'.*MODULE:ATTRIBUTE"):
            resolve_factory(spec)
