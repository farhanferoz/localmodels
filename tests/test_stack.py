"""Stamps: the host fingerprint, the llama.cpp commit read from a library, and started servers' stamps."""

from __future__ import annotations

import struct
import subprocess
import sys
from typing import TYPE_CHECKING

import httpx

from localmodels.enums import ModelSizeClass, Precision, QuantLevel, WorkloadKind
from localmodels.launcher import LocalServerManager
from localmodels.lease import GpuLease, GpuLeaseClient
from localmodels.params import ServingParams
from localmodels.pinned import PinnedFile
from localmodels.process import ServerSpec
from localmodels.serving.host import ModelHost
from localmodels.serving.protocol import LoadMessage
from localmodels.settings import ModelIdentity, SettingsAnswer
from localmodels.stack import (
    UNKNOWN,
    Engine,
    LoadOptions,
    ServerStamp,
    Settings,
    SettingsSource,
    SettingsStatus,
    fingerprint,
    llama_cpp_build,
)
from localmodels.version import package_version
from tests.test_serving import ECHO, ECHO_ARGV, ECHO_ENV

if TYPE_CHECKING:
    from pathlib import Path

NVIDIA_VERSION = (
    "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.173.02  Release Build  (dvs-builder)\n"
    "GCC version:  gcc version 13.3.0\n"
)
DGX_RELEASE = (
    'DGX_NAME="DGX Spark"\nDGX_SWBUILD_VERSION="7.2.3"\n\n'
    'DGX_OTA_VERSION="7.5.0"\nDGX_OTA_DATE="Thu Aug 27"\n\nDGX_OTA_VERSION="7.6.0"\n'
)
AARCH64, X86_64 = 183, 62
ANSWER = SettingsAnswer(
    model=ModelIdentity(
        repository="google/medgemma-1.5-4b-it", files=(PinnedFile(path="m.gguf", sha256="cd" * 32),)
    ),
    kind=WorkloadKind.K5,
    settings=Settings(dtype=Precision.BFLOAT16),
    status=SettingsStatus.TUNED,
    source=SettingsSource.RECORD,
)
RELATIVE = {AARCH64: 1027, X86_64: 8}  # R_*_RELATIVE


def fake_llama_library(path: Path, commit: str, *, relocated: bool = True, machine: int = AARCH64) -> Path:
    """A minimal 64-bit ELF shared object holding ``char const * LLAMA_COMMIT = commit``, as llama.cpp has it.

    One loaded segment (address = file offset), a dynamic symbol table and, when ``relocated``, the relative
    relocation that fills the pointer at load time; otherwise the pointer is stored in the file.
    """
    strtab = b"\0LLAMA_COMMIT\0"
    text = commit.encode() + b"\0"
    ehdr, phdr, sym, rela = 64, 56, 24, 24
    strtab_at = ehdr + phdr
    text_at = strtab_at + len(strtab)
    slot_at = (text_at + len(text) + 7) // 8 * 8
    symtab_at = slot_at + 8
    rela_at = symtab_at + 2 * sym
    end = rela_at + (rela if relocated else 0)
    symbols = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0) + struct.pack("<IBBHQQ", 1, 0x11, 0, 1, slot_at, 8)
    relocation = struct.pack("<QQq", slot_at, RELATIVE[machine], text_at) if relocated else b""
    pointer = struct.pack("<Q", 0 if relocated else text_at)
    ident = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    header = struct.pack("<HHIQQQIHHHHHH", 3, machine, 1, 0, ehdr, end, 0, ehdr, phdr, 1, 64, 4, 0)
    segment = struct.pack("<IIQQQQQQ", 1, 4, 0, 0, 0, end, end, 0x1000)
    sections = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 11, 2, symtab_at, symtab_at, 2 * sym, 2, 1, 8, sym),  # .dynsym, its strings in section 2
        (0, 3, 2, strtab_at, strtab_at, len(strtab), 0, 0, 1, 0),  # .dynstr
        (0, 4, 2, rela_at, rela_at, rela if relocated else 0, 1, 0, 8, rela),  # .rela.dyn
    ]
    body = ident + header + segment + strtab + text + bytes(slot_at - text_at - len(text)) + pointer
    body += symbols + relocation
    path.write_bytes(body + b"".join(struct.pack("<IIQQQQIIQQ", *section) for section in sections))
    return path


def test_the_fingerprint_reads_the_host_files_and_says_unknown_for_a_missing_source(tmp_path: Path) -> None:
    driver, dgx = tmp_path / "version", tmp_path / "dgx-release"
    driver.write_text(NVIDIA_VERSION)
    dgx.write_text(DGX_RELEASE)
    found = fingerprint(nvidia_version_file=driver, dgx_release_file=dgx, cuda_version=lambda: "13.0")
    assert (found["driver"], found["cuda"], found["dgx_os"]) == ("580.173.02", "13.0", "7.6.0")
    assert found["localmodels"] == package_version()
    assert found["kernel"]

    gone = fingerprint(
        nvidia_version_file=tmp_path / "none",
        dgx_release_file=tmp_path / "none",
        cuda_version=lambda: UNKNOWN,
    )
    assert (gone["driver"], gone["cuda"], gone["dgx_os"]) == (UNKNOWN, UNKNOWN, UNKNOWN)


def test_the_build_commit_is_read_from_the_library_beside_the_binary(tmp_path: Path) -> None:
    library = fake_llama_library(tmp_path / "libllama-common.so", "ccc3646")
    assert llama_cpp_build(tmp_path / "llama-server") == "ccc3646"
    library.write_bytes(b"not an ELF file")
    assert llama_cpp_build(tmp_path / "llama-server") == UNKNOWN
    assert llama_cpp_build(tmp_path / "missing" / "llama-server") == UNKNOWN
    # with no library beside it, the binary itself is read: relocated or stored in the file, either machine
    (tmp_path / "bin").mkdir()
    for relocated, machine in [(True, AARCH64), (True, X86_64), (False, X86_64)]:
        own = fake_llama_library(
            tmp_path / "bin" / "llama-server", "abc1234", relocated=relocated, machine=machine
        )
        assert llama_cpp_build(own) == "abc1234"


def test_a_manager_stamps_what_it_started(tmp_path: Path) -> None:
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir()
    fake_llama_library(binary.parent / "libllama-common.so", "ccc3646")
    params = ServingParams(
        n_gpu_layers=99, n_ctx=8192, n_parallel=2, recommended_quant=QuantLevel.Q4_K_M,
        recommended_size_class=ModelSizeClass.LARGE, assumptions=[],
    )  # fmt: skip
    manager = LocalServerManager(
        ServerSpec(host="127.0.0.1", port=8091, model_path=tmp_path / "m.gguf"),
        params,
        GpuLease(tmp_path / "lease", project="localmodels-test"),
        http=httpx.AsyncClient(),
        pid_dir=tmp_path / "pids",
        binary=binary,
        extra_flags=("--no-warmup",),
    )
    stamp = manager.stamp(model_sha256="ab" * 32)
    assert (stamp.engine, stamp.build, stamp.model_path, stamp.model_sha256) == (
        Engine.LLAMA_CPP, "ccc3646", str(tmp_path / "m.gguf"), "ab" * 32,
    )  # fmt: skip
    assert stamp.layout is not None
    assert (stamp.layout.n_ctx, stamp.layout.n_parallel, stamp.layout.extra_flags) == (
        8192,
        2,
        ("--no-warmup",),
    )
    assert (stamp.load, stamp.host) == (None, fingerprint())
    assert manager.stamp().model_sha256 is None
    # no settings: the JSON it always had, which parses; with settings: a stamp that round-trips
    assert "settings" not in manager.stamp().model_dump_json()
    old = manager.stamp().model_dump(mode="json")
    assert "settings" not in old
    assert ServerStamp.model_validate(old).settings is None
    stamped = manager.stamp(settings=ANSWER)
    assert stamped.settings is not None
    assert (stamped.settings.status, stamped.settings.applied) == (
        SettingsStatus.TUNED,
        Settings(),
    )  # this manager has no answer, so it passed none of the settings
    assert ServerStamp.model_validate_json(stamped.model_dump_json()) == stamped
    # an in-process load's stamp: the identity's sole file is its hash unless given; its engine and precision
    assert ANSWER.server_stamp("/m", "2.10.0").model_sha256 == "cd" * 32
    assert ANSWER.server_stamp("/m", "2.10.0", model_sha256="ef" * 32).model_sha256 == "ef" * 32
    python = ANSWER.model_copy(update={"engine": Engine.LLAMA_CPP_PYTHON, "tune_detail": "job 7"})
    in_process = python.server_stamp("/m", "0.3.1", applied=Settings(dtype=Precision.FLOAT16))
    assert (in_process.engine, in_process.load) == (
        Engine.LLAMA_CPP_PYTHON,
        LoadOptions(dtype=Precision.FLOAT16),
    )
    assert in_process.settings is not None
    assert (in_process.settings.applied.dtype, in_process.settings.tune_detail) == (
        Precision.FLOAT16,
        "job 7",
    )


def test_a_host_stamps_the_model_it_loaded(tmp_path: Path) -> None:
    host = ModelHost(GpuLeaseClient(GpuLease(tmp_path, project="service_b")), idle_unload_s=300, poll_s=0.01)
    assert host.stamp() is None
    load = LoadMessage(model_dir="/m", revision="r1", dtype=Precision.FLOAT16, options={"attn": "sdpa"})
    try:
        host.server(ECHO, ECHO_ARGV, ECHO_ENV, load, lambda: False)
        stamp = host.stamp()
        stamped = host.stamp(settings=ANSWER)  # the answer says bfloat16; the load used float16
    finally:
        host.unload()
    assert stamp is not None
    assert (stamp.engine, stamp.build, stamp.model_path, stamp.layout) == (
        Engine.PYTORCH, package_version(), "/m", None,
    )  # fmt: skip
    assert stamp.load is not None
    assert (stamp.load.dtype, stamp.load.options) == (Precision.FLOAT16, {"attn": "sdpa"})
    assert stamp.settings is None
    assert stamped is not None
    assert stamped.settings is not None
    assert stamped.settings.status is SettingsStatus.TUNED
    assert stamped.settings.applied == Settings(dtype=Precision.FLOAT16, options={"attn": "sdpa"})
    assert host.stamp() is None


def test_importing_localmodels_does_not_import_the_stack_or_engines_modules() -> None:
    code = (
        "import sys, localmodels; "
        "lazy = ('localmodels.stack', 'localmodels.engines', 'localmodels.engine_bench', "
        "'localmodels.container_run', 'localmodels.engine_refresh'); "
        "sys.exit(any(m in sys.modules for m in lazy))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603 - fixed argv
