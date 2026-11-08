"""What a result was made with: the host's software stack and the stamp of a started model server. Torch-free.

Also the settings types (``Settings`` and the statuses of an answer) that ``settings.py`` and a stamp share,
so that ``stack`` never imports ``settings``.

``fingerprint`` reads the host without starting CUDA (driver, CUDA, kernel, DGX OS, this package).
``llama_cpp_build`` reads the commit llama.cpp embeds in its own library without running it. ``ServerStamp``
joins them with what a server was started with, so a result can say how it was made; a tune record keeps the
fingerprint, and ``stale_components`` names what changed since. Imported on demand: ``import localmodels``
does not load this module.
"""

from __future__ import annotations

import platform
import re
import shutil
import struct
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field

from localmodels.enums import Precision  # noqa: TC001 - pydantic needs it at runtime
from localmodels.hardware import _nvml_read
from localmodels.pinned import PinnedFile  # noqa: TC001 - pydantic needs it at runtime
from localmodels.version import package_version

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

UNKNOWN: Final = "unknown"
NVIDIA_VERSION_FILE: Final = Path("/proc/driver/nvidia/version")
DGX_RELEASE_FILE: Final = Path("/etc/dgx-release")
LLAMA_COMMIT_SYMBOL: Final = "LLAMA_COMMIT"
LLAMA_COMMON_LIBRARY: Final = "libllama-common.so"


class Component(StrEnum):
    """The parts of the host stack the fingerprint names (its dictionary keys)."""

    DRIVER = "driver"
    CUDA = "cuda"
    KERNEL = "kernel"
    DGX_OS = "dgx_os"
    LOCALMODELS = "localmodels"


#: A change in any of these can change what a model computes or how fast, so a record made before it is stale.
STALE_COMPONENTS: Final = (Component.DRIVER, Component.CUDA, Component.KERNEL, Component.DGX_OS)


class Engine(StrEnum):
    """What serves the model."""

    LLAMA_CPP = "llama.cpp"
    LLAMA_CPP_PYTHON = (
        "llama-cpp-python"  # carries its own llama.cpp: a llama-server record never applies to it
    )
    PYTORCH = "pytorch"
    #: The container engines start through sparkrun inside localmodels' own wrapper (``engines.py``).
    VLLM = "vllm"
    SGLANG = "sglang"
    TRTLLM = "trtllm"


class SpecType(StrEnum):
    """llama.cpp speculative decoding (``--spec-type``)."""

    NONE = "none"
    DRAFT_MTP = "draft-mtp"
    DRAFT_SIMPLE = "draft-simple"
    NGRAM_MOD = "ngram-mod"


class SettingsStatus(StrEnum):
    """How far an answer can be trusted."""

    TUNED = "tuned"  # a record made on this machine and the current stack
    STALE = "stale"  # a record exists but a stamped part of the stack changed since
    RULE = "rule"  # no record; the rule was measured on this model, but not on this file and machine
    RULE_OTHER_MODEL = "rule-other-model"  # no record; the rule was measured on other models only
    UNTUNED = "untuned"  # nothing measured applies
    PINNED = "pinned"  # a study's own copy of an earlier answer, kept whatever the store says now


class SettingsSource(StrEnum):
    """Where the settings of an answer came from."""

    RECORD = "record"
    RULE = "rule"
    DEFAULT = "default"
    PIN = "pin"


class TuneState(StrEnum):
    """What the lookup did about a tune."""

    NOT_NEEDED = "not-needed"
    NOT_AVAILABLE = "not-available"  # a tune is wanted, but this engine's automatic re-tune is not enabled
    NO_RECIPE = "no-recipe"  # a tune is wanted and the caller gave no recipe
    QUEUED = "queued"  # this lookup submitted it; ``tune_detail`` is the slurm job id
    ALREADY_QUEUED = "already-queued"  # another lookup did and it has not finished (the job id)
    DISABLED = "disabled"  # the caller or the environment turned the automatic re-tune off
    NO_SCHEDULER = "no-scheduler"  # no ``submit.py`` here, or slurm could not be asked about a queued job
    NOT_APPLICABLE = "not-applicable"  # the lookup's hardware is one a tune job cannot produce
    NOTHING_TO_TUNE = "nothing-to-tune"  # the caller's explicit settings cover every field a tune sets
    FAILED_BEFORE = "failed-before"  # a tune was tried and did not finish (the log, or why it was not queued)
    MEASURED_NOT_ADOPTED = "measured-not-adopted"  # a tune measured but its result was refused (the reason)
    WAITING = "waiting"  # the cap on simultaneous tune jobs is reached; a later lookup submits


class RetunePolicy(StrEnum):
    """Whether a lookup may queue a tune by itself (D2); ``LOCALMODELS_RETUNE=off`` has the same effect."""

    AUTO = "auto"
    OFF = "off"


class TunePlan(StrEnum):
    """The tune an answer calls for: every candidate (first use) or the reference and the current choice."""

    FULL = "full"
    CONFIRM = "confirm"


class StoreState(StrEnum):
    """The state of the files the answer was read from."""

    OK = "ok"
    UNREADABLE = "unreadable"
    NEWER_SCHEMA = "newer-schema"


class Settings(BaseModel):
    """What to load a model with. A missing field is "no opinion": the caller keeps its own."""

    model_config = ConfigDict(frozen=True)

    dtype: Precision | None = None
    options: dict[str, Any] = Field(default_factory=dict)  # token_budget, matmul_precision, batch_size
    n_ubatch: int | None = None
    spec: SpecType | None = None
    spec_draft_file: str | None = None
    # the quantised file set to load, when the tune chose another than the reference
    model_file: tuple[PinnedFile, ...] | None = None


class LlamaLayout(BaseModel):
    """How a llama-server was laid out: the settings that decide its slots and so, even greedy, its output."""

    model_config = ConfigDict(frozen=True)

    n_ctx: int
    n_parallel: int
    kv_unified: bool = False
    extra_flags: tuple[str, ...] = ()


class LoadOptions(BaseModel):
    """What a PyTorch model server was asked to load with."""

    model_config = ConfigDict(frozen=True)

    dtype: Precision
    options: dict[str, Any] = Field(default_factory=dict)


class SettingsStamp(BaseModel):
    """How a result's settings were decided: what ran, where it came from and how far it could be trusted.

    ``record_id`` is the store file and the time the record was measured (``<path>@<ISO time>``);
    ``rule`` is the id of the rule that answered; ``stale`` is one line per changed part of the stack;
    ``overridden`` names the fields the caller's own settings replaced; ``pinned`` is the pin file the
    answer came from or was written to; ``changed_from`` is the previous choice's name when a re-tune
    changed it; ``recipe_project`` is the project whose inputs made the record.

    For a pinned answer ``status`` is PINNED and ``pinned_status`` is the status the pinned settings had when
    they were pinned, with ``record_id``, ``rule``, ``stale`` and ``recipe_project`` as they were then: a
    result pinned from an untuned answer still reads untuned. ``pin_problem`` says why a pin that was asked
    for could not be used (the answer is then not pinned). ``tune_detail`` is the job id or log path of a
    queued or failed tune.
    """

    model_config = ConfigDict(frozen=True)

    status: SettingsStatus
    source: SettingsSource
    applied: Settings
    record_id: str | None = None
    rule: str | None = None
    stale: list[str] = Field(default_factory=list)
    tune: TuneState
    store: StoreState
    overridden: list[str] = Field(default_factory=list)
    pinned: str | None = None
    pinned_status: SettingsStatus | None = None
    pin_problem: str | None = None
    tune_detail: str | None = None
    changed_from: str | None = None
    recipe_project: str | None = None
    #: Which engine the chooser picked and why (engine, rule, version, image, file); None for a caller that
    #: named its engine, and left out of a dump then so that caller's stamp is the JSON it always was.
    engine_choice: dict[str, str] | None = Field(default=None, exclude_if=lambda value: value is None)


class ServerStamp(BaseModel):
    """How a result's server was made: engine, build, model file, layout or load options, and the host.

    ``layout`` is set for llama.cpp and ``load`` for PyTorch. ``model_sha256`` is known only when the caller
    has it (from a ``PinnedFile``); it is never computed at start.
    """

    model_config = ConfigDict(frozen=True)

    engine: Engine
    build: str
    model_path: str
    model_sha256: str | None = None
    layout: LlamaLayout | None = None
    load: LoadOptions | None = None
    host: dict[str, str]
    # set by a caller that took its settings from a lookup; left out of a dump when None, so a caller with
    # no settings keeps the JSON it always had
    settings: SettingsStamp | None = Field(default=None, exclude_if=lambda value: value is None)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _driver_version(version_file: Path) -> str:
    """The driver version in /proc/driver/nvidia/version ("NVRM version: ... aarch64  580.173.02  ...")."""
    text = _read_text(version_file)
    match = re.search(r"NVRM version:.*?\s(\d+(?:\.\d+)+)\s", text or "")
    return match.group(1) if match else UNKNOWN


def _dgx_os_version(release_file: Path) -> str:
    """The newest over-the-air update version in /etc/dgx-release, else the version it was installed at."""
    text = _read_text(release_file) or ""
    ota = re.findall(r'^DGX_OTA_VERSION="([^"]+)"', text, re.MULTILINE)
    base = re.search(r'^DGX_SWBUILD_VERSION="([^"]+)"', text, re.MULTILINE)
    return ota[-1] if ota else base.group(1) if base else UNKNOWN


def _cuda_driver_version() -> str:
    """The CUDA version the driver supports, by NVML and then nvidia-smi; neither starts a CUDA context."""
    try:
        raw = _nvml_read("0", lambda nv, _handle: nv.nvmlSystemGetCudaDriverVersion())
        return f"{raw // 1000}.{raw % 1000 // 10}"
    except Exception:  # noqa: BLE001,S110 - a failed probe means "try nvidia-smi"
        pass
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return UNKNOWN
    try:
        out = subprocess.run(  # noqa: S603 - smi is a shutil.which() result
            [smi], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return UNKNOWN
    match = re.search(r"CUDA Version:\s*(\d+(?:\.\d+)*)", out)
    return match.group(1) if match else UNKNOWN


def fingerprint(
    *,
    nvidia_version_file: Path = NVIDIA_VERSION_FILE,
    dgx_release_file: Path = DGX_RELEASE_FILE,
    cuda_version: Callable[[], str] = _cuda_driver_version,
) -> dict[str, str]:
    """The host's stack by component name (``Component``); "unknown" for a source that is missing."""
    return {
        Component.DRIVER.value: _driver_version(nvidia_version_file),
        Component.CUDA.value: cuda_version(),
        Component.KERNEL.value: platform.release() or UNKNOWN,
        Component.DGX_OS.value: _dgx_os_version(dgx_release_file),
        Component.LOCALMODELS.value: package_version(),
    }


def stale_components(recorded: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """One "name: recorded -> current" line per stale component (``STALE_COMPONENTS``) that differs."""
    return [
        f"{name}: {recorded.get(name, UNKNOWN)} -> {current.get(name, UNKNOWN)}"
        for name in map(str, STALE_COMPONENTS)
        if recorded.get(name, UNKNOWN) != current.get(name, UNKNOWN)
    ]


# The commit is stored as ``char const * LLAMA_COMMIT = "..."`` (common/build-info.cpp.in): a global pointer
# in the data segment that points at a string in .rodata. Searching the library for a hex run would hit
# unrelated constants, so the ELF symbol table finds the pointer and its relocation (or its stored value)
# gives the string. Only 64-bit little-endian ELF is read; anything else is rejected with ValueError.
_ELF_MAGIC: Final = b"\x7fELF"
_ELF64_LITTLE_ENDIAN: Final = (2, 1)  # EI_CLASS, EI_DATA
_ELF_HEADER: Final = struct.Struct("<HHIQQQIHHHHHH")  # e_type through e_shstrndx, after the 16-byte e_ident
_ELF_SECTION: Final = struct.Struct("<IIQQQQIIQQ")
_ELF_SEGMENT: Final = struct.Struct("<IIQQQQQQ")
_ELF_SYMBOL: Final = struct.Struct("<IBBHQQ")
_ELF_RELA: Final = struct.Struct("<QQq")
_POINTER: Final = struct.Struct("<Q")
_SHT_SYMTAB, _SHT_RELA, _SHT_DYNSYM = 2, 4, 11
_PT_LOAD: Final = 1
_RELATIVE_RELOCATION: Final = {62: 8, 183: 1027}  # e_machine (x86-64, AArch64) -> R_*_RELATIVE
_COMMIT_MAX_LEN: Final = 64


class _Elf:
    """The parts of a 64-bit little-endian ELF file needed to follow a global string pointer."""

    def __init__(self, data: bytes) -> None:
        if data[:4] != _ELF_MAGIC or (data[4], data[5]) != _ELF64_LITTLE_ENDIAN:
            msg = "not a 64-bit little-endian ELF file"
            raise ValueError(msg)
        self.data = data
        header = _ELF_HEADER.unpack_from(data, 16)
        self.machine = header[1]
        phoff, shoff = header[4], header[5]
        phentsize, phnum, shentsize, shnum = header[8:12]
        segments = [_ELF_SEGMENT.unpack_from(data, phoff + i * phentsize) for i in range(phnum)]
        #: (virtual address, bytes in the file, file offset) of each loaded segment
        self.loads: list[tuple[int, int, int]] = [
            (seg[3], seg[5], seg[2]) for seg in segments if seg[0] == _PT_LOAD
        ]
        self.sections: list[tuple[int, ...]] = [
            _ELF_SECTION.unpack_from(data, shoff + i * shentsize) for i in range(shnum)
        ]

    def _bytes(self, index: int) -> bytes:
        """The contents of section ``index``."""
        offset, size = self.sections[index][4:6]
        return self.data[offset : offset + size]

    def _tables(self, kind: int, entsize: int) -> Iterator[tuple[int, int]]:
        """(index, link) of each section of type ``kind`` whose entries are ``entsize`` bytes."""
        for index, section in enumerate(self.sections):
            if section[1] == kind and section[9] == entsize:
                yield index, section[6]

    def _symbol_address(self, name: str) -> int | None:
        """The address of the defined symbol ``name`` in a symbol table (.dynsym or .symtab)."""
        wanted = re.escape(name.encode() + b"\0")
        for kind in (_SHT_DYNSYM, _SHT_SYMTAB):
            for index, strtab in self._tables(kind, _ELF_SYMBOL.size):
                name_offsets = {m.start() for m in re.finditer(wanted, self._bytes(strtab))}
                for st_name, _i, _o, shndx, value, _size in _ELF_SYMBOL.iter_unpack(self._bytes(index)):
                    if st_name in name_offsets and shndx != 0:
                        return int(value)
        return None

    def _relocated_to(self, slot: int) -> int | None:
        """What a relative relocation makes the pointer at address ``slot`` point to."""
        relative = _RELATIVE_RELOCATION.get(self.machine)
        for index, _link in self._tables(_SHT_RELA, _ELF_RELA.size):
            for offset, info, addend in _ELF_RELA.iter_unpack(self._bytes(index)):
                if offset == slot and info & 0xFFFFFFFF == relative:
                    return int(addend)
        return None

    def _file_offset(self, vaddr: int) -> int | None:
        """Where in the file the loaded address ``vaddr`` is stored; None outside the file's segments."""
        for seg_vaddr, filesz, offset in self.loads:
            if seg_vaddr <= vaddr < seg_vaddr + filesz:
                return vaddr - seg_vaddr + offset
        return None

    def string_pointed_to_by(self, symbol: str) -> str | None:
        """The C string that the global ``char const *`` named ``symbol`` points at, or None."""
        slot = self._symbol_address(symbol)
        if slot is None:
            return None
        target = self._relocated_to(slot)
        if target is None:  # not position-independent: the pointer itself is in the file
            slot_offset = self._file_offset(slot)
            if slot_offset is None:
                return None
            (target,) = _POINTER.unpack_from(self.data, slot_offset)
        start = self._file_offset(target)
        end = -1 if start is None else self.data.find(b"\0", start, start + _COMMIT_MAX_LEN)
        if start is None or end <= start:
            return None
        return self.data[start:end].decode("ascii", errors="replace")


def llama_cpp_build(binary: Path) -> str:
    """The commit llama.cpp was built from, read from ``libllama-common.so`` beside ``binary`` or ``binary``.

    The file is read, never run (running a llama.cpp program starts CUDA on a shared card). "unknown" when
    neither file holds it.
    """
    for candidate in (binary.with_name(LLAMA_COMMON_LIBRARY), binary):
        try:
            commit = _Elf(candidate.read_bytes()).string_pointed_to_by(LLAMA_COMMIT_SYMBOL)
        except (OSError, ValueError, struct.error, IndexError):
            continue
        if commit:
            return commit
    return UNKNOWN
