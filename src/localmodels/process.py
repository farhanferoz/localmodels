"""Process arguments, binary resolution and llama-server argv construction, torch-free."""

from __future__ import annotations

import contextlib
import ipaddress
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .params import ServingParams
    from .stack import Settings


class LocalModelsError(RuntimeError):
    """Base for this package's own errors; a boundary never raises a bare Exception."""


class BinaryNotFoundError(LocalModelsError):
    """No llama-server binary could be resolved through any of the tried locations."""


def _process_args(pid: int) -> list[str]:
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.exists():
        with contextlib.suppress(OSError):
            raw = proc_cmdline.read_bytes()
            parts = [p.decode(errors="ignore") for p in raw.split(b"\x00") if p]
            if parts:
                return parts
    try:
        out = subprocess.check_output(  # noqa: S603
            ["ps", "-ww", "-o", "args=", "-p", str(pid)],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().split()
    except (subprocess.SubprocessError, OSError):
        return []


def _is_pid_alive(pid: int) -> bool:
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
    except (ChildProcessError, OSError):
        pass

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    stat_file = Path(f"/proc/{pid}/stat")
    if stat_file.exists():
        with contextlib.suppress(OSError):
            content = stat_file.read_text().split(")")[-1].strip().split()
            if content and content[0] == "Z":
                return False

    args = _process_args(pid)
    return bool(args and not any("defunct" in a for a in args))


def resolve_llama_server_bin(
    *,
    env_var: str = "LLAMA_SERVER_BIN",
    explicit: Path | None = None,
    lib_dir: Path | None = None,
) -> Path:
    """Resolve the llama-server binary: env var, then an explicit path, then a lib dir, then PATH."""
    tried: list[str] = []

    env_bin = os.environ.get(env_var)
    if env_bin:
        tried.append(f"${env_var} ({env_bin})")
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return p

    if explicit is not None:
        tried.append(f"explicit ({explicit})")
        if explicit.is_file() and os.access(explicit, os.X_OK):
            return explicit

    if lib_dir is not None:
        lib_bin = lib_dir / "llama-server"
        tried.append(f"lib_dir ({lib_bin})")
        if lib_bin.is_file() and os.access(lib_bin, os.X_OK):
            return lib_bin

    which_bin = shutil.which("llama-server")
    if which_bin:
        tried.append(f"PATH ('{which_bin}')")
        return Path(which_bin)
    tried.append("PATH lookup ('llama-server')")

    msg = f"could not resolve llama-server binary; tried: {', '.join(tried)}"
    raise BinaryNotFoundError(msg)


@dataclass(frozen=True)
class ServerSpec:
    """Specification of a local server instance to launch."""

    host: str
    port: int
    model_path: Path
    #: llama-server's ``--api-key-file``; required whenever ``host`` is not loopback.
    api_key_file: Path | None = None


def build_args(
    binary: str | Path,
    spec: ServerSpec,
    params: ServingParams,
    extra_flags: tuple[str, ...] = (),
) -> list[str]:
    """Construct command-line arguments for llama-server.

    ``extra_flags`` carries a caller's own additions (e.g. service_c's ``--spec-type`` options)
    appended after the shared flags, so a caller never has to reimplement this function to add one.
    """
    ngl = "99" if params.n_gpu_layers == -1 else str(params.n_gpu_layers)
    return [
        str(binary),
        "-m",
        str(spec.model_path),
        "-ngl",
        ngl,
        "-c",
        str(params.n_ctx),
        "-np",
        str(params.n_parallel),
        *(["-kvu"] if params.kv_unified else []),
        "--jinja",
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        *(["--api-key-file", str(spec.api_key_file)] if spec.api_key_file is not None else []),
        *extra_flags,
    ]


# The flag and variable names below are read from llama.cpp 5aeac37a3, common/arg.cpp: `-b` and `-ub` at
# 1672-1684; the speculation options (those marked `.set_spec()`, their negated names such as
# `--no-spec-draft-backend-sampling` at 4234-4236, `--spec-default`, and the n-gram cache files `-lcs`
# and `-lcd` at 1627-1640, which set `params.speculative.ngram_cache`) at 3996-4780. Re-derive them when
# the build in use moves on.
BATCH_FLAGS: Final = ("-b", "--batch-size")
BATCH_ENV: Final = "LLAMA_ARG_BATCH"
UBATCH_FLAGS: Final = ("-ub", "--ubatch-size")
UBATCH_ENV: Final = "LLAMA_ARG_UBATCH"
DRAFT_MODEL_FLAGS: Final = ("--spec-draft-model", "-md", "--model-draft")
DRAFT_MODEL_ENV: Final = "LLAMA_ARG_SPEC_DRAFT_MODEL"
SPEC_TYPE_FLAG: Final = "--spec-type"
SPEC_FLAGS: Final = (
    *DRAFT_MODEL_FLAGS,
    SPEC_TYPE_FLAG,
    "--spec-default",
    "--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft",
    "--spec-draft-threads", "-td", "--threads-draft",
    "--spec-draft-threads-batch", "-tbd", "--threads-batch-draft",
    "--spec-draft-cpu-mask", "-Cd", "--cpu-mask-draft",
    "--spec-draft-cpu-range", "-Crd", "--cpu-range-draft",
    "--spec-draft-cpu-strict", "--cpu-strict-draft",
    "--spec-draft-prio", "--prio-draft",
    "--spec-draft-poll", "--poll-draft",
    "--spec-draft-cpu-mask-batch", "-Cbd", "--cpu-mask-batch-draft",
    "--spec-draft-cpu-range-batch", "-Crbd", "--cpu-range-batch-draft",
    "--spec-draft-cpu-strict-batch", "--cpu-strict-batch-draft",
    "--spec-draft-prio-batch", "--prio-batch-draft",
    "--spec-draft-poll-batch", "--poll-batch-draft",
    "--spec-draft-type-k", "-ctkd", "--cache-type-k-draft",
    "--spec-draft-type-v", "-ctvd", "--cache-type-v-draft",
    "--spec-draft-override-tensor", "-otd", "--override-tensor-draft",
    "--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft",
    "--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft",
    "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-n-ubatch",
    "--spec-synth-len", "--spec-synth-rates",
    "--spec-draft-p-split", "--draft-p-split", "--spec-draft-p-min", "--draft-p-min",
    "--spec-draft-backend-sampling", "--spec-draft-sampling",
    "--spec-draft-device", "-devd", "--device-draft",
    "--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft",
    "--draft", "--draft-n", "--draft-max", "--draft-min", "--draft-n-min",
    "--spec-ngram-mod-n-min", "--spec-ngram-mod-n-max", "--spec-ngram-mod-n-match",
    "--spec-ngram-simple-size-n", "--spec-ngram-simple-size-m", "--spec-ngram-simple-min-hits",
    "--spec-ngram-map-k-size-n", "--spec-ngram-map-k-size-m", "--spec-ngram-map-k-min-hits",
    "--spec-ngram-map-k4v-size-n", "--spec-ngram-map-k4v-size-m", "--spec-ngram-map-k4v-min-hits",
    "--spec-ngram-size-n", "--spec-ngram-size-m", "--spec-ngram-min-hits",
    "--no-spec-draft-backend-sampling",
    "-lcs", "--lookup-cache-static", "-lcd", "--lookup-cache-dynamic",
)  # fmt: skip
SPEC_ENV_PREFIX: Final = "LLAMA_ARG_SPEC_"
SPEC_ENV: Final = ("LLAMA_ARG_DRAFT_MAX", "LLAMA_ARG_DRAFT_MIN", "LLAMA_ARG_N_GPU_LAYERS_DRAFT")


def _split_flag(token: str) -> tuple[str, str, str]:
    """``(name, "=" or "", value)`` of one argument, reading the name as llama.cpp does.

    llama.cpp turns every underscore of an argument that starts with ``--`` into a dash before it looks the
    name up (arg.cpp:826-829), so ``--ubatch_size`` is ``--ubatch-size``. It rejects ``--flag=value``; that
    form is read here all the same.
    """
    name, equals, value = token.partition("=")
    return (name.replace("_", "-") if name.startswith("--") else name), equals, value


def llama_option(
    flags: tuple[str, ...], env: Mapping[str, str] | None, names: tuple[str, ...], env_name: str
) -> str | None:
    """The value a caller's flags (the last one wins, as in llama.cpp) or else its environment give an option.

    ``--flag value`` and ``--flag=value`` both count; None when neither the flags nor the environment name it.
    """
    value = (env or {}).get(env_name)
    for i, token in enumerate(flags):
        name, equals, tail = _split_flag(token)
        if name in names and equals:
            value = tail
        elif name in names and i + 1 < len(flags):
            value = flags[i + 1]
    return value


def _carries(flags: tuple[str, ...], names: tuple[str, ...]) -> bool:
    return any(_split_flag(token)[0] in names for token in flags)


def carries_ubatch(flags: tuple[str, ...], env: Mapping[str, str] | None) -> bool:
    """Whether the caller's own flags or environment already set the prompt batch size (``-ub``)."""
    return _carries(flags, UBATCH_FLAGS) or UBATCH_ENV in (env or {})


def carries_speculation(flags: tuple[str, ...], env: Mapping[str, str] | None) -> bool:
    """Whether the caller's flags or environment hold any speculative-decoding option, of any kind."""
    return _carries(flags, SPEC_FLAGS) or any(
        name.startswith(SPEC_ENV_PREFIX) or name in SPEC_ENV for name in (env or {})
    )


def adds_ubatch(settings: Settings, flags: tuple[str, ...], env: Mapping[str, str] | None) -> bool:
    """Whether ``llama_flags`` adds ``-ub``: the answer has one, the caller gave none and no smaller ``-b``.

    llama.cpp runs ``min(n_batch, n_ubatch)`` (llama-context.cpp:334), so a caller's own smaller batch size
    would cap the answer's ``-ub``: the caller has decided, and the answer's value is not passed.
    """
    if settings.n_ubatch is None or carries_ubatch(flags, env):
        return False
    batch = llama_option(flags, env, BATCH_FLAGS, BATCH_ENV)
    return batch is None or not batch.isdigit() or int(batch) >= settings.n_ubatch


def speculates(settings: Settings) -> bool:
    """Whether ``settings`` ask for speculative decoding (a type other than ``none``)."""
    from .stack import SpecType  # noqa: PLC0415 - stack is not part of ``import localmodels``

    return settings.spec is not None and settings.spec is not SpecType.NONE


def llama_flags(
    settings: Settings, extra_flags: tuple[str, ...], extra_env: Mapping[str, str] | None
) -> tuple[str, ...]:
    """The llama-server flags a settings answer adds to a caller's own: ``-ub`` and speculation.

    Each is one group, left out whole when the caller's flags or environment already carry any of it: a
    caller's ``--spec-type`` beside a record's draft file would start a speculation nobody chose. ``none``
    adds nothing: it is llama.cpp's default.
    """
    flags: list[str] = []
    if adds_ubatch(settings, extra_flags, extra_env):
        flags += [UBATCH_FLAGS[0], str(settings.n_ubatch)]
    if speculates(settings) and not carries_speculation(extra_flags, extra_env):
        assert settings.spec is not None  # noqa: S101 - narrowing for mypy; ``speculates`` checked
        flags += [SPEC_TYPE_FLAG, str(settings.spec)]
        if settings.spec_draft_file is not None:
            flags += [DRAFT_MODEL_FLAGS[0], settings.spec_draft_file]
    return tuple(flags)


class ApiKeyRequiredError(LocalModelsError):
    """Serving on a non-loopback address without ``api_key_file``."""


def require_key_off_loopback(spec: ServerSpec) -> None:
    """Refuse to serve off loopback without a key: llama-server has no other access control.

    Reachable from a non-loopback address (a docker bridge included), any container or process on
    the host could otherwise fill its slots.
    """
    if spec.api_key_file is not None or spec.host == "localhost":
        return
    try:
        loopback = ipaddress.ip_address(spec.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        msg = f"api_key_file is required to serve on {spec.host} (not loopback)"
        raise ApiKeyRequiredError(msg)


def auth_headers(api_key_file: Path | None) -> dict[str, str]:
    """The ``Authorization`` header for a keyed llama-server; empty when no key is configured.

    llama-server exempts only ``/health`` and ``/v1/health`` from the key
    (``tools/server/server-http.cpp`` at ``ccc3646``), so every ``/props`` probe needs this. The key
    is the file's first line that is not blank or a ``#`` comment, llama-server's own rule.
    """
    if api_key_file is None:
        return {}
    for line in api_key_file.read_text().splitlines():
        if line and not line.startswith("#"):
            return {"Authorization": f"Bearer {line}"}
    msg = f"{api_key_file} holds no key"
    raise ApiKeyRequiredError(msg)


@dataclass(frozen=True)
class ServerHandle:
    """Handle to a started or adopted llama-server process."""

    pid: int
    port: int
    model_path: Path


@dataclass(frozen=True)
class ServerStatus:
    """Status information for a local server endpoint."""

    port: int
    pid: int | None
    alive: bool
    healthy: bool
    model_path: str
