"""Process arguments, binary resolution and llama-server argv construction, torch-free."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import ServingParams


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
        "--jinja",
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        *extra_flags,
    ]


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
