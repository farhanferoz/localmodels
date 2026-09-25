"""Process arguments, binary resolution and llama-server argv construction, torch-free."""

from __future__ import annotations

import contextlib
import ipaddress
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
        "--jinja",
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        *(["--api-key-file", str(spec.api_key_file)] if spec.api_key_file is not None else []),
        *extra_flags,
    ]


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
