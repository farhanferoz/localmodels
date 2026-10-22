"""`localmodels`: `setup`, `serve`, and `lease {status,acquire,wait,begin-wait,admitted,release}`.

The `lease` subcommands carry B12.3's DEVIATION: service_c's bash queue writes/reads GPU
lease files through inline ``python3 -c`` (``scripts/run_gpu_queue.sh``); these give it the
same operations as a command with documented exit codes, so it can shell out instead.

Exit codes (stable; a caller scripts against these, not against stdout text):
    0  EXIT_OK              the operation succeeded.
    1  EXIT_ERROR           an unexpected error (bad arguments, I/O failure, a failed step).
    2  EXIT_LEASE_CONFLICT  `lease acquire`: another project's lease is already live.
    3  EXIT_LEASE_TIMEOUT   `lease wait`: timed out before the GPU lease became free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from .catalogue import ModelSpec
from .digest import register as register_digest
from .enums import WorkloadKind
from .lease import DEFAULT_LEASE_DIR, GpuLease, GpuLeaseClient, LeaseRecord
from .llama_builds import register as register_llama
from .rules import register as register_rules
from .service import _cmd_serve
from .setup import (
    DEFAULT_MODELS_DIR,
    DEFAULT_PROJECT,
    DEFAULT_STATE_PATH,
    Role,
    print_summary,
    record_setup,
    run_setup,
)
from .stack import Engine

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LEASE_CONFLICT = 2
EXIT_LEASE_TIMEOUT = 3


def _lease_from_args(args: argparse.Namespace) -> GpuLease:
    return GpuLease(Path(args.lease_dir), project=args.project)


def cmd_lease_status(args: argparse.Namespace) -> int:
    """Report this project's own lease record plus every other live one, then any stay-up records."""
    lease = _lease_from_args(args)
    mine = lease.read_own()
    others = lease.live_others()
    report = {
        "project": lease.project,
        "mine": mine.model_dump(exclude_none=True) if mine else None,
        "others": [o.model_dump(exclude_none=True) for o in others],
    }
    # Stay-up records only when there are some: the output of a host that does not use the mode is unchanged.
    if residents := lease.residents():
        report["residents"] = [r.model_dump() for r in residents]
    if quiet := lease.quiet_requests():
        report["quiet"] = [q.model_dump() for q in quiet]
    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
    else:
        print(text)
    return EXIT_OK


def cmd_lease_acquire(args: argparse.Namespace) -> int:
    """Write this project's lease immediately; refuse (EXIT_LEASE_CONFLICT) if another is live."""
    lease = _lease_from_args(args)
    others = lease.live_others()
    if others:
        holders = ", ".join(o.project for o in others)
        print(f"GPU lease held by: {holders}", file=sys.stderr)
        return EXIT_LEASE_CONFLICT
    lease.write(
        LeaseRecord(
            pid=args.pid,
            project=args.project,
            port=args.port,
            model_path=args.model_path,
            started_at=int(time.time()),
        )
    )
    return EXIT_OK


def cmd_lease_wait(args: argparse.Namespace) -> int:
    """Poll until it is this project's turn at the GPU, or EXIT_LEASE_TIMEOUT after ``--timeout``.

    After `lease begin-wait`, the turn comes in request order (`GpuLeaseClient`): holders and earlier waiters
    block, later waiters do not. Without a waiting lease of its own, every other live lease blocks.
    """
    lease = _lease_from_args(args)
    own = lease.read_own()
    client = GpuLeaseClient.resuming(lease, own) if own is not None else GpuLeaseClient(lease)
    deadline = time.time() + args.timeout if args.timeout is not None else None
    if client.wait_turn(args.poll, lambda: deadline is not None and time.time() > deadline):
        return EXIT_OK
    print(f"timed out after {args.timeout}s waiting for the GPU lease", file=sys.stderr)
    return EXIT_LEASE_TIMEOUT


def cmd_lease_begin_wait(args: argparse.Namespace) -> int:
    """Write this project's lease with ``requested_at`` set: "I am waiting for the GPU"."""
    lease = _lease_from_args(args)
    lease.write(
        LeaseRecord(
            pid=args.pid,
            project=args.project,
            job_id=args.job_id,
            lane=args.lane,
            requested_at=int(time.time()),
        )
    )
    return EXIT_OK


def cmd_lease_admitted(args: argparse.Namespace) -> int:
    """Rewrite this project's lease without ``requested_at``: it now holds the GPU."""
    lease = _lease_from_args(args)
    lease.write(
        LeaseRecord(
            pid=args.pid,
            project=args.project,
            port=args.port,
            model_path=args.model_path,
            job_id=args.job_id,
            lane=args.lane,
            started_at=int(time.time()),
        )
    )
    return EXIT_OK


def cmd_lease_release(args: argparse.Namespace) -> int:
    """Remove this project's lease file; idempotent when there is nothing to remove."""
    _lease_from_args(args).remove()
    return EXIT_OK


def _load_catalogue(path: str) -> list[ModelSpec]:
    """Load a caller's own catalogue from a JSON file (a list of `ModelSpec`-shaped objects)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TypeAdapter(list[ModelSpec]).validate_python(data)


def cmd_setup(args: argparse.Namespace) -> int:
    """`localmodels setup`: run the bootstrap pipeline and print a plain-language summary."""
    catalogue = _load_catalogue(args.catalogue) if args.catalogue else None
    lease = GpuLease(Path(args.lease_dir), project=args.project)
    report = run_setup(
        role=Role(args.role),
        models_dir=Path(args.models_dir),
        catalogue=catalogue,
        lease=lease,
        dry_run=args.dry_run,
        yes=args.yes,
        device_override=args.device,
        binary=Path(args.llama_server_bin) if args.llama_server_bin else None,
    )
    print_summary(report, stream=sys.stdout)
    if not args.dry_run:
        record_setup(report, state_path=Path(args.state_path))
    return EXIT_OK if report.ok else EXIT_ERROR


def cmd_serve(args: argparse.Namespace) -> int:
    """`localmodels serve`: run and supervise one configured llama-server instance."""
    return _cmd_serve(Path(args.config))


def cmd_settings_show(args: argparse.Namespace) -> int:
    """`localmodels settings show`: print one settings answer without loading or queueing anything."""
    from .settings import cmd_settings_show as run  # noqa: PLC0415 - only this command loads the lookup

    return run(args)


def cmd_settings_check(args: argparse.Namespace) -> int:
    """`localmodels settings check`: list the stored records that are stale on this host."""
    from .settings import cmd_settings_check as run  # noqa: PLC0415 - only this command loads the lookup

    return run(args)


def _add_lease_common_args(parser: argparse.ArgumentParser, *, with_pid: bool) -> None:
    parser.add_argument("--project", required=True, help="this project's name, e.g. 'service_a'")
    parser.add_argument("--lease-dir", default=str(DEFAULT_LEASE_DIR), help="shared GPU lease directory")
    if with_pid:
        parser.add_argument(
            "--pid",
            required=True,
            type=int,
            help="the PID that lives exactly as long as this lease (the caller's own $$, "
            "never this CLI process's PID, which exits immediately after writing)",
        )


def _build_lease_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    lease_p = subparsers.add_parser("lease", help="GPU lease operations (service_c's bash queue contract)")
    lease_sub = lease_p.add_subparsers(dest="lease_command", required=True)

    status_p = lease_sub.add_parser("status", help="report this project's lease and every other live one")
    _add_lease_common_args(status_p, with_pid=False)
    status_p.add_argument("--json-out", default=None, help="write the report here instead of stdout")
    status_p.set_defaults(fn=cmd_lease_status)

    acquire_p = lease_sub.add_parser(
        "acquire", help="acquire the GPU lease now, or refuse if another is live"
    )
    _add_lease_common_args(acquire_p, with_pid=True)
    acquire_p.add_argument("--port", type=int, default=None)
    acquire_p.add_argument("--model-path", default=None)
    acquire_p.set_defaults(fn=cmd_lease_acquire)

    wait_p = lease_sub.add_parser("wait", help="poll until the GPU lease is free")
    _add_lease_common_args(wait_p, with_pid=False)
    wait_p.add_argument("--timeout", type=float, default=None, help="seconds; omit to wait forever")
    wait_p.add_argument("--poll", type=float, default=2.0, help="poll interval in seconds")
    wait_p.set_defaults(fn=cmd_lease_wait)

    begin_wait_p = lease_sub.add_parser("begin-wait", help="record that this project is waiting for the GPU")
    _add_lease_common_args(begin_wait_p, with_pid=True)
    begin_wait_p.add_argument("--job-id", default=None)
    begin_wait_p.add_argument("--lane", default=None)
    begin_wait_p.set_defaults(fn=cmd_lease_begin_wait)

    admitted_p = lease_sub.add_parser("admitted", help="record that this project now holds the GPU")
    _add_lease_common_args(admitted_p, with_pid=True)
    admitted_p.add_argument("--port", type=int, default=None)
    admitted_p.add_argument("--model-path", default=None)
    admitted_p.add_argument("--job-id", default=None)
    admitted_p.add_argument("--lane", default=None)
    admitted_p.set_defaults(fn=cmd_lease_admitted)

    release_p = lease_sub.add_parser("release", help="remove this project's lease file")
    _add_lease_common_args(release_p, with_pid=False)
    release_p.set_defaults(fn=cmd_lease_release)


def _build_setup_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("setup", help="one-command hardware-aware bootstrap for a local chat model")
    p.add_argument("--role", default=Role.CHAT.value, choices=[r.value for r in Role])
    p.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    p.add_argument(
        "--catalogue",
        default=None,
        help="JSON file of catalogue entries; default: this package's bootstrap catalogue",
    )
    p.add_argument("--yes", action="store_true", help="proceed past the download/bench plan")
    p.add_argument("--dry-run", action="store_true", help="report the plan; make no changes")
    p.add_argument("--project", default=DEFAULT_PROJECT, help="this run's GPU lease project name")
    p.add_argument("--lease-dir", default=str(DEFAULT_LEASE_DIR))
    p.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    p.add_argument("--device", default=None, help="override the measured device (testing/manual use)")
    p.add_argument(
        "--llama-server-bin",
        default=None,
        help="explicit llama-server binary path (also resolved via $LLAMA_SERVER_BIN, then PATH)",
    )
    p.set_defaults(fn=cmd_setup)


def _build_serve_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("serve", help="run and supervise one configured llama-server instance")
    p.add_argument("--config", required=True, help="path to a ServeConfig TOML file's [server] table")
    p.set_defaults(fn=cmd_serve)


def _build_settings_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    settings_p = subparsers.add_parser("settings", help="the measured settings a model should load with")
    commands = settings_p.add_subparsers(dest="settings_command", required=True)
    show_p = commands.add_parser("show", help="print the settings answer for one model; loads nothing")
    show_p.add_argument("--model", required=True, help="Hugging Face repository id")
    show_p.add_argument(
        "--revision", default=None, help="pinned revision (40-character commit); or give --file"
    )
    show_p.add_argument(
        "--file",
        action="append",
        help="a GGUF file of the model in the Hub cache (repeat for split parts); needs --revision",
    )
    show_p.add_argument("--kind", required=True, choices=[k.value for k in WorkloadKind])
    show_p.add_argument("--engine", default=Engine.PYTORCH.value, choices=[e.value for e in Engine])
    show_p.add_argument("--build", default=None, help="the llama.cpp build the caller will run")
    show_p.add_argument(
        "--legacy", default=None, help="the caller's own tune file, read when the store has none"
    )
    show_p.set_defaults(fn=cmd_settings_show)
    check_p = commands.add_parser("check", help="list stored records stale on this host; exit 1 when any")
    check_p.add_argument("--home", default=None, help="llama home, where the `current` build is")
    check_p.set_defaults(fn=cmd_settings_check)


def build_parser() -> argparse.ArgumentParser:
    """Build the `localmodels` argparse parser: `setup`, `serve`, `lease ...`, `rules ...`, `settings ...`."""
    parser = argparse.ArgumentParser(prog="localmodels")
    subparsers = parser.add_subparsers(dest="command")
    _build_setup_parser(subparsers)
    _build_serve_parser(subparsers)
    _build_lease_parser(subparsers)
    register_digest(subparsers)
    register_rules(subparsers)
    register_llama(subparsers)
    _build_settings_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `localmodels` console script; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    fn: Callable[[argparse.Namespace], int] | None = getattr(args, "fn", None)
    if fn is None:
        parser.print_usage(sys.stderr)
        return EXIT_ERROR
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
