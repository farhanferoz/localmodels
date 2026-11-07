"""`localmodels llama ...`: one llama.cpp build per commit, which of them is ``current``, and who names which.

Layout under the llama home (``~/cache/llama.cpp``, or ``$LOCALMODELS_LLAMA_HOME``; outside ``~/dev``, which
Syncthing copies to every machine):

- ``src/``: the one clone of llama.cpp that every build is made from.
- ``builds/<commit>/``: a CMake build folder, named by the short commit llama.cpp embeds in itself; the
  finished build holds ``bin/llama-server``. CMake build folders are not relocatable, so a build is made in
  place and a folder without ``bin/llama-server`` is an unfinished build.
- ``builds/current``: a symlink to one build folder; the build a project that follows ``current`` runs.
- ``builds/previous``: a symlink to the build ``current`` pointed to before the last switch; never pruned, so
  ``llama switch <it>`` rolls back.
- ``builds/users.toml``: ``[projects]`` with ``<project> = "<commit>"`` or ``"current"``. Each project's
  configuration step may update it; editing it by hand is fine. A missing file names nobody.
- ``builds/<commit>/comparison.json``: written by whoever ran the K1-K3 recipes on this build and on the
  baseline, and read by ``llama switch``. Fields: ``passed`` (bool: no quality failure and no slowdown),
  ``quality_failures`` (list of the failed checks, empty when none), ``slowdown`` (worst relative slowdown
  against the baseline, 0.03 is 3 % slower; 0 or less when none), ``baseline_commit`` (the build compared
  with), ``measured_at`` (ISO 8601 time of the comparison). Only ``passed`` decides. ``localmodels refresh``
  writes more fields beside these (the figures and the reasons); only these five are part of the contract.

``llama build`` compiles only: it never starts a GPU program, and nothing here runs a llama.cpp binary.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .atomic import write_text_atomically

if TYPE_CHECKING:
    import argparse
    from collections.abc import Collection, Mapping, Sequence

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1

HOME_ENV: Final = "LOCALMODELS_LLAMA_HOME"
DEFAULT_HOME: Final = Path.home() / "cache" / "llama.cpp"
#: The hand-made build that is in use today: its CMakeCache.txt is the configuration to reproduce, and
#: its llama-server is "current" until a build is switched to.
REFERENCE_BUILD: Final = Path.home() / "dev" / "llama.cpp" / "build"
REPO_URL: Final = "https://github.com/ggml-org/llama.cpp"
SERVER_BINARY: Final = Path("bin") / "llama-server"
CURRENT: Final = "current"
#: The build ``current`` pointed to before the last switch: kept by every prune, for a rollback.
PREVIOUS: Final = "previous"
#: Written by a switch the owner made although the comparison did not pass: their words and the refusal.
OVERRIDE_FILE: Final = "override.json"
USERS_FILE: Final = "users.toml"
COMPARISON_FILE: Final = "comparison.json"
#: A project that follows the ``current`` link.
FOLLOWS_CURRENT: Final = "current"
#: The commit llama.cpp embeds in itself is ``git rev-parse --short HEAD``.
SHORT_SHA_LENGTH: Final = 7

# CMakeCache.txt lines look like ``NAME:TYPE=value``; ``//`` lines are the help text of the entry below.
_CACHE_ENTRY: Final = re.compile(r"^(?P<name>[A-Za-z0-9_.+-]+):(?P<type>[A-Z]+)=(?P<value>.*)$")


class CacheType(StrEnum):
    """The CMake cache entry types this module reads."""

    #: ``cmake -DNAME=value`` on a name the project does not declare itself: the entry keeps this type.
    UNINITIALIZED = "UNINITIALIZED"


#: Options that were given on the command line but that llama.cpp declares itself (``option(GGML_CUDA ...)``
#: replaces the ``UNINITIALIZED`` type, so the cache no longer says they were set by hand). They are read
#: from the cache by name. Everything else in the cache is a default or an internal detection result.
DECLARED_OPTIONS: Final = ("CMAKE_BUILD_TYPE", "GGML_CUDA", "CMAKE_CUDA_COMPILER")
#: Binaries are unchanged by it: the test programs are not needed to run or compare a build.
EXTRA_OPTIONS: Final = ("LLAMA_BUILD_TESTS=OFF",)


def llama_home(home: str | Path | None = None) -> Path:
    """The llama home: ``home`` if given, else ``$LOCALMODELS_LLAMA_HOME``, else ``~/cache/llama.cpp``."""
    return Path(home or os.environ.get(HOME_ENV) or DEFAULT_HOME).expanduser()


def builds_dir(home: Path) -> Path:
    """The folder of the per-commit builds, ``current`` and ``users.toml``."""
    return home / "builds"


def server_binary(build: Path) -> Path:
    """Where a build folder holds its ``llama-server``."""
    return build / SERVER_BINARY


def current_server(home: Path, reference: Path = REFERENCE_BUILD) -> Path:
    """The ``llama-server`` of the ``current`` build, else of the hand-made reference build."""
    link = server_binary(builds_dir(home) / CURRENT)
    return link if link.is_file() else server_binary(reference)


def cmake_options(cache_file: Path) -> list[str]:
    """The ``-D`` arguments that reproduce the build configured in ``cache_file`` (a CMakeCache.txt).

    The options set on the command line (type ``UNINITIALIZED``) and ``DECLARED_OPTIONS``; not the other
    thousand entries, which are defaults and what CMake found on the machine.
    """
    chosen: dict[str, str] = {}
    for line in cache_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _CACHE_ENTRY.match(line)
        if match is None:
            continue
        name, kind, value = match.group("name", "type", "value")
        if kind == CacheType.UNINITIALIZED or name in DECLARED_OPTIONS:
            chosen[name] = (
                f"-D{name}={value}" if kind == CacheType.UNINITIALIZED else f"-D{name}:{kind}={value}"
            )
    return [*chosen.values(), *(f"-D{extra}" for extra in EXTRA_OPTIONS)]


def git_commands(commit: str, src: Path) -> list[list[str]]:
    """Clone (once), fetch ``commit`` and check it out detached, in order."""
    clone = [] if (src / ".git").exists() else [["git", "clone", "--no-checkout", REPO_URL, str(src)]]
    return [
        *clone,
        ["git", "-C", str(src), "fetch", "origin", commit],
        ["git", "-C", str(src), "checkout", "--detach", "FETCH_HEAD"],
    ]


def cmake_commands(src: Path, build: Path, options: Sequence[str], jobs: int) -> list[list[str]]:
    """Configure and compile ``src`` into ``build`` on ``jobs`` CPU cores."""
    return [
        ["cmake", "-S", str(src), "-B", str(build), *options],
        ["cmake", "--build", str(build), "--parallel", str(jobs)],
    ]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=False)  # noqa: S603


def _short_commit(src: Path) -> str:
    out = subprocess.run(  # noqa: S603
        ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def build_jobs() -> int:
    """The CPU cores this process may use: what slurm gave the job, not the machine's ``os.cpu_count()``."""
    return len(os.sched_getaffinity(0))


def compile_commit(home: Path, commit: str, reference: Path) -> Path:
    """Fetch ``commit`` and compile it unless its folder already holds a ``llama-server``; the build folder.

    Raises:
        subprocess.CalledProcessError: a git or cmake step failed.
        OSError: a step could not be started, or the folders could not be made.
    """
    src = home / "src"
    options = cmake_options(reference / "CMakeCache.txt")
    src.parent.mkdir(parents=True, exist_ok=True)
    for step in git_commands(commit, src):
        _run(step)
    build = builds_dir(home) / _short_commit(src)
    if not server_binary(build).is_file():
        for step in cmake_commands(src, build, options, build_jobs()):
            _run(step)
    return build


def cmd_build(args: argparse.Namespace) -> int:
    """Fetch and compile one commit into its build folder, then print the path of its ``llama-server``."""
    home = llama_home(args.home)
    reference = Path(args.reference).expanduser()
    out = sys.stdout
    if args.dry_run:
        # The folder is named by the commit git resolves after the fetch; the commit as given stands in.
        options = cmake_options(reference / "CMakeCache.txt")
        steps = [
            *git_commands(args.commit, home / "src"),
            *cmake_commands(home / "src", builds_dir(home) / args.commit, options, build_jobs()),
        ]
        out.writelines(f"{' '.join(step)}\n" for step in steps)
        out.write(f"# then: {server_binary(builds_dir(home) / args.commit)}\n")
        return EXIT_OK
    try:
        build = compile_commit(home, args.commit, reference)
    except (subprocess.CalledProcessError, OSError) as exc:
        sys.stderr.write(f"llama build failed: {exc}\n")
        return EXIT_ERROR
    out.write(f"{server_binary(build)}\n")
    return EXIT_OK


def read_users(home: Path) -> dict[str, str]:
    """``[projects]`` of ``users.toml``: project -> commit or ``current``; empty when there is no file."""
    path = builds_dir(home) / USERS_FILE
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        projects = tomllib.load(handle).get("projects", {})
    return {str(name): str(commit) for name, commit in projects.items()}


def current_commit(home: Path, link_name: str = CURRENT) -> str | None:
    """The build the ``current`` link (or ``link_name``) points to; None when there is no link."""
    link = builds_dir(home) / link_name
    return link.resolve().name if link.is_symlink() else None


def _point(folder: Path, link_name: str, target: str) -> None:
    """Make ``folder/link_name`` a symlink to ``target`` in one rename, so no reader sees it missing."""
    staging = folder / f".{link_name}.{os.getpid()}"
    staging.unlink(missing_ok=True)
    staging.symlink_to(target)
    staging.replace(folder / link_name)


def build_folders(home: Path) -> list[Path]:
    """The per-commit build folders (not the ``current`` link), sorted by name."""
    folder = builds_dir(home)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.is_symlink())


def folder_for(home: Path, commit: str) -> Path | None:
    """The build folder of ``commit``: the one whose name (a short commit) starts it; None when none does."""
    return next((folder for folder in build_folders(home) if commit.startswith(folder.name)), None)


def named_by(commit: str, current: str | None, users: Mapping[str, str]) -> list[str]:
    """The projects that name ``commit`` (a longer hash of it counts) or follow ``current`` while it is."""
    return sorted(
        project
        for project, named in users.items()
        if named.startswith(commit) or (named == FOLLOWS_CURRENT and commit == current)
    )


def cmd_list(args: argparse.Namespace) -> int:
    """Print each build, whether it is finished and ``current``, and the projects that name it."""
    home = llama_home(args.home)
    current, users = current_commit(home), read_users(home)
    for folder in build_folders(home):
        notes = [
            *([CURRENT] if folder.name == current else []),
            *([PREVIOUS] if folder.name == current_commit(home, PREVIOUS) else []),
            *([f"switched by override ({OVERRIDE_FILE})"] if (folder / OVERRIDE_FILE).is_file() else []),
            *([] if server_binary(folder).is_file() else ["unfinished"]),
        ]
        projects = named_by(folder.name, current, users)
        sys.stdout.write(
            f"{folder.name}  {', '.join(notes) or '-'}  projects: {', '.join(projects) or 'none'}\n"
        )
    return EXIT_OK


def comparison_passed(build: Path) -> str | None:
    """None when ``build`` has a comparison that passed, else why it may not be switched to."""
    path = build / COMPARISON_FILE
    if not path.is_file():
        return f"{path} does not exist: run the K1-K3 recipes on this build and the current one first"
    try:
        passed = json.loads(path.read_text(encoding="utf-8")).get("passed") is True
    except (OSError, ValueError, AttributeError):
        return f"{path} is not a JSON object"
    return None if passed else f"{path} does not record passed: true"


def switch(home: Path, commit: str, *, override: str = "") -> str | None:
    """Move ``current`` to a finished build whose comparison passed; None, or why it was not moved.

    ``override`` is the owner's decision to switch although the comparison did not pass: it is written to
    ``override.json`` beside the comparison, which stays as measured. It never stands in for a comparison
    that was not made. The build ``current`` left becomes ``previous``.
    """
    build = builds_dir(home) / commit
    if not server_binary(build).is_file():
        return f"{build} is not a finished build"
    problem = comparison_passed(build)
    if problem is not None:
        if not override or not (build / COMPARISON_FILE).is_file():
            return problem
        record = {
            "override": override,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "refused": problem,
        }
        write_text_atomically(build / OVERRIDE_FILE, json.dumps(record, indent=2) + "\n")
    before = current_commit(home)
    if before is not None and before != build.name:
        _point(build.parent, PREVIOUS, before)
    _point(build.parent, CURRENT, build.name)
    return None


def cmd_switch(args: argparse.Namespace) -> int:
    """Move the ``current`` link to a finished build whose comparison passed."""
    problem = switch(llama_home(args.home), args.commit, override=args.override or "")
    if problem is not None:
        sys.stderr.write(f"not switching: {problem}\n")
        return EXIT_ERROR
    sys.stdout.write(f"current -> {args.commit}\n")
    return EXIT_OK


def prune_builds(home: Path, *, keep: Collection[str] = (), dry_run: bool = False) -> list[Path]:
    """The build folders that are not ``current``, ``previous``, in ``keep`` or named by a project: deleted.

    With ``dry_run`` they are only listed.
    """
    current, users = current_commit(home), read_users(home)
    kept = {current, current_commit(home, PREVIOUS), *keep}
    doomed = [
        folder
        for folder in build_folders(home)
        if folder.name not in kept and not named_by(folder.name, current, users)
    ]
    if not dry_run:
        for folder in doomed:
            shutil.rmtree(folder)
    return doomed


def cmd_prune(args: argparse.Namespace) -> int:
    """Delete the builds that are not ``current`` and that no project names."""
    for folder in prune_builds(llama_home(args.home), dry_run=args.dry_run):
        sys.stdout.write(f"{'would remove' if args.dry_run else 'removed'} {folder}\n")
    return EXIT_OK


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``llama build|list|switch|prune`` to the ``localmodels`` command."""
    llama = subparsers.add_parser("llama", help="llama.cpp builds, one per commit")
    llama.add_argument("--home", help=f"llama home (default: ${HOME_ENV}, else {DEFAULT_HOME})")
    commands = llama.add_subparsers(dest="llama_command", required=True)

    build = commands.add_parser(
        "build", help="compile one commit into builds/<commit> and print its llama-server"
    )
    build.add_argument("--commit", required=True, help="commit (or branch) of ggml-org/llama.cpp")
    build.add_argument(
        "--reference", default=str(REFERENCE_BUILD), help="build whose CMakeCache.txt to reproduce"
    )
    build.add_argument("--dry-run", action="store_true", help="print the git and cmake commands and stop")
    build.set_defaults(fn=cmd_build)

    commands.add_parser("list", help="the builds, `current` and the projects that name each").set_defaults(
        fn=cmd_list
    )

    switch = commands.add_parser("switch", help="move `current` to a build whose comparison passed")
    switch.add_argument("commit", help="build folder name")
    switch.add_argument(
        "--override",
        metavar="WORDS",
        help="the owner's decision to switch although the comparison did not pass (kept in override.json)",
    )
    switch.set_defaults(fn=cmd_switch)

    prune = commands.add_parser(
        "prune", help="delete builds that are neither current nor named in users.toml"
    )
    prune.add_argument("--dry-run", action="store_true", help="list what would be deleted")
    prune.set_defaults(fn=cmd_prune)
