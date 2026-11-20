"""`localmodels settings tune`: measure a project's in-process model on this machine and adopt the choice.

A project names its loader (``module:attribute``, a ``serving.loader.LoaderFactory``), the model's snapshot
and a file of its own texts. The candidates are every combination the loader declares
(``serving.loader.candidates``); each runs in a fresh ``python -m localmodels.serving.loader`` process under
``serving.tune.run_tune``, scoring the whole text set as one request; the fastest that agrees with the
reference by K6's score check and fits in memory is adopted through ``settings.record_from_tune`` and
``settings.adopt`` (design docs/research/in-process-tune-design-2026-11-20.md, section 2.3).
``in_process_recipe`` is the same command as a ``TuneRecipe``, for ``lookup``'s automatic re-tune.
Torch-free: only the child process imports PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from localmodels.enums import WorkloadKind
from localmodels.lease import DEFAULT_LEASE_DIR, GpuLease, QuietCard, QuietCardTimeoutError
from localmodels.pinned import snapshot_dir
from localmodels.pull import _default_hub_cache
from localmodels.serving.errors import ServingCancelled
from localmodels.serving.loader import SCORES_KEY, TEXTS_FILE, candidates, resolve_factory
from localmodels.serving.protocol import InferMessage, LoadMessage
from localmodels.serving.server import library_versions
from localmodels.serving.tune import (
    DEFAULT_HEADROOM_GIB,
    EXIT_ADOPTED,
    EXIT_FAILED,
    EXIT_NOT_ADOPTED,
    K6_MAX_ABS_DIFF,
    K6_MIN_SPEARMAN,
    TUNE_LEASE_PROJECT,
    NoEligibleCandidateError,
    ReferenceFailedError,
    TuneBudget,
    TuneCandidate,
    TuneRecord,
    free_margin_gib,
    measurement_line,
    run_tune,
    score_agreement,
)

if TYPE_CHECKING:
    import argparse

    from localmodels.settings import ModelIdentity, TuneInputs, TuneRecipe

DEFAULT_MIN_TEXTS: Final = 50  # fewer cannot tell a 0.99 rank agreement from noise (design 2.3)
DEFAULT_MAX_TEXTS: Final = 500  # the set is one request under the server's 900-s timeout, at up to 1 s a text
DEFAULT_MAX_CANDIDATES: Final = 12  # about what fits under the 40-minute ceiling at Mesmer 9B's measured cost
QUALITY_NAME: Final = (
    f"score_agreement({SCORES_KEY}, min_spearman={K6_MIN_SPEARMAN}, max_abs_diff={K6_MAX_ABS_DIFF})"
)


class TuneInputError(ValueError):
    """What the command was given cannot be tuned: the reason names what to fix."""


def loader_argv(spec: str, entry: str) -> list[str]:
    """How a candidate's model server is started: this interpreter running the loader server."""
    return [sys.executable, "-m", "localmodels.serving.loader", spec, entry]


def _snapshot(model: ModelIdentity) -> Path:
    """The model's snapshot folder in the Hub cache; refused when absent (the tune never downloads)."""
    folder = snapshot_dir(model.repository, model.revision, _default_hub_cache())
    if not folder.is_dir():
        msg = (
            f"{model.repository} at revision {model.revision} is not in the Hub cache ({folder}); "
            "download it before tuning"
        )
        raise TuneInputError(msg)
    return folder


def _read_texts(path: Path, minimum: int, maximum: int) -> TuneInputs:
    """Check the texts file (one ``{"id": str, "text": str}`` per line, unique ids, a count in range)."""
    from localmodels.settings import TuneInputs  # noqa: PLC0415 - the lookup's module, loaded by the command

    try:
        data = path.read_bytes()
    except OSError as exc:
        msg = f"the texts file {path} cannot be read: {exc}"
        raise TuneInputError(msg) from exc
    ids: set[str] = set()
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"{path} line {number} is not JSON ({exc.msg}); write one object per line"
            raise TuneInputError(msg) from exc
        if not (
            isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("text"), str)
        ):
            msg = f'{path} line {number} is not {{"id": str, "text": str}}'
            raise TuneInputError(msg)
        if row["id"] in ids:
            msg = f"{path} repeats the id {row['id']!r} (line {number}); give each text its own id"
            raise TuneInputError(msg)
        ids.add(row["id"])
    if not minimum <= len(ids) <= maximum:
        msg = (
            f"{path} holds {len(ids)} texts; a tune needs between {minimum} and {maximum} "
            "(--min-texts, --max-texts)"
        )
        raise TuneInputError(msg)
    return TuneInputs(sha256=hashlib.sha256(data).hexdigest(), count=len(ids), path=str(path.resolve()))


def cmd_settings_tune(args: argparse.Namespace) -> int:
    """`localmodels settings tune`: measure every declared candidate, then adopt the choice."""
    # the lookup's module and the re-tune's load only for a settings command (as cli.cmd_settings_show)
    from localmodels.retune import TUNE_LIMIT_S  # noqa: PLC0415
    from localmodels.settings import (  # noqa: PLC0415
        MIN_PASSES,
        AdoptError,
        ModelIdentity,
        adopt,
        record_from_tune,
    )

    started = time.monotonic()
    args.repeats = MIN_PASSES if args.repeats is None else args.repeats
    args.limit_s = TUNE_LIMIT_S if args.limit_s is None else args.limit_s
    try:
        model = ModelIdentity(repository=args.model, revision=args.revision)
        snapshot = _snapshot(model)
        texts = Path(args.texts)
        inputs = _read_texts(texts, args.min_texts, args.max_texts)
        declaration = resolve_factory(args.loader)(args.entry).declare()
        measured = candidates(declaration, args.max_candidates)
        budget = TuneBudget.for_job(started=started, limit_s=args.limit_s)
    except (
        ValueError
    ) as exc:  # refused input: identity, snapshot, texts, loader, declaration, candidates, limit
        sys.stderr.write(f"not tuned: {exc}\n")
        return EXIT_FAILED
    with tempfile.TemporaryDirectory(prefix="localmodels-tune-") as folder:
        shutil.copyfile(texts, Path(folder) / TEXTS_FILE)
        try:
            record = _run(args, snapshot, Path(folder), measured, budget)
            stored = adopt(
                record_from_tune(
                    record,
                    model=model,
                    kind=WorkloadKind(args.kind),
                    libraries=library_versions(),
                    recipe_project=args.project,
                    tune_job=os.environ.get("SLURM_JOB_ID"),
                    inputs=inputs,
                )
            )
        except (NoEligibleCandidateError, AdoptError) as exc:
            sys.stderr.write(f"measured, nothing adopted: {exc}\n")
            return EXIT_NOT_ADOPTED
        except (ReferenceFailedError, QuietCardTimeoutError) as exc:
            sys.stderr.write(f"not tuned: {exc}\n")
            return EXIT_FAILED
        except ServingCancelled:
            sys.stderr.write("not tuned: the tune was cancelled\n")
            return EXIT_FAILED
    sys.stdout.write(f"{args.entry} on {record.hardware} (reference {record.reference}, {record.quality})\n")
    sys.stdout.writelines(f"{measurement_line(m, stored.candidate)}\n" for m in record.measurements)
    return EXIT_ADOPTED


def _run(
    args: argparse.Namespace, snapshot: Path, folder: Path, measured: list[TuneCandidate], budget: TuneBudget
) -> TuneRecord:
    """Measure the checked candidates on the texts in ``folder``."""
    return run_tune(
        argv=loader_argv(args.loader, args.entry),
        env={},
        load=LoadMessage(model_dir=str(snapshot), revision=args.revision),
        inputs=[InferMessage(request_id="texts", input_dir=str(folder))],
        candidates=measured,
        quality=score_agreement(SCORES_KEY),
        quality_name=QUALITY_NAME,
        headroom_gib=free_margin_gib(args.headroom_gib),
        repeats=args.repeats,
        quiet_card=QuietCard(
            GpuLease(Path(args.lease_dir), TUNE_LEASE_PROJECT),
            reason=f"localmodels settings tune of {args.entry}",
        ),
        budget=budget,
    )


def in_process_recipe(
    *,
    project: str,
    loader: str,
    entry: str,
    model: ModelIdentity,
    kind: WorkloadKind,
    texts: Path,
    workdir: Path,
    reference: str,
) -> TuneRecipe:
    """The tune command as ``lookup``'s automatic re-tune runs it, in the project's environment and folder."""
    from localmodels.settings import TuneRecipe  # noqa: PLC0415 - a project building a recipe has the lookup loaded

    command = (
        str(Path(sys.executable).with_name("localmodels")),
        "settings",
        "tune",
        "--project",
        project,
        "--loader",
        loader,
        "--entry",
        entry,
        "--model",
        model.repository,
        "--revision",
        model.revision,
        "--kind",
        kind.value,
        "--texts",
        str(texts),
    )
    return TuneRecipe(
        project=project, command=command, workdir=workdir, reference=reference, quality=QUALITY_NAME
    )


def register_tune(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `settings tune` to the `settings` group."""
    p = commands.add_parser("tune", help="measure a project's in-process model here and adopt the choice")
    p.add_argument("--project", required=True, help="the project the record and the re-tune belong to")
    p.add_argument("--loader", required=True, help="MODULE:ATTRIBUTE of the project's loader factory")
    p.add_argument("--entry", required=True, help="the project's name for the model, passed to the factory")
    p.add_argument("--model", required=True, help="Hugging Face repository id")
    p.add_argument("--revision", required=True, help="the snapshot's 40-character commit")
    p.add_argument("--kind", required=True, choices=[k.value for k in WorkloadKind])
    p.add_argument(
        "--texts", required=True, help='JSONL of {"id": ..., "text": ...}, the project\'s own texts'
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="timed passes per candidate (default: settings.MIN_PASSES, 2)",
    )
    p.add_argument(
        "--headroom-gib",
        type=float,
        default=DEFAULT_HEADROOM_GIB,
        help="memory a candidate must leave free (never less than the host's kill floor)",
    )
    p.add_argument(
        "--limit-s",
        type=float,
        default=None,
        help="the job's ceiling (its timeout), in seconds (default: retune.TUNE_LIMIT_S, 2400, D19)",
    )
    p.add_argument("--min-texts", type=int, default=DEFAULT_MIN_TEXTS)
    p.add_argument("--max-texts", type=int, default=DEFAULT_MAX_TEXTS)
    p.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    p.add_argument("--lease-dir", default=str(DEFAULT_LEASE_DIR), help="the GPU lease folder (quiet request)")
    p.set_defaults(fn=cmd_settings_tune)
