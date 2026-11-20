"""A model server around a project's own loader, and the candidates a tune takes from its declaration.

A project writes a ``ScorerLoader``: ``declare`` says which precisions and options its model accepts (and
which setting is the reference), ``load`` returns a scorer, texts in and one float per text out.
``LoaderServer`` is the child process of an in-process tune: ``python -m localmodels.serving.loader
<module:attribute> <entry>``, run with the project's interpreter. It is a ``ModelServer``, so the protocol,
error handling and network block of ``serving.server`` apply unchanged. The manager's side
(``resolve_factory``, ``candidates``) is torch-free; PyTorch is imported only inside the child's ``load`` and
``infer``, as ``server.open_load`` does.

The child servers run with ``client.MODEL_SERVER_ENV`` (``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``);
a project's production process should set the same, or its memory differs from what the tune measured.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from localmodels.enums import Precision  # noqa: TC001 - pydantic needs it at runtime
from localmodels.serving.errors import ServingError, ServingErrorCode
from localmodels.serving.protocol import InferMessage, LoadedMessage, LoadMessage, ResultMessage
from localmodels.serving.server import ModelServer, ProgressEmitter, open_load, run
from localmodels.serving.tune import GIB, TuneCandidate
from localmodels.stack import Settings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type JsonScalar = bool | int | float | str
type TextScorer = Callable[[Sequence[str]], Sequence[float]]

# protocol facts between the tune's manager and this server, not settings
TEXTS_FILE: Final = "texts.jsonl"  # in the request's input_dir: one {"id": str, "text": str} per line
SCORES_KEY: Final = "scores"  # the name -> float map in the result's native_outputs, as score_agreement reads
_CUDA_DEVICE: Final = "cuda"  # the only device a model server runs on (open_load refuses without a GPU)
#: ``Settings`` fields that belong to llama.cpp's tune, never to a loader's reference
_LLAMA_FIELDS: Final = frozenset({"n_ubatch", "spec", "spec_draft_file", "model_file"})


class TooManyCandidates(ValueError):  # noqa: N818 - reads as a refusal of the declaration, not an error state
    """A declaration whose product of values is more than the tune may measure."""


def _same(a: JsonScalar, b: JsonScalar) -> bool:
    """Equal values of one type: ``True`` is not ``1``."""
    return type(a) is type(b) and a == b


def _declared(value: JsonScalar, values: Sequence[JsonScalar]) -> bool:
    return any(_same(value, v) for v in values)


class LoaderDeclaration(BaseModel):
    """What a loader accepts: precisions, per-option values, and the explicit reference setting.

    The reference is the highest-precision, default setting; it is stated, never inferred from list order.
    """

    model_config = ConfigDict(frozen=True)

    dtypes: tuple[Precision, ...]
    options: dict[str, tuple[JsonScalar, ...]]
    reference: Settings

    @model_validator(mode="after")
    def _consistent(self) -> LoaderDeclaration:
        if not self.dtypes:
            msg = "A loader must declare at least one dtype."
            raise ValueError(msg)
        for name, values in self.options.items():
            if not values:
                msg = f"Option {name!r} declares no values; declare at least one."
                raise ValueError(msg)
        llama_fields = sorted(f for f in _LLAMA_FIELDS if getattr(self.reference, f) is not None)
        if llama_fields:
            msg = (
                f"The reference sets llama.cpp fields {llama_fields}; a loader's has dtype and options only."
            )
            raise ValueError(msg)
        if self.reference.dtype not in self.dtypes:
            msg = (
                f"The reference dtype {self.reference.dtype} is not among the declared dtypes {self.dtypes}."
            )
            raise ValueError(msg)
        if self.reference.options.keys() != self.options.keys():
            msg = (
                f"The reference has options {sorted(self.reference.options)} but the loader declares "
                f"{sorted(self.options)}; the reference needs exactly one value per declared option."
            )
            raise ValueError(msg)
        for name, value in self.reference.options.items():
            if not _declared(value, self.options[name]):
                msg = (
                    f"The reference value {value!r} of option {name!r} is not among its declared "
                    f"values {self.options[name]}."
                )
                raise ValueError(msg)
        return self

    def reference_settings(self) -> Settings:
        """The reference setting: what a project passes as ``lookup``'s ``default``."""
        return self.reference

    def complete(self, settings: Settings) -> Settings:
        """``settings`` with a missing dtype or option filled from the reference.

        A lookup answer from a rule carries a dtype and no options; production loads
        ``declaration.complete(answer.settings)`` so the loader always gets one value per declared option.
        """
        return Settings(
            dtype=settings.dtype or self.reference.dtype,
            options={**self.reference.options, **settings.options},
        )


class ScorerLoader(Protocol):
    """What a project writes: its declaration and its loader."""

    def declare(self) -> LoaderDeclaration:
        """The precisions and options this loader accepts, and the reference setting."""
        ...

    def load(self, model_dir: Path, settings: Settings, device: str) -> TextScorer:
        """Load the snapshot in ``model_dir`` as ``settings`` say and return its scorer."""
        ...


type LoaderFactory = Callable[[str], ScorerLoader]
"""What the entry point names: called with the project's name for the model, it returns its loader."""


def resolve_factory(spec: str) -> LoaderFactory:
    """The factory named ``"module:attribute"``, imported now (as ``registry.LazyRegistry.get`` imports).

    Raises:
        ValueError: ``spec`` is not of that form, or names a module or attribute that does not exist.
    """
    module, _, name = spec.partition(":")
    if not module or not name:
        msg = f"Loader {spec!r} must be written MODULE:ATTRIBUTE, such as my_project.tune_loaders:model."
        raise ValueError(msg)
    try:
        factory: LoaderFactory = getattr(import_module(module), name)
    except (ImportError, AttributeError) as exc:
        msg = f"Loader {spec!r} cannot be imported ({exc}); write it MODULE:ATTRIBUTE naming a factory."
        raise ValueError(msg) from exc
    return factory


def _name(dtype: Precision, options: dict[str, JsonScalar]) -> str:
    shown = (str(v).lower() if isinstance(v, bool) else str(v) for v in options.values())
    return ",".join([str(dtype), *(f"{k}={v}" for k, v in zip(options, shown, strict=True))])


def candidates(declaration: LoaderDeclaration, max_candidates: int) -> list[TuneCandidate]:
    """Every combination of the declared dtypes and option values, the reference first.

    After the reference come the combinations that change one field (dtype counts as a field), in
    declaration order, then two, and so on: when the time budget stops a tune, each field's own effect has
    been measured before any combination.

    Raises:
        TooManyCandidates: the product of the declared values is above ``max_candidates``.
    """
    fields: list[Sequence[JsonScalar]] = [declaration.dtypes, *declaration.options.values()]
    count = math.prod(len(values) for values in fields)
    if count > max_candidates:
        msg = (
            f"The loader declares {count} candidates, above the cap of {max_candidates}: "
            "declare fewer values or raise --max-candidates."
        )
        raise TooManyCandidates(msg)
    reference = [declaration.reference.dtype, *declaration.reference.options.values()]
    reference_at = [
        next(i for i, v in enumerate(values) if _same(v, ref))
        for values, ref in zip(fields, reference, strict=True)
    ]

    def order(point: tuple[int, ...]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        changed = tuple(i for i, (a, b) in enumerate(zip(point, reference_at, strict=True)) if a != b)
        return len(changed), changed, tuple(point[i] for i in changed)

    found = []
    for point in sorted(itertools.product(*(range(len(values)) for values in fields)), key=order):
        dtype = declaration.dtypes[point[0]]
        options = {
            name: values[i] for (name, values), i in zip(declaration.options.items(), point[1:], strict=True)
        }
        found.append(TuneCandidate(name=_name(dtype, options), dtype=dtype, options=options))
    return found


class _AnyOptions(BaseModel):
    """The load request's options as ``open_load`` reads them: any object; ``LoaderServer`` checks them."""

    model_config = ConfigDict(extra="allow")


class LoaderServer(ModelServer):
    """The model server of an in-process tune: loads a project's loader and scores texts with it."""

    def __init__(self, spec: str, entry: str) -> None:
        """Remember the factory spec (``module:attribute``) and the model's name to call it with."""
        self.spec = spec
        self.entry = entry
        self._scorer: TextScorer | None = None

    def load(self, msg: LoadMessage) -> LoadedMessage:
        """Load the candidate ``msg`` names, which the loader must have declared.

        Raises:
            ServingError: ``bad_input`` for an undeclared dtype, option or value, or a missing file.
        """
        import torch  # noqa: PLC0415 - only the model-server process imports PyTorch

        open_load(_AnyOptions, msg)
        loader = resolve_factory(self.spec)(self.entry)
        declaration = loader.declare()
        self._check(msg, declaration)
        started = time.perf_counter()
        try:
            self._scorer = loader.load(
                Path(msg.model_dir), Settings(dtype=msg.dtype, options=msg.options), _CUDA_DEVICE
            )
        except FileNotFoundError as exc:
            raise ServingError(
                ServingErrorCode.BAD_INPUT,
                f"{exc.filename} is missing from the snapshot; download it before tuning",
            ) from None
        return LoadedMessage(
            seconds=time.perf_counter() - started,
            torch_allocated_gib=torch.cuda.memory_allocated() / GIB,
            torch_reserved_gib=torch.cuda.memory_reserved() / GIB,
        )

    @staticmethod
    def _check(msg: LoadMessage, declaration: LoaderDeclaration) -> None:
        if msg.dtype not in declaration.dtypes:
            raise ServingError(
                ServingErrorCode.BAD_INPUT,
                f"dtype {msg.dtype} is not declared; declared: {list(declaration.dtypes)}",
            )
        for name, value in msg.options.items():
            if name not in declaration.options:
                raise ServingError(
                    ServingErrorCode.BAD_INPUT,
                    f"option {name!r} is not declared; declared: {list(declaration.options)}",
                )
            if not _declared(value, declaration.options[name]):
                raise ServingError(
                    ServingErrorCode.BAD_INPUT,
                    f"value {value!r} of option {name!r} is not declared; "
                    f"declared: {list(declaration.options[name])}",
                )

    def infer(self, msg: InferMessage, emit_progress: ProgressEmitter) -> ResultMessage:
        """Score the texts in ``<input_dir>/texts.jsonl``, in file order.

        Raises:
            ServingError: ``internal`` when the scorer does not return one finite float per text.
        """
        del emit_progress  # a scorer reports no progress
        import torch  # noqa: PLC0415 - only the model-server process imports PyTorch

        if self._scorer is None:
            raise ServingError(ServingErrorCode.BAD_INPUT, "infer before load: load the model first")
        lines = (Path(msg.input_dir) / TEXTS_FILE).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        torch.cuda.reset_peak_memory_stats()
        values = list(self._scorer([row["text"] for row in rows]))
        if len(values) != len(rows):
            raise ServingError(
                ServingErrorCode.INTERNAL, f"the scorer returned {len(values)} values for {len(rows)} texts"
            )
        scores: dict[str, float] = {}
        for row, value in zip(rows, values, strict=True):
            if not math.isfinite(float(value)):
                raise ServingError(
                    ServingErrorCode.INTERNAL, f"the scorer returned a non-finite score for {row['id']}"
                )
            scores[row["id"]] = float(value)
        return ResultMessage(
            request_id=msg.request_id,
            raw_text="",
            native_outputs={SCORES_KEY: scores},
            peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
        )


def main(argv: Sequence[str]) -> int:
    """Serve ``LoaderServer`` on stdin/stdout for the factory and entry named in ``argv``."""
    parser = argparse.ArgumentParser(prog="python -m localmodels.serving.loader")
    parser.add_argument("loader", help="MODULE:ATTRIBUTE of the loader factory")
    parser.add_argument("entry", help="the project's name for the model, passed to the factory")
    args = parser.parse_args(list(argv))
    return run(LoaderServer(args.loader, args.entry))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
