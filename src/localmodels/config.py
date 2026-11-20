"""Settings a run might change, read from the tracked files in ``data/`` (coding standard rule 3).

``data/job_settings.toml`` holds the slurm asks, hard stops and measurement sizes of the jobs this package
submits or runs, each with its source; ``data/digest_sources.toml`` holds what the monthly digest watches.
Every timer and job runs this checkout, so a value changes by a reviewed commit to the file. Values that are
ratified decisions, or that calibrate one, stay in code beside their decision.

Callers read ``config.job_settings()`` where the value is used (not ``from .config import ...`` at import),
so a test can replace the settings with ``monkeypatch.setattr(config, "job_settings", ...)``.
"""

from __future__ import annotations

import functools
import re  # noqa: TC003 - pydantic needs it at runtime
import tomllib
from importlib import resources
from typing import Annotated, Any, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    ValidationError,
)

JOB_SETTINGS_FILE: Final = "data/job_settings.toml"
DIGEST_SOURCES_FILE: Final = "data/digest_sources.toml"

#: slurm's ``--mem`` form: a whole number and a unit
SlurmMem = Annotated[str, StringConstraints(pattern=r"^[1-9]\d*[KMGT]$")]
#: slurm's ``--time`` form used here: hours, minutes and seconds
SlurmTime = Annotated[str, StringConstraints(pattern=r"^\d+:[0-5]\d:[0-5]\d$")]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RefreshJobs(_Section):
    """``refresh run``'s llama.cpp build and measure jobs."""

    build_cpus: PositiveInt
    build_mem: SlurmMem
    measure_mem: SlurmMem
    measure_limit_s: PositiveInt
    kill_after_s: PositiveInt
    urgent_time: SlurmTime


class EngineRefreshJobs(_Section):
    """The container engines' pull, measure, calibrate, same-client and sparkrun jobs."""

    measure_limit_s: PositiveInt
    calibrate_limit_s: PositiveInt
    same_client_limit_s: PositiveInt
    sparkrun_limit_s: PositiveInt
    job_mem_margin_gib: NonNegativeInt
    kill_after_s: PositiveInt
    pull_mem: SlurmMem
    pull_cpus: PositiveInt
    pull_timeout_s: PositiveFloat
    min_free_disk_gb: PositiveInt


class EngineBench(_Section):
    """llama-benchy, the cross-check client."""

    timeout_s: PositiveFloat
    prompt_tokens: PositiveInt
    runs: PositiveInt


class LlamaTune(_Section):
    """The llama.cpp tune's measurement sizes."""

    bench_repetitions: PositiveInt


class Retune(_Section):
    """The tune jobs ``settings.lookup`` queues by itself."""

    max_live_tunes: PositiveInt
    kill_after_s: PositiveInt


class JobSettings(_Section):
    """Every section of ``data/job_settings.toml``."""

    refresh: RefreshJobs
    engine_refresh: EngineRefreshJobs
    engine_bench: EngineBench
    llama_tune: LlamaTune
    retune: Retune


class DigestSources(_Section):
    """What the monthly digest watches."""

    max_lines_per_source: PositiveInt
    llama_paths: tuple[str, ...]
    llama_title_words: re.Pattern[str]
    llama_discussions: tuple[PositiveInt, ...]
    library_repos: tuple[str, ...]
    gb10_words: re.Pattern[str]
    engine_repos: tuple[str, ...]
    forum_words: re.Pattern[str]
    playbook_dirs: tuple[str, ...]
    arena_tools: tuple[str, ...]
    hf_orgs: tuple[str, ...]
    hf_families: re.Pattern[str]


def _read[T: BaseModel](name: str, model: type[T]) -> T:
    """The packaged file ``name`` as ``model``.

    Raises:
        ValueError: the file cannot be read, is not valid TOML or does not match ``model``, naming the file
            (where this installation holds it) and the field.
    """
    resource = resources.files("localmodels").joinpath(name)
    try:
        data: dict[str, Any] = tomllib.loads(resource.read_text(encoding="utf-8"))
        return model.model_validate(data)
    except OSError as exc:
        msg = f"cannot read {resource}: {exc}; reinstall localmodels, or restore the file from git"
        raise ValueError(msg) from exc
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        msg = f"{resource} is not valid: {exc}; fix that file"
        raise ValueError(msg) from exc


@functools.cache
def job_settings() -> JobSettings:
    """The tracked job settings (``data/job_settings.toml``), read once per process."""
    return _read(JOB_SETTINGS_FILE, JobSettings)


@functools.cache
def digest_sources() -> DigestSources:
    """What the digest watches (``data/digest_sources.toml``), read once per process."""
    return _read(DIGEST_SOURCES_FILE, DigestSources)
