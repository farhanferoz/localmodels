"""Catalogue model selection and download for `localmodels setup`, torch-free.

Selection is pure (no network, no filesystem): the download half is `pull.pull_model`, which
already resumes interrupted pulls and pins `spec.revision` -- `download_model` exists only so
`setup.py` has one name to call for "get this spec onto disk", not a second implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .catalogue import LicencePolicy, select
from .pull import pull_model

if TYPE_CHECKING:
    from pathlib import Path

    from .catalogue import ModelCategory, ModelSpec
    from .enums import ModelSizeClass


class NoCatalogueMatchError(RuntimeError):
    """No catalogue entry satisfies the requested category, licence policy and size class."""


def choose_model(
    catalogue: list[ModelSpec],
    *,
    category: ModelCategory,
    size_class: ModelSizeClass,
    policy: LicencePolicy = LicencePolicy.OSI_ONLY,
) -> ModelSpec:
    """Choose the largest catalogue entry of ``category``/``size_class`` that ``policy`` allows.

    "Largest ... of that size class": catalogue entries are already bucketed by ``size_class``
    (the same bucket ``derive_serving_params`` recommends for the measured hardware) -- among
    the entries in that exact bucket, ``default_for_category`` breaks the tie deterministically
    (it names the one entry per category/size_class a catalogue owner has vetted as its pick),
    and alias order breaks any tie still left, so the result never depends on list order alone.
    """
    allowed = select(catalogue, policy)
    candidates = [spec for spec in allowed if spec.category is category and spec.size_class is size_class]
    if not candidates:
        msg = f"no {policy.value} catalogue entry for category={category.value} size_class={size_class.value}"
        raise NoCatalogueMatchError(msg)
    defaults = [c for c in candidates if c.default_for_category]
    pool = defaults or candidates
    return min(pool, key=lambda c: c.alias)


def download_model(spec: ModelSpec, *, models_dir: Path) -> Path:
    """Download ``spec`` (pinned to its recorded revision) into ``models_dir``."""
    return pull_model(spec, models_dir=models_dir)
