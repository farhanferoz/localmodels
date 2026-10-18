"""Unit tests for catalogue model selection ("largest OSI-licensed model of a size class").

`choose_model` is pure (no network, no filesystem) -- the download half is `pull.pull_model`,
covered by `tests/test_pull.py`.
"""

from __future__ import annotations

import pytest

from localmodels.catalogue import LicencePolicy, ModelCategory, ModelSpec
from localmodels.download import NoCatalogueMatchError, choose_model
from localmodels.enums import ModelSizeClass

_LARGE = ModelSpec(
    alias="qwen3-30b-a3b",
    category=ModelCategory.GENERATOR,
    repo_id="Qwen/Qwen3-30B-A3B-GGUF",
    filename="Qwen3-30B-A3B-Q4_K_M.gguf",
    revision="e4d4bafdfb96a411a163846265362aceb0b9c63a",
    description="workhorse MoE generator",
    licence="apache-2.0",
    osi_approved=True,
    size_class=ModelSizeClass.LARGE,
    default_for_category=True,
)
_SMALL_DEFAULT = ModelSpec(
    alias="phi-4-mini",
    category=ModelCategory.GENERATOR,
    repo_id="unsloth/Phi-4-mini-instruct-GGUF",
    filename="Phi-4-mini-instruct-Q4_K_M.gguf",
    revision="78eb92a46fc37e6b524df991ed9aca9bc6aa7b80",
    description="small generator",
    licence="mit",
    osi_approved=True,
    size_class=ModelSizeClass.SMALL,
    default_for_category=True,
)
_SMALL_NON_DEFAULT = ModelSpec(
    alias="qwen3-4b",
    category=ModelCategory.GENERATOR,
    repo_id="Qwen/Qwen3-4B-GGUF",
    filename="Qwen3-4B-Q4_K_M.gguf",
    revision="bc640142c66e1fdd12af0bd68f40445458f3869b",
    description="small generator, alternate",
    licence="apache-2.0",
    osi_approved=True,
    size_class=ModelSizeClass.SMALL,
    default_for_category=False,
)
_SMALL_NON_OSI = ModelSpec(
    alias="gemma-small",
    category=ModelCategory.GENERATOR,
    repo_id="google/gemma-2b",
    description="small generator, non-OSI licence",
    licence="gemma-terms-of-use",
    osi_approved=False,
    size_class=ModelSizeClass.SMALL,
    default_for_category=True,
)
_SMALL_EMBED = ModelSpec(
    alias="bge-small",
    category=ModelCategory.EMBED,
    repo_id="BAAI/bge-small-en-v1.5",
    description="small embedding model",
    licence="mit",
    osi_approved=True,
    size_class=ModelSizeClass.SMALL,
    default_for_category=True,
)

_CATALOGUE = [_LARGE, _SMALL_DEFAULT, _SMALL_NON_DEFAULT, _SMALL_NON_OSI, _SMALL_EMBED]


def test_chooses_the_entry_matching_category_and_size_class() -> None:
    chosen = choose_model(_CATALOGUE, category=ModelCategory.GENERATOR, size_class=ModelSizeClass.LARGE)
    assert chosen is _LARGE


def test_never_chooses_a_non_osi_entry() -> None:
    catalogue = [_SMALL_NON_OSI]
    with pytest.raises(NoCatalogueMatchError):
        choose_model(catalogue, category=ModelCategory.GENERATOR, size_class=ModelSizeClass.SMALL)


def test_default_for_category_breaks_a_tie() -> None:
    chosen = choose_model(_CATALOGUE, category=ModelCategory.GENERATOR, size_class=ModelSizeClass.SMALL)
    assert chosen is _SMALL_DEFAULT


def test_any_policy_may_still_choose_a_non_osi_entry() -> None:
    catalogue = [_SMALL_NON_OSI]
    chosen = choose_model(
        catalogue, category=ModelCategory.GENERATOR, size_class=ModelSizeClass.SMALL, policy=LicencePolicy.ANY
    )
    assert chosen is _SMALL_NON_OSI


def test_raises_when_no_entry_matches_the_size_class() -> None:
    with pytest.raises(NoCatalogueMatchError, match="size_class=large"):
        choose_model(_CATALOGUE, category=ModelCategory.EMBED, size_class=ModelSizeClass.LARGE)
