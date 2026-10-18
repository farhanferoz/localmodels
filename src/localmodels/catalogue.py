"""Model catalogue shape and licence-gated selection, torch-free.

Catalogue *entries* are each caller's own data — service_a's and service_c's model lists stay
in their own repositories, each filtered by its own ``LicencePolicy`` (service_a requires
``OSI_ONLY``, D1; service_c keeps its own rules). This module defines only the shared shape.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from localmodels.enums import ModelSizeClass  # noqa: TC001 - pydantic resolves field annotations at runtime


class ModelCategory(StrEnum):
    """What a catalogue entry is for."""

    GENERATOR = "generator"
    NLI = "nli"
    EMBED = "embed"
    DETECTOR = "detector"


class LicencePolicy(StrEnum):
    """Which catalogue entries a caller's pull/search may select."""

    OSI_ONLY = "osi_only"
    ANY = "any"


class ModelSpec(BaseModel):
    """One catalogue entry: a Hugging Face Hub repository, and what it is for."""

    alias: str
    category: ModelCategory
    repo_id: str
    filename: str | None = None
    revision: str | None = None  # a pinned Hub revision; None = the default branch
    sha256: str | None = None  # the SHA-256 of ``filename``, checked by ``resolve_model_file``
    description: str
    licence: str
    osi_approved: bool
    size_class: ModelSizeClass
    tier_recommendation: str = "high"
    default_for_category: bool = False
    local: bool = False


def licence_allowed(policy: LicencePolicy, spec: ModelSpec) -> bool:
    """Whether ``spec`` may be selected under ``policy``."""
    if policy is LicencePolicy.OSI_ONLY:
        return spec.osi_approved
    return True


def select(specs: list[ModelSpec], policy: LicencePolicy) -> list[ModelSpec]:
    """Filter a caller's catalogue down to the entries its ``LicencePolicy`` allows."""
    return [s for s in specs if licence_allowed(policy, s)]
