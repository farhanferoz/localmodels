"""Closed-set variants for local model detection, tiering and serving — torch-free.

Every value here is either measured hardware terminology or a serving-policy verdict; nothing
in this package compares a string literal against one of these sets directly (code standard
rule: closed sets get an Enum, never scattered magic-string comparisons).
"""

from __future__ import annotations

from enum import StrEnum


class DeviceType(StrEnum):
    """Compute device family; measured, never assumed from a card name."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class DeviceTier(StrEnum):
    """Serving tier derived from measured device and memory."""

    FAST_ACCELERATED = "fast_accelerated"
    MODERATE_ACCELERATED = "moderate_accelerated"
    SLOW_CPU = "slow_cpu"


class QuantLevel(StrEnum):
    """llama.cpp quantisation names, verified against its own ``LLAMA_FTYPE_MOSTLY_*`` enum."""

    IQ2_XXS = "IQ2_XXS"
    Q3_K_M = "Q3_K_M"
    Q4_K_M = "Q4_K_M"


class ModelSizeClass(StrEnum):
    """Recommended model size bucket for the measured hardware, not a specific model."""

    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"
    REMOTE_RECOMMENDED = "remote_recommended"


class Precision(StrEnum):
    """Numeric precision a model runs in; a member is added once measured on the hardware that runs it."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class PreflightVerdict(StrEnum):
    """Answer to "can this machine run local models at all?" — never a bare pass/fail.

    Distinguishes *cannot run local models* (``LOCAL_MODELS_INFEASIBLE``) from *will run
    slowly* (``DEGRADED``), so a caller can give an actionable fallback rather than a refusal.
    """

    READY = "ready"
    DEGRADED = "degraded"
    LOCAL_MODELS_INFEASIBLE = "local_models_infeasible"


class WorkloadKind(StrEnum):
    """The kinds of model work the measured rules are organised by (``localmodels rules show``)."""

    K1 = "K1"  # interactive text generation
    K2 = "K2"  # bulk text generation
    K3 = "K3"  # text scoring (per-token log-probabilities)
    K4 = "K4"  # embeddings and reranking
    K5 = "K5"  # vision-language
    K6 = "K6"  # image, 3D and text encoders, classifiers
    K7 = "K7"  # speech
    K8 = "K8"  # training, small models from scratch
    K9 = "K9"  # training, larger models and heads on frozen encoders
    K10 = "K10"  # fine-tuning
