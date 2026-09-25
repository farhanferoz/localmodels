from localmodels.catalogue import LicencePolicy, ModelCategory, ModelSpec, licence_allowed, select
from localmodels.enums import ModelSizeClass

_MIT_ENTRY = ModelSpec(
    alias="qwen3-moe",
    category=ModelCategory.GENERATOR,
    repo_id="Qwen/Qwen3-30B-A3B-GGUF",
    filename="Qwen3-30B-A3B-Q4_K_M.gguf",
    description="workhorse generator",
    licence="apache-2.0",
    osi_approved=True,
    size_class=ModelSizeClass.LARGE,
)
_NON_OSI_ENTRY = ModelSpec(
    alias="gemma-license",
    category=ModelCategory.GENERATOR,
    repo_id="google/gemma-2b",
    description="non-OSI custom licence",
    licence="gemma-terms-of-use",
    osi_approved=False,
    size_class=ModelSizeClass.SMALL,
)


def test_licence_policy_osi_only_never_selects_a_non_osi_entry() -> None:
    catalogue = [_MIT_ENTRY, _NON_OSI_ENTRY]
    selected = select(catalogue, LicencePolicy.OSI_ONLY)
    assert selected == [_MIT_ENTRY]
    assert all(s.osi_approved for s in selected)


def test_licence_policy_any_selects_everything() -> None:
    catalogue = [_MIT_ENTRY, _NON_OSI_ENTRY]
    assert select(catalogue, LicencePolicy.ANY) == catalogue


def test_licence_allowed_matches_select() -> None:
    assert licence_allowed(LicencePolicy.OSI_ONLY, _MIT_ENTRY) is True
    assert licence_allowed(LicencePolicy.OSI_ONLY, _NON_OSI_ENTRY) is False
    assert licence_allowed(LicencePolicy.ANY, _NON_OSI_ENTRY) is True
