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


def test_the_licence_policy_decides_which_entries_are_selected() -> None:
    catalogue = [_MIT_ENTRY, _NON_OSI_ENTRY]
    assert select(catalogue, LicencePolicy.OSI_ONLY) == [_MIT_ENTRY]
    assert select(catalogue, LicencePolicy.ANY) == catalogue
    assert licence_allowed(LicencePolicy.OSI_ONLY, _MIT_ENTRY) is True
    assert licence_allowed(LicencePolicy.OSI_ONLY, _NON_OSI_ENTRY) is False
    assert licence_allowed(LicencePolicy.ANY, _NON_OSI_ENTRY) is True
