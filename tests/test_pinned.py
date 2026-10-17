"""Pinned snapshots: what a pin records, and the check that every file is the one recorded.

Moved here from service_b with the code (its ``tests/unit/test_catalogue.py`` hash and validation cases).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from localmodels.pinned import PinnedFile, PinnedModel, hash_snapshot, sha256_file, snapshot_dir

if TYPE_CHECKING:
    from pathlib import Path

REVISION = "a" * 40
DIGEST = "0" * 64


@pytest.mark.parametrize(
    "fields",
    [
        {"path": "/abs/w.bin", "sha256": DIGEST},
        {"path": "../w.bin", "sha256": DIGEST},
        {"path": "w.bin", "sha256": "ABC"},
    ],
    ids=["absolute", "escapes", "not-sha256"],
)
def test_a_pinned_file_stays_inside_the_snapshot_and_has_a_sha256(fields: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        PinnedFile(**fields)


@pytest.mark.parametrize("revision", ["main", "a" * 39], ids=["branch", "short"])
def test_a_revision_is_a_full_commit_sha(revision: str) -> None:
    with pytest.raises(ValidationError):
        PinnedModel(repo_id="org/m", revision=revision)


def test_a_pin_records_every_file_and_the_check_catches_a_changed_or_missing_one(tmp_path: Path) -> None:
    snapshot = snapshot_dir("org/m", REVISION, tmp_path)
    (snapshot / "sub").mkdir(parents=True)
    (snapshot / "w.bin").write_bytes(b"weights")
    (snapshot / "sub" / "c.json").write_text("{}")
    model = PinnedModel(repo_id="org/m", revision=REVISION, files=tuple(hash_snapshot(snapshot)))

    assert [f.path for f in model.files] == ["sub/c.json", "w.bin"]
    assert (model.is_pinned, model.present(tmp_path), model.mismatching(tmp_path)) == (True, True, [])
    assert not PinnedModel(repo_id="org/m", revision=REVISION).is_pinned
    (snapshot / "w.bin").write_bytes(b"changed")
    assert model.mismatching(tmp_path) == ["w.bin"]
    assert sha256_file(snapshot / "w.bin") == hashlib.sha256(b"changed").hexdigest()
    (snapshot / "sub" / "c.json").unlink()
    assert not model.present(tmp_path)
    assert model.mismatching(tmp_path) == ["sub/c.json", "w.bin"]
