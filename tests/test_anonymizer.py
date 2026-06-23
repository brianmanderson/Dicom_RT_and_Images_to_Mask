"""Tests for the deterministic anonymization hashing + key file.

The hashing must match the C# ``AnonymizationService`` byte-for-byte:
``prefix + first N bytes of SHA256(f"{input}:{salt}") as lowercase hex``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from DicomRTTool import (
    AnonymizationKey,
    deterministic_hash_string,
    hash_patient,
    hash_series,
    hash_study,
)
from DicomRTTool._internal.anonymizer import DEFAULT_SALT


def _expected(input_str: str, salt: str, prefix: str, num_bytes: int) -> str:
    """Independent re-implementation of the C# formula for cross-checking."""
    digest = hashlib.sha256(f"{input_str}:{salt}".encode()).digest()
    return prefix + digest[:num_bytes].hex()


class TestDeterministicHash:
    def test_matches_independent_formula(self):
        assert deterministic_hash_string("PATIENT:12345", DEFAULT_SALT, "P", 5) == _expected(
            "PATIENT:12345", DEFAULT_SALT, "P", 5
        )

    def test_slices_raw_bytes_not_hexdigest(self):
        # 6 bytes -> 12 hex chars; this fails if the impl used hexdigest()[:6].
        h = deterministic_hash_string("STUDY:abc", DEFAULT_SALT, "ST", 6)
        assert len(h) == len("ST") + 12

    def test_patient_study_series_lengths_and_prefixes(self):
        p = hash_patient("12345")
        st = hash_study("1.2.3")
        se = hash_series("1.2.3.4")
        assert p.startswith("P") and len(p) == 1 + 10   # 5 bytes
        assert st.startswith("ST") and len(st) == 2 + 12  # 6 bytes
        assert se.startswith("SE") and len(se) == 2 + 12  # 6 bytes

    def test_regression_anchor_vectors(self):
        # Frozen vectors: guard against accidental algorithm drift.
        assert hash_patient("12345") == "P5e91ac4196"
        assert hash_study("1.2.3") == "ST9b7493c25553"
        assert hash_series("1.2.3.4") == "SE5f79c3efc041"

    def test_salt_changes_hash(self):
        assert hash_patient("12345", salt="A") != hash_patient("12345", salt="B")

    def test_namespacing_prevents_collisions(self):
        # Same raw value, different identifier type -> different hash.
        assert hash_patient("X", salt="s") != hash_study("X", salt="s")

    def test_none_identifier_is_stable(self):
        assert hash_patient(None) == hash_patient("")


class TestAnonymizationKey:
    def test_register_returns_hashes_and_records_reverse_maps(self):
        key = AnonymizationKey()
        p, st, se = key.register("MRN1", "study-uid", "series-uid")
        assert key.patients[p] == "MRN1"
        assert key.studies[st] == "study-uid"
        assert key.series[se] == "series-uid"

    def test_register_is_idempotent(self):
        key = AnonymizationKey()
        first = key.register("MRN1", "s", "se")
        second = key.register("MRN1", "s", "se")
        assert first == second
        assert len(key.patients) == 1

    def test_save_and_load_round_trip(self, tmp_path: Path):
        key = AnonymizationKey(salt="my-salt")
        key.register("MRN1", "study-uid", "series-uid")
        path = tmp_path / "anon.json"
        key.save(path)

        # Schema matches the C# AnonymizationKeyFile.
        data = json.loads(path.read_text())
        assert set(data) == {"Salt", "Patients", "Studies", "Series"}
        assert data["Salt"] == "my-salt"

        reloaded = AnonymizationKey.load(path)
        assert reloaded.salt == "my-salt"
        assert reloaded.patients == key.patients
        assert reloaded.studies == key.studies
        assert reloaded.series == key.series

    def test_load_missing_returns_fresh(self, tmp_path: Path):
        key = AnonymizationKey.load(tmp_path / "does_not_exist.json")
        assert key.patients == {} and key.studies == {} and key.series == {}

    def test_merge_folds_in_other_maps(self):
        a = AnonymizationKey()
        a.register("MRN1", "s1", "se1")
        b = AnonymizationKey()
        b.register("MRN2", "s2", "se2")
        a.merge(b)
        assert "MRN1" in a.patients.values()
        assert "MRN2" in a.patients.values()
