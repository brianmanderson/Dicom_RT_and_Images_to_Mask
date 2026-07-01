"""Deterministic identifier hashing for anonymized NIfTI exports.

Replicates the C# ``AnonymizationService`` byte-for-byte so the two tools
produce identical hashes for the same identifiers and salt:

* a hash is ``prefix + first N bytes of SHA256(f"{input}:{salt}") as hex``;
* patient: input ``"PATIENT:" + MRN``, prefix ``"P"``, 5 bytes (10 hex chars);
* study:   input ``"STUDY:" + StudyInstanceUID``, prefix ``"ST"``, 6 bytes;
* series:  input ``"SERIES:" + SeriesInstanceUID``, prefix ``"SE"``, 6 bytes.

The three reverse-lookup maps (hash -> original identifier) are persisted to a
JSON key file so an anonymized export can be de-anonymized later and so
re-running an export reuses the same hashes.
"""
from __future__ import annotations

import hashlib
import json
import os

from ..Services.DicomBases import PathLike

DEFAULT_SALT = "DicomToNifti"


def deterministic_hash_string(input_string: str, salt: str, prefix: str = "A", num_bytes: int = 6) -> str:
    """Return ``prefix`` + the first ``num_bytes`` of ``SHA256(input:salt)`` as hex.

    Slicing happens on the raw digest bytes *before* hex-encoding (so
    ``num_bytes=6`` yields 12 hex characters), matching the C# implementation.
    """
    salted = f"{input_string}:{salt}"
    digest = hashlib.sha256(salted.encode("utf-8")).digest()
    return prefix + digest[:num_bytes].hex()


def hash_patient(mrn: str | None, salt: str = DEFAULT_SALT) -> str:
    """Hash a patient MRN (input ``"PATIENT:<mrn>"``, prefix ``"P"``, 5 bytes)."""
    return deterministic_hash_string(f"PATIENT:{mrn or ''}", salt, "P", 5)


def hash_study(study_uid: str | None, salt: str = DEFAULT_SALT) -> str:
    """Hash a StudyInstanceUID (input ``"STUDY:<uid>"``, prefix ``"ST"``, 6 bytes)."""
    return deterministic_hash_string(f"STUDY:{study_uid or ''}", salt, "ST", 6)


def hash_series(series_uid: str | None, salt: str = DEFAULT_SALT) -> str:
    """Hash a SeriesInstanceUID (input ``"SERIES:<uid>"``, prefix ``"SE"``, 6 bytes)."""
    return deterministic_hash_string(f"SERIES:{series_uid or ''}", salt, "SE", 6)


class AnonymizationKey:
    """Reverse-lookup maps (hash -> original identifier) plus the salt.

    Mirrors the C# ``AnonymizationKeyFile`` JSON schema so key files are
    interchangeable between the two tools::

        {"Salt": "...", "Patients": {...}, "Studies": {...}, "Series": {...}}
    """

    def __init__(self, salt: str = DEFAULT_SALT) -> None:
        self.salt = salt or DEFAULT_SALT
        self.patients: dict[str, str] = {}  # patient hash -> MRN
        self.studies: dict[str, str] = {}   # study hash   -> StudyInstanceUID
        self.series: dict[str, str] = {}    # series hash  -> SeriesInstanceUID

    # -- persistence --------------------------------------------------------

    @classmethod
    def load(cls, path: PathLike) -> AnonymizationKey:
        """Load a key file, or return a fresh instance when it does not exist."""
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            obj = cls(salt=data.get("Salt", DEFAULT_SALT))
            obj.patients = dict(data.get("Patients", {}))
            obj.studies = dict(data.get("Studies", {}))
            obj.series = dict(data.get("Series", {}))
            return obj
        return cls()

    def save(self, path: PathLike) -> None:
        """Write the key file as indented JSON (creating parent dirs)."""
        directory = os.path.dirname(os.fspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "Salt": self.salt,
            "Patients": self.patients,
            "Studies": self.studies,
            "Series": self.series,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    # -- registration -------------------------------------------------------

    def register(
        self,
        mrn: str | None,
        study_uid: str | None,
        series_uid: str | None,
    ) -> tuple[str, str, str]:
        """Hash the three identifiers, record the reverse maps, return the hashes."""
        patient_hash = hash_patient(mrn, self.salt)
        study_hash = hash_study(study_uid, self.salt)
        series_hash = hash_series(series_uid, self.salt)
        self.patients.setdefault(patient_hash, mrn or "")
        self.studies.setdefault(study_hash, study_uid or "")
        self.series.setdefault(series_hash, series_uid or "")
        return patient_hash, study_hash, series_hash

    def merge(self, other: AnonymizationKey) -> None:
        """Fold another key's reverse maps into this one (in place)."""
        for mine, theirs in (
            (self.patients, other.patients),
            (self.studies, other.studies),
            (self.series, other.series),
        ):
            for key, value in theirs.items():
                mine.setdefault(key, value)
