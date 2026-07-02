"""Tests for the standalone, incremental metadata manifest (``create_manifest``)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from DicomRTTool import hash_series
from DicomRTTool.ReaderWriter import DicomReaderWriter


def _build_reader(dataset, *, require_all: bool = True) -> DicomReaderWriter:
    r = DicomReaderWriter(
        description="manifest",
        Contour_Names=[p.name for p in dataset.primitives],
        arg_max=True,
        verbose=False,
        require_all_contours=require_all,
    )
    r.walk_through_folders(str(dataset.walk_root), thread_count=1)
    return r


def _primary_entry(reader):
    idx = reader.indexes_with_contours[0]
    return reader.series_instances_dictionary[idx]


class TestCreateManifestColumns:
    def test_columns_and_no_nifti(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))

        assert out.exists()
        # No NIfTI files are produced by create_manifest.
        assert not list(tmp_path.glob("**/*.nii.gz"))

        df = pd.read_csv(out)
        for col in ["patient_hash", "study_hash", "series_hash",
                    "spacing_x", "spacing_y", "spacing_z"]:
            assert col in df.columns
        # The raw-identifier columns are gone.
        for col in ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]:
            assert col not in df.columns
        for p in synthetic_dataset.primitives:
            col = f"{p.name.lower()} cc"
            assert col in df.columns
            assert (df[col] > 0).all()

    def test_not_anonymized_hash_columns_hold_raw_ids(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        entry = _primary_entry(r)
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))
        df = pd.read_csv(out)

        assert str(entry.SeriesInstanceUID) in df["series_hash"].astype(str).tolist()
        assert str(entry.StudyInstanceUID) in df["study_hash"].astype(str).tolist()
        assert str(entry.PatientID) in df["patient_hash"].astype(str).tolist()

    def test_anonymized_hash_columns_hold_hashes(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        entry = _primary_entry(r)
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out), anonymize=True, salt="unit-salt")
        df = pd.read_csv(out)

        assert "PatientID" not in df.columns
        expected = hash_series(entry.SeriesInstanceUID, salt="unit-salt")
        assert expected in df["series_hash"].astype(str).tolist()
        # Raw UID must NOT appear when anonymized.
        assert str(entry.SeriesInstanceUID) not in df["series_hash"].astype(str).tolist()

    def test_spacing_matches_image(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        r.set_index(r.indexes_with_contours[0])
        r.get_images()
        sx, sy, sz = r.dicom_handle.GetSpacing()
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))
        df = pd.read_csv(out)
        assert df["spacing_x"].iloc[0] == pytest.approx(sx)
        assert df["spacing_y"].iloc[0] == pytest.approx(sy)
        assert df["spacing_z"].iloc[0] == pytest.approx(sz)

    def test_defaults_to_all_rois_when_contour_names_unset(self, synthetic_dataset, tmp_path: Path):
        r = DicomReaderWriter(description="manifest", verbose=False, require_all_contours=False)
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        assert r.all_rois and not r.Contour_Names

        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))
        df = pd.read_csv(out)
        roi_cols = [c for c in df.columns if c.endswith(" cc")]
        assert len(roi_cols) == len(r.all_rois)
        assert (df[roi_cols] > 0).any().any()

    def test_absent_roi_is_blank(self, synthetic_dataset, tmp_path: Path):
        r = DicomReaderWriter(
            description="manifest",
            Contour_Names=[p.name for p in synthetic_dataset.primitives] + ["not_a_real_roi"],
            arg_max=True,
            verbose=False,
            require_all_contours=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))
        df = pd.read_csv(out)
        # An absent ROI leaves an empty cell (NaN when read back), not -1.
        assert df["not_a_real_roi cc"].isna().all()


class TestCreateManifestKeyFile:
    def test_key_file_written_next_to_manifest(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        entry = _primary_entry(r)
        sub = tmp_path / "nested"
        out = sub / "manifest.csv"
        r.create_manifest(str(out), anonymize=True, salt="unit-salt")

        key_path = sub / "anonymization_key.json"
        assert key_path.exists(), "key file should sit beside the manifest"
        data = json.loads(key_path.read_text())
        assert set(data) == {"Salt", "Patients", "Studies", "Series"}
        assert data["Salt"] == "unit-salt"
        # The series hash maps back to the original UID.
        series_hash = hash_series(entry.SeriesInstanceUID, salt="unit-salt")
        assert data["Series"][series_hash] == entry.SeriesInstanceUID

    def test_constructor_anonymize_default(self, synthetic_dataset, tmp_path: Path):
        # anonymize set at construction is used when not passed per-call.
        r = DicomReaderWriter(
            description="manifest",
            Contour_Names=[p.name for p in synthetic_dataset.primitives],
            verbose=False,
            anonymize=True, salt="ctor-salt",
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        entry = _primary_entry(r)
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))   # no anonymize= -> uses ctor default

        df = pd.read_csv(out)
        assert hash_series(entry.SeriesInstanceUID, salt="ctor-salt") in df["series_hash"].astype(str).tolist()
        assert str(entry.SeriesInstanceUID) not in df["series_hash"].astype(str).tolist()

    def test_existing_key_salt_is_reused(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        entry = _primary_entry(r)
        out = tmp_path / "manifest.csv"

        # Pre-seed a key file with a specific salt next to the manifest.
        key_path = tmp_path / "anonymization_key.json"
        key_path.write_text(json.dumps(
            {"Salt": "preset-salt", "Patients": {}, "Studies": {}, "Series": {}}
        ))

        # Even though we pass a different salt, the existing key's salt wins.
        r.create_manifest(str(out), anonymize=True, salt="ignored-salt")
        df = pd.read_csv(out)
        assert hash_series(entry.SeriesInstanceUID, salt="preset-salt") in df["series_hash"].astype(str).tolist()
        assert json.loads(key_path.read_text())["Salt"] == "preset-salt"


class TestCreateManifestUpsert:
    def test_rerun_updates_not_duplicates(self, synthetic_multi_series_dataset, tmp_path: Path):
        names = [p.name for p in synthetic_multi_series_dataset.pairs[0].primitives]
        r = DicomReaderWriter(description="manifest", Contour_Names=names, verbose=False)
        r.walk_through_folders(str(synthetic_multi_series_dataset.walk_root), thread_count=1)

        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))
        n_first = len(pd.read_csv(out))
        assert n_first >= 2

        r.create_manifest(str(out))
        df = pd.read_csv(out)
        assert len(df) == n_first
        assert df["series_hash"].is_unique

    def test_toggling_anonymize_updates_not_duplicates(self, synthetic_dataset, tmp_path: Path):
        # Same manifest, anonymize True then False: the series must be updated
        # in place (matched on the canonical hash), not duplicated.
        r = _build_reader(synthetic_dataset)
        entry = _primary_entry(r)
        out = tmp_path / "manifest.csv"

        r.create_manifest(str(out), anonymize=True, salt="toggle-salt")
        n1 = len(pd.read_csv(out))

        r.create_manifest(str(out), anonymize=False, salt="toggle-salt")
        df = pd.read_csv(out)
        assert len(df) == n1, "toggling anonymize must not duplicate the series"
        # Row now shows the raw identifier (anonymize=False).
        assert str(entry.SeriesInstanceUID) in df["series_hash"].astype(str).tolist()

        # And back to anonymized: still one row.
        r.create_manifest(str(out), anonymize=True, salt="toggle-salt")
        df = pd.read_csv(out)
        assert len(df) == n1
        assert hash_series(entry.SeriesInstanceUID, "toggle-salt") in df["series_hash"].astype(str).tolist()

    def test_existing_row_is_refreshed(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))

        roi_col = f"{synthetic_dataset.primitives[0].name.lower()} cc"
        df = pd.read_csv(out)
        true_vol = df[roi_col].iloc[0]

        # Corrupt a value, then re-run: the row must be recomputed (upsert update).
        df[roi_col] = -999.0
        df.to_csv(out, index=False)
        r.create_manifest(str(out))

        refreshed = pd.read_csv(out)
        assert refreshed[roi_col].iloc[0] == pytest.approx(true_vol)
        assert (refreshed[roi_col] != -999.0).all()

    def test_append_new_series_and_backfill_columns(self, synthetic_dataset, tmp_path: Path):
        out = tmp_path / "manifest.csv"
        roi_name = synthetic_dataset.primitives[0].name.lower()

        # Seed a row for a different series (new-format columns) + a legacy ROI.
        seed = pd.DataFrame([{
            "patient_hash": "SEED_MRN", "study_hash": "seed.study", "series_hash": "seed.series",
            "spacing_x": 1.0, "spacing_y": 1.0, "spacing_z": 1.0,
            "legacy_roi cc": 12.5,
        }])
        seed.to_csv(out, index=False)

        r = _build_reader(synthetic_dataset)
        r.create_manifest(str(out))
        df = pd.read_csv(out)

        assert "seed.series" in df["series_hash"].astype(str).tolist()
        assert len(df) >= 2
        # New ROI columns added; the seed row is left blank for them.
        assert f"{roi_name} cc" in df.columns
        seed_row = df[df["series_hash"] == "seed.series"]
        assert seed_row[f"{roi_name} cc"].isna().all()
        # Legacy column preserved; new rows are blank for it.
        assert "legacy_roi cc" in df.columns
        new_rows = df[df["series_hash"] != "seed.series"]
        assert new_rows["legacy_roi cc"].isna().all()


class TestManifestInterop:
    """Round-trips between the two manifest writers and dtype safety."""

    def test_all_digit_mrn_keeps_leading_zeros(self, synthetic_dataset, tmp_path: Path):
        """Regression: an all-digit ``patient_hash`` column parsed as int64 on
        reload, so ``"00123"`` was rewritten as ``123`` by the next
        incremental run."""
        out = tmp_path / "manifest.csv"
        seed = pd.DataFrame([{
            "patient_hash": "00123", "study_hash": "seed.study",
            "series_hash": "seed.series",
            "spacing_x": 1.0, "spacing_y": 1.0, "spacing_z": 1.0,
        }])
        seed.to_csv(out, index=False)

        r = _build_reader(synthetic_dataset)
        r.create_manifest(str(out), anonymize=False)

        df = pd.read_csv(out, dtype=str)
        assert "00123" in df["patient_hash"].tolist()

    def test_updating_a_write_to_folder_manifest_keeps_case_id(
        self, synthetic_dataset, tmp_path: Path,
    ):
        """``case_id`` is produced by write_to_folder only; a create_manifest
        update of the same file must carry it onto the replaced row, not
        blank it."""
        out = tmp_path / "out"
        r = _build_reader(synthetic_dataset)
        r.write_to_folder(str(out), anonymize=True, salt="sticky")
        manifest = out / "manifest.csv"
        before = pd.read_csv(manifest, dtype=str)
        series_hash = before["series_hash"].iloc[0]
        case_id = before["case_id"].iloc[0]

        r.create_manifest(str(manifest), anonymize=True, salt="sticky")

        after = pd.read_csv(manifest, dtype=str)
        assert len(after) == len(before)
        row = after[after["series_hash"] == series_hash]
        assert row["case_id"].iloc[0] == case_id
