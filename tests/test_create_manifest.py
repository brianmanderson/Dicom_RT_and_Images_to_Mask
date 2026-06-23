"""Tests for the standalone, incremental metadata manifest (``create_manifest``)."""
from __future__ import annotations

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


class TestCreateManifestBasics:
    def test_writes_expected_columns_and_no_nifti(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))

        assert out.exists()
        # No NIfTI files are produced by create_manifest.
        assert not list(tmp_path.glob("**/*.nii.gz"))

        df = pd.read_csv(out)
        for col in ["patient_hash", "study_hash", "series_hash",
                    "PatientID", "StudyInstanceUID", "SeriesInstanceUID",
                    "spacing_x", "spacing_y", "spacing_z"]:
            assert col in df.columns
        # ROI volume columns present and positive for the synthetic primitives.
        for p in synthetic_dataset.primitives:
            col = f"{p.name.lower()} cc"
            assert col in df.columns
            assert (df[col] > 0).all()

    def test_spacing_matches_image(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        idx = r.indexes_with_contours[0]
        r.set_index(idx)
        r.get_images()
        sx, sy, sz = r.dicom_handle.GetSpacing()

        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))
        df = pd.read_csv(out)
        assert df["spacing_x"].iloc[0] == pytest.approx(sx)
        assert df["spacing_y"].iloc[0] == pytest.approx(sy)
        assert df["spacing_z"].iloc[0] == pytest.approx(sz)

    def test_anonymize_omits_raw_identifiers(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out), anonymize=True, salt="unit-salt")
        df = pd.read_csv(out)
        assert "series_hash" in df.columns
        assert "PatientID" not in df.columns
        idx = r.indexes_with_contours[0]
        series_uid = r.series_instances_dictionary[idx].SeriesInstanceUID
        assert hash_series(series_uid, salt="unit-salt") in df["series_hash"].astype(str).tolist()

    def test_absent_roi_is_negative_one(self, synthetic_dataset, tmp_path: Path):
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
        assert (df["not_a_real_roi cc"] == -1).all()


class TestCreateManifestIncremental:
    def test_existing_file_is_extended_not_duplicated(
        self, synthetic_multi_series_dataset, tmp_path: Path,
    ):
        # Two independent series under one walk root.
        names = [p.name for p in synthetic_multi_series_dataset.pairs[0].primitives]
        r = DicomReaderWriter(
            description="manifest",
            Contour_Names=names,
            arg_max=True,
            verbose=False,
        )
        r.walk_through_folders(str(synthetic_multi_series_dataset.walk_root), thread_count=1)
        assert len(r.indexes_with_contours) >= 2

        out = tmp_path / "manifest.csv"
        r.create_manifest(str(out))
        df_first = pd.read_csv(out)
        n_first = len(df_first)
        assert n_first >= 2

        # Re-running on the same data must NOT duplicate any rows.
        r.create_manifest(str(out))
        df_again = pd.read_csv(out)
        assert len(df_again) == n_first
        assert df_again["SeriesInstanceUID"].is_unique

    def test_append_adds_new_series_and_backfills_columns(self, synthetic_dataset, tmp_path: Path):
        out = tmp_path / "manifest.csv"

        # Seed the manifest with a pre-existing row for a different series + a
        # ROI column that the reader does not know about.
        roi_name = synthetic_dataset.primitives[0].name.lower()
        seed = pd.DataFrame([{
            "patient_hash": "Pseed", "study_hash": "STseed", "series_hash": "SEseed",
            "PatientID": "SEED", "StudyInstanceUID": "seed.study",
            "SeriesInstanceUID": "seed.series",
            "spacing_x": 1.0, "spacing_y": 1.0, "spacing_z": 1.0,
            "legacy_roi cc": 12.5,
        }])
        seed.to_csv(out, index=False)

        r = _build_reader(synthetic_dataset)
        r.create_manifest(str(out))
        df = pd.read_csv(out)

        # Seed row preserved, plus the newly walked series appended.
        assert "seed.series" in df["SeriesInstanceUID"].astype(str).tolist()
        assert len(df) >= 2
        # New ROI columns added; the seed row gets -1 for them.
        assert f"{roi_name} cc" in df.columns
        seed_row = df[df["SeriesInstanceUID"] == "seed.series"]
        assert (seed_row[f"{roi_name} cc"] == -1).all()
        # The legacy column is preserved; new rows are backfilled to -1.
        assert "legacy_roi cc" in df.columns
        new_rows = df[df["SeriesInstanceUID"] != "seed.series"]
        assert (new_rows["legacy_roi cc"] == -1).all()
