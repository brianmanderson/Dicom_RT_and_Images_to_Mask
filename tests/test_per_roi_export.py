"""Tests for the C#-compatible per-ROI NIfTI export (``write_per_roi``)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import SimpleITK as sitk

from DicomRTTool import hash_series
from DicomRTTool.ReaderWriter import DicomReaderWriter


def _build_reader(dataset, *, with_dose: bool = False) -> DicomReaderWriter:
    r = DicomReaderWriter(
        description="per-roi",
        Contour_Names=[p.name for p in dataset.primitives],
        arg_max=True,
        verbose=False,
        get_dose_output=with_dose,
    )
    r.walk_through_folders(str(dataset.walk_root), thread_count=1)
    return r


class TestWritePerRoiLayout:
    @pytest.mark.parametrize("thread_count", [1, 2])
    def test_writes_image_masks_and_manifest(
        self, synthetic_dataset, tmp_path: Path, thread_count: int,
    ):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / f"out_t{thread_count}"
        r.write_per_roi(str(out), thread_count=thread_count)

        manifest = out / "manifest.csv"
        assert manifest.exists()

        case_dirs = [p for p in out.iterdir() if p.is_dir()]
        assert case_dirs, "no case folder produced"
        case = case_dirs[0]
        assert (case / "image.nii.gz").exists()
        masks = list((case / "masks").glob("*.nii.gz"))
        roi_names = {p.name.lower() for p in synthetic_dataset.primitives}
        assert {m.stem.replace(".nii", "").lower() for m in masks} == roi_names

    def test_manifest_columns_and_volumes(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        r.write_per_roi(str(out))

        df = pd.read_csv(out / "manifest.csv")
        for col in ["case_id", "patient_hash", "study_hash", "series_hash",
                    "spacing_x", "spacing_y", "spacing_z"]:
            assert col in df.columns
        # one ``<roi> cc`` column per contour, all present here -> volume > 0.
        for p in synthetic_dataset.primitives:
            col = f"{p.name.lower()} cc"
            assert col in df.columns
            assert (df[col] > 0).all()

    def test_absent_roi_is_negative_one(self, synthetic_dataset, tmp_path: Path):
        # Ask for an ROI that does not exist -> its column should be -1.
        # require_all_contours=False so the series is still indexed despite the
        # missing ROI (this is the union-of-ROIs manifest case).
        r = DicomReaderWriter(
            description="per-roi",
            Contour_Names=[p.name for p in synthetic_dataset.primitives] + ["not_a_real_roi"],
            arg_max=True,
            verbose=False,
            require_all_contours=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        out = tmp_path / "out"
        r.write_per_roi(str(out))

        df = pd.read_csv(out / "manifest.csv")
        assert (df["not_a_real_roi cc"] == -1).all()

    def test_no_masks_subfolder_when_dose_absent(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        r.write_per_roi(str(out))
        case = next(p for p in out.iterdir() if p.is_dir())
        assert not (case / "doses").exists()


class TestWritePerRoiAnonymize:
    def test_folder_named_by_series_hash_and_key_file_written(
        self, synthetic_dataset, tmp_path: Path,
    ):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        r.write_per_roi(str(out), anonymize=True, salt="unit-test-salt")

        idx = r.indexes_with_contours[0]
        series_uid = r.series_instances_dictionary[idx].SeriesInstanceUID
        expected = hash_series(series_uid, salt="unit-test-salt")
        assert (out / expected).is_dir()
        assert (out / "anonymization_key.json").exists()

        # Manifest should carry hashes but not raw identifiers.
        df = pd.read_csv(out / "manifest.csv")
        assert "series_hash" in df.columns
        assert "PatientID" not in df.columns


class TestWritePerRoiResample:
    def test_image_and_mask_resampled_to_spacing(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        target = (3.0, 3.0, 3.0)
        r.write_per_roi(str(out), output_spacing=target)

        case = next(p for p in out.iterdir() if p.is_dir())
        img = sitk.ReadImage(str(case / "image.nii.gz"))
        assert img.GetSpacing() == pytest.approx(target)

        mask = next((case / "masks").glob("*.nii.gz"))
        m = sitk.ReadImage(str(mask))
        assert m.GetSpacing() == pytest.approx(target)
        assert m.GetSize() == img.GetSize()

        df = pd.read_csv(out / "manifest.csv")
        assert df["spacing_x"].iloc[0] == pytest.approx(3.0)


class TestWritePerRoiWithDose:
    def test_dose_subfolder_written(self, synthetic_dataset_with_dose, tmp_path: Path):
        r = _build_reader(synthetic_dataset_with_dose, with_dose=True)
        out = tmp_path / "out"
        r.write_per_roi(str(out))
        case = next(p for p in out.iterdir() if p.is_dir())
        doses = list((case / "doses").glob("*.nii.gz")) if (case / "doses").exists() else []
        assert doses, "expected a dose NIfTI in doses/"

    def test_resampled_dose_shares_image_and_mask_grid(
        self, synthetic_dataset_with_dose, tmp_path: Path,
    ):
        r = _build_reader(synthetic_dataset_with_dose, with_dose=True)
        out = tmp_path / "out"
        r.write_per_roi(str(out), output_spacing=(1.5, 2.5, 3.5))
        case = next(p for p in out.iterdir() if p.is_dir())

        image = sitk.ReadImage(str(case / "image.nii.gz"))
        dose = sitk.ReadImage(str(next((case / "doses").glob("*.nii.gz"))))
        mask = sitk.ReadImage(str(next((case / "masks").glob("*.nii.gz"))))

        # Dose, image, and mask must be voxel-aligned on one grid.
        for other in (dose, mask):
            assert other.GetSize() == image.GetSize()
            assert other.GetSpacing() == pytest.approx(image.GetSpacing())
            assert other.GetOrigin() == pytest.approx(image.GetOrigin())
            assert other.GetDirection() == pytest.approx(image.GetDirection())
