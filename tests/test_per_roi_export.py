"""Tests for the C#-compatible folder export (``write_to_folder``)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import SimpleITK as sitk

from DicomRTTool import hash_patient, hash_series, hash_study
from DicomRTTool.ReaderWriter import DicomReaderWriter


def _build_reader(dataset, *, with_dose: bool = False, image_keys=None) -> DicomReaderWriter:
    r = DicomReaderWriter(
        description="per-roi",
        Contour_Names=[p.name for p in dataset.primitives],
        arg_max=True,
        verbose=False,
        get_dose_output=with_dose,
        image_sitk_string_keys=image_keys,
    )
    r.walk_through_folders(str(dataset.walk_root), thread_count=1)
    return r


def _case_dir(out: Path) -> Path:
    """The series folder (parent of image.nii.gz) in the nested output tree."""
    return next(p for p in out.rglob("image.nii.gz")).parent


class TestWritePerRoiLayout:
    @pytest.mark.parametrize("thread_count", [1, 2])
    def test_writes_image_masks_and_manifest(
        self, synthetic_dataset, tmp_path: Path, thread_count: int,
    ):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / f"out_t{thread_count}"
        r.write_to_folder(str(out), thread_count=thread_count)

        assert (out / "manifest.csv").exists()
        case = _case_dir(out)
        assert (case / "image.nii.gz").exists()
        masks = list((case / "masks").glob("*.nii.gz"))
        roi_names = {p.name.lower() for p in synthetic_dataset.primitives}
        assert {m.stem.replace(".nii", "").lower() for m in masks} == roi_names

    def test_nested_patient_study_series_structure(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        r.write_to_folder(str(out), anonymize=True, salt="s")

        entry = r.series_instances_dictionary[r.indexes_with_contours[0]]
        case = (
            out
            / hash_patient(entry.PatientID, "s")
            / hash_study(entry.StudyInstanceUID, "s")
            / hash_series(entry.SeriesInstanceUID, "s")
        )
        assert (case / "image.nii.gz").exists()

    def test_manifest_columns_and_volumes(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        r.write_to_folder(str(out))

        df = pd.read_csv(out / "manifest.csv")
        for col in ["case_id", "patient_hash", "study_hash", "series_hash",
                    "spacing_x", "spacing_y", "spacing_z"]:
            assert col in df.columns
        for p in synthetic_dataset.primitives:
            col = f"{p.name.lower()} cc"
            assert col in df.columns
            assert (df[col] > 0).all()

    def test_absent_roi_is_negative_one(self, synthetic_dataset, tmp_path: Path):
        r = DicomReaderWriter(
            description="per-roi",
            Contour_Names=[p.name for p in synthetic_dataset.primitives] + ["not_a_real_roi"],
            arg_max=True,
            verbose=False,
            require_all_contours=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        out = tmp_path / "out"
        r.write_to_folder(str(out))

        df = pd.read_csv(out / "manifest.csv")
        assert (df["not_a_real_roi cc"] == -1).all()

    def test_no_dose_subfolder_when_dose_absent(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        r.write_to_folder(str(out))
        assert not (_case_dir(out) / "doses").exists()


class TestWritePerRoiMetadata:
    def test_metadata_json_written_for_extra_tags(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset, image_keys={"MyPatientID": "0010|0020", "MyModality": "0008|0060"})
        out = tmp_path / "out"
        r.write_to_folder(str(out))

        meta_path = _case_dir(out) / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta == {"MyPatientID": "DICOMRTTOOL_TEST", "MyModality": "CT"}

    def test_no_metadata_json_without_extra_tags(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)   # no *_string_keys requested
        out = tmp_path / "out"
        r.write_to_folder(str(out))
        assert not list(out.rglob("metadata.json"))


class TestWritePerRoiAnonymize:
    def test_folders_named_by_hash_and_key_file_written(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        r.write_to_folder(str(out), anonymize=True, salt="unit-test-salt")

        entry = r.series_instances_dictionary[r.indexes_with_contours[0]]
        case = (
            out
            / hash_patient(entry.PatientID, "unit-test-salt")
            / hash_study(entry.StudyInstanceUID, "unit-test-salt")
            / hash_series(entry.SeriesInstanceUID, "unit-test-salt")
        )
        assert case.is_dir()
        assert (out / "anonymization_key.json").exists()

        df = pd.read_csv(out / "manifest.csv")
        assert "series_hash" in df.columns
        assert "PatientID" not in df.columns


class TestWritePerRoiResample:
    def test_image_and_mask_resampled_to_spacing(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        target = (3.0, 3.0, 3.0)
        r.write_to_folder(str(out), output_spacing=target)

        case = _case_dir(out)
        img = sitk.ReadImage(str(case / "image.nii.gz"))
        assert img.GetSpacing() == pytest.approx(target)

        m = sitk.ReadImage(str(next((case / "masks").glob("*.nii.gz"))))
        assert m.GetSpacing() == pytest.approx(target)
        assert m.GetSize() == img.GetSize()

        df = pd.read_csv(out / "manifest.csv")
        assert df["spacing_x"].iloc[0] == pytest.approx(3.0)


class TestWritePerRoiWithDose:
    def test_dose_subfolder_written(self, synthetic_dataset_with_dose, tmp_path: Path):
        r = _build_reader(synthetic_dataset_with_dose, with_dose=True)
        out = tmp_path / "out"
        r.write_to_folder(str(out))
        case = _case_dir(out)
        doses = list((case / "doses").glob("*.nii.gz")) if (case / "doses").exists() else []
        assert doses, "expected a dose NIfTI in doses/"

    def test_resampled_dose_shares_image_and_mask_grid(
        self, synthetic_dataset_with_dose, tmp_path: Path,
    ):
        r = _build_reader(synthetic_dataset_with_dose, with_dose=True)
        out = tmp_path / "out"
        r.write_to_folder(str(out), output_spacing=(1.5, 2.5, 3.5))
        case = _case_dir(out)

        image = sitk.ReadImage(str(case / "image.nii.gz"))
        dose = sitk.ReadImage(str(next((case / "doses").glob("*.nii.gz"))))
        mask = sitk.ReadImage(str(next((case / "masks").glob("*.nii.gz"))))

        for other in (dose, mask):
            assert other.GetSize() == image.GetSize()
            assert other.GetSpacing() == pytest.approx(image.GetSpacing())
            assert other.GetOrigin() == pytest.approx(image.GetOrigin())
            assert other.GetDirection() == pytest.approx(image.GetDirection())


class TestWriteToFolderImageOnly:
    def test_no_rois_exports_image_only(self, synthetic_dataset, tmp_path: Path):
        # No Contour_Names and no rois= -> image only, no masks folder.
        r = DicomReaderWriter(description="img", verbose=False)
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        out = tmp_path / "out"
        r.write_to_folder(str(out))

        case = _case_dir(out)
        assert (case / "image.nii.gz").exists()
        assert not (case / "masks").exists(), "no empty masks/ folder for image-only export"
        assert (out / "manifest.csv").exists()

    def test_no_rois_exports_image_and_dose(self, synthetic_dataset_with_dose, tmp_path: Path):
        r = DicomReaderWriter(description="img", verbose=False, get_dose_output=True)
        r.walk_through_folders(str(synthetic_dataset_with_dose.walk_root), thread_count=1)
        out = tmp_path / "out"
        r.write_to_folder(str(out))

        case = _case_dir(out)
        assert (case / "image.nii.gz").exists()
        assert list((case / "doses").glob("*.nii.gz")), "dose should be exported when present"
        assert not (case / "masks").exists()


class TestWriteToFolderAnonymizeDefault:
    def test_constructor_anonymize_default(self, synthetic_dataset, tmp_path: Path):
        r = DicomReaderWriter(
            description="per-roi",
            Contour_Names=[p.name for p in synthetic_dataset.primitives],
            verbose=False,
            anonymize=True, salt="ctor-salt",
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        out = tmp_path / "out"
        r.write_to_folder(str(out))   # no anonymize= passed -> uses ctor default

        entry = r.series_instances_dictionary[r.indexes_with_contours[0]]
        case = (
            out
            / hash_patient(entry.PatientID, "ctor-salt")
            / hash_study(entry.StudyInstanceUID, "ctor-salt")
            / hash_series(entry.SeriesInstanceUID, "ctor-salt")
        )
        assert (case / "image.nii.gz").exists()
        assert (out / "anonymization_key.json").exists()

    def test_per_call_overrides_constructor_default(self, synthetic_dataset, tmp_path: Path):
        r = DicomReaderWriter(
            description="per-roi",
            Contour_Names=[p.name for p in synthetic_dataset.primitives],
            verbose=False,
            anonymize=True, salt="ctor-salt",
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        out = tmp_path / "out"
        r.write_to_folder(str(out), anonymize=False)   # override off

        entry = r.series_instances_dictionary[r.indexes_with_contours[0]]
        # Folder named by the raw (sanitised) PatientID, not the hash.
        assert (out / str(entry.PatientID)).is_dir()
        assert not (out / "anonymization_key.json").exists()
        df = pd.read_csv(out / "manifest.csv")
        assert str(entry.PatientID) in df["patient_hash"].astype(str).tolist()


class TestBackwardCompatAlias:
    def test_write_per_roi_alias(self, synthetic_dataset, tmp_path: Path):
        r = _build_reader(synthetic_dataset)
        out = tmp_path / "out"
        r.write_per_roi(str(out))   # deprecated alias still works
        assert (out / "manifest.csv").exists()
        assert (_case_dir(out) / "image.nii.gz").exists()
