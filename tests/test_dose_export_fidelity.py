"""End-to-end fidelity checks for dose: DICOM in, manifest / sidecar / NIfTI out.

Every expected value here is derived **independently with pydicom**, straight
from the DICOM tags — never through SimpleITK or DicomRTTool. If the library and
this module ever agree only because they share a bug, these tests stop being
worth anything, so they deliberately re-derive Dmax, grid spacing, and every
identifier from the raw headers.

The second half covers the geometric contract of an export: the dose is written
on a **genuinely different native grid** from the CT here (finer, like real
planning data), so "the exported dose matches the image grid" is a real
assertion rather than a tautology.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
from tests.synthetic import (
    CTSeriesUIDs,
    Geometry,
    build_synthetic_dataset,
    build_synthetic_dose,
    build_synthetic_plan,
)

# A dose grid that does NOT match the CT: finer in-plane and in z, covering
# roughly the same physical extent. Mirrors real planning data, where the dose
# grid is its own thing.
DOSE_GEOMETRY = Geometry(origin=(0.0, 0.0, 0.0), spacing=(1.25, 1.25, 1.0), size=(102, 102, 64))

PEAK_DOSE_GY = 54.0            # what the synthetic dose grid peaks at, in Gy
EXPECTED_DMAX_CGY = 5400.0     # ...which is this in cGy
PLAN_NAME = "Fidelity Plan"


@pytest.fixture
def corpus(tmp_path: Path):
    """CT + RTSTRUCT + RT Plan + an RT Dose on its own finer grid."""
    image_dir, rt_path, geometry, primitives = build_synthetic_dataset(tmp_path)
    rt_sop = pydicom.dcmread(str(rt_path), stop_before_pixels=True).SOPInstanceUID
    first = pydicom.dcmread(
        str(sorted(image_dir.glob("*.dcm"))[0]), stop_before_pixels=True,
    )
    uids = CTSeriesUIDs(
        study=first.StudyInstanceUID,
        series=first.SeriesInstanceUID,
        frame_of_reference=first.FrameOfReferenceUID,
    )
    _, plan_sop = build_synthetic_plan(
        tmp_path / "RP.dcm", uids, rt_struct_sop_uid=rt_sop, plan_name=PLAN_NAME,
    )
    dose_path = build_synthetic_dose(
        tmp_path / "RD.dcm", DOSE_GEOMETRY, uids,
        rt_struct_sop_uid=rt_sop, rt_plan_sop_uid=plan_sop,
        dose_units="GY", peak_dose=PEAK_DOSE_GY,
    )

    reader = DicomReaderWriter(
        Contour_Names=[p.name for p in primitives],
        associations=[ROIAssociationClass(p.name, [p.name]) for p in primitives],
        get_dose_output=True,
        verbose=False,
    )
    reader.walk_through_folders(str(tmp_path), thread_count=1)

    return {
        "root": tmp_path,
        "reader": reader,
        "dose_path": dose_path,
        "image_dir": image_dir,
        "geometry": geometry,
        "primitives": primitives,
    }


# ---------------------------------------------------------------------------
# Ground truth, read straight from the DICOM with pydicom
# ---------------------------------------------------------------------------

def dicom_truth_dose(dose_path: Path) -> dict:
    """Everything we expect about the dose, derived only from its DICOM tags."""
    ds = pydicom.dcmread(str(dose_path))
    scaling = float(ds.DoseGridScaling)
    peak_native = float(ds.pixel_array.max()) * scaling
    to_cgy = {"GY": 100.0, "CGY": 1.0}[ds.DoseUnits.upper()]

    # Dose grid spacing: PixelSpacing is [row (y), column (x)]; z comes from the
    # frame offsets, not SliceThickness.
    offsets = [float(v) for v in ds.GridFrameOffsetVector]
    row_spacing, col_spacing = (float(v) for v in ds.PixelSpacing)

    return {
        "patient_id": ds.PatientID,
        "study_uid": ds.StudyInstanceUID,
        "series_uid": ds.SeriesInstanceUID,
        "sop_uid": ds.SOPInstanceUID,
        "dose_units": ds.DoseUnits,
        "dose_type": ds.DoseType,
        "summation": ds.DoseSummationType,
        "dmax_cgy": round(peak_native * to_cgy, 2),
        "spacing": (col_spacing, row_spacing, abs(offsets[1] - offsets[0])),
        "referenced_plan_sop": ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID,
    }


def dicom_truth_image(image_dir: Path) -> dict:
    """Image identifiers and grid spacing, derived only from the CT slices."""
    slices = [
        pydicom.dcmread(str(p), stop_before_pixels=True)
        for p in sorted(image_dir.glob("*.dcm"))
    ]
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    row_spacing, col_spacing = (float(v) for v in slices[0].PixelSpacing)
    z_spacing = abs(
        float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2])
    )
    return {
        "patient_id": slices[0].PatientID,
        "study_uid": slices[0].StudyInstanceUID,
        "series_uid": slices[0].SeriesInstanceUID,
        "spacing": (col_spacing, row_spacing, z_spacing),
        "n_slices": len(slices),
    }


# ---------------------------------------------------------------------------
class TestManifestAgainstDicom:
    """Every manifest cell must trace back to a DICOM tag."""

    def test_dose_row_matches_the_dose_dicom(self, corpus):
        truth = dicom_truth_dose(corpus["dose_path"])
        manifest_path = corpus["root"] / "manifest.csv"
        corpus["reader"].create_manifest(str(manifest_path), anonymize=False)
        df = pd.read_csv(manifest_path)

        rows = df[df["modality"] == "RTDOSE"]
        assert len(rows) == 1
        row = rows.iloc[0]

        # Identifiers are the dose's own, not the image series' it links to.
        assert row["patient_hash"] == truth["patient_id"]
        assert row["study_hash"] == truth["study_uid"]
        assert row["series_hash"] == truth["series_uid"]

        # Spacing is the dose grid's, from PixelSpacing + GridFrameOffsetVector.
        assert (row["spacing_x"], row["spacing_y"], row["spacing_z"]) == pytest.approx(
            truth["spacing"]
        )

        # Dmax under the plan's own column.
        assert row[f"{PLAN_NAME} cGy"] == pytest.approx(truth["dmax_cgy"], abs=0.05)
        assert truth["dmax_cgy"] == pytest.approx(EXPECTED_DMAX_CGY, abs=0.05)

    def test_image_row_matches_the_ct_dicom(self, corpus):
        truth = dicom_truth_image(corpus["image_dir"])
        manifest_path = corpus["root"] / "manifest.csv"
        corpus["reader"].create_manifest(str(manifest_path), anonymize=False)
        df = pd.read_csv(manifest_path)

        rows = df[df["modality"] == "CT"]
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row["patient_hash"] == truth["patient_id"]
        assert row["study_hash"] == truth["study_uid"]
        assert row["series_hash"] == truth["series_uid"]
        assert (row["spacing_x"], row["spacing_y"], row["spacing_z"]) == pytest.approx(
            truth["spacing"]
        )

    def test_dose_and_image_rows_agree_on_patient_and_study(self, corpus):
        """Same patient and study, different series -- the linkage that lets you
        join a dose row back to the images it belongs to."""
        dose = dicom_truth_dose(corpus["dose_path"])
        image = dicom_truth_image(corpus["image_dir"])
        assert dose["patient_id"] == image["patient_id"]
        assert dose["study_uid"] == image["study_uid"]
        assert dose["series_uid"] != image["series_uid"]

        manifest_path = corpus["root"] / "manifest.csv"
        corpus["reader"].create_manifest(str(manifest_path), anonymize=True, salt="fidelity")
        df = pd.read_csv(manifest_path)
        dose_row = df[df["modality"] == "RTDOSE"].iloc[0]
        image_row = df[df["modality"] == "CT"].iloc[0]

        assert dose_row["patient_hash"] == image_row["patient_hash"]
        assert dose_row["study_hash"] == image_row["study_hash"]
        assert dose_row["series_hash"] != image_row["series_hash"]

    def test_roi_volume_matches_the_exported_mask(self, corpus):
        """The manifest's `<roi> cc` must equal what is actually in the mask."""
        out = corpus["root"] / "out"
        corpus["reader"].write_to_folder(str(out), thread_count=1)
        df = pd.read_csv(out / "manifest.csv")
        row = df[df["modality"] == "CT"].iloc[0]

        for mask_path in sorted((next(out.rglob("image.nii.gz")).parent / "masks").glob("*.nii.gz")):
            roi = mask_path.name.replace(".nii.gz", "")
            mask = sitk.ReadImage(str(mask_path))
            voxel_cc = float(np.prod(mask.GetSpacing())) / 1000.0
            counted = float(sitk.GetArrayViewFromImage(mask).sum()) * voxel_cc
            assert row[f"{roi} cc"] == pytest.approx(counted, abs=0.01)


class TestSidecarAgainstDicom:
    def test_metadata_dose_block_matches_the_dose_dicom(self, corpus):
        truth = dicom_truth_dose(corpus["dose_path"])
        out = corpus["root"] / "out"
        corpus["reader"].write_to_folder(str(out), metadata_style="grouped", thread_count=1)

        meta = json.loads(next(out.rglob("metadata.json")).read_text())
        assert len(meta["doses"]) == 1
        dose = meta["doses"][0]

        assert dose["series_instance_uid"] == truth["series_uid"]
        assert dose["sop_instance_uid"] == truth["sop_uid"]
        assert dose["dose_units"] == truth["dose_units"]
        assert dose["dose_type"] == truth["dose_type"]
        assert dose["dose_summation_type"] == truth["summation"]
        assert dose["referenced_plan_sop_instance_uid"] == truth["referenced_plan_sop"]
        assert dose["dose_max_cgy"] == pytest.approx(truth["dmax_cgy"], abs=0.05)
        assert dose["plan_name"] == PLAN_NAME

    def test_sidecar_and_manifest_report_the_same_dmax(self, corpus):
        out = corpus["root"] / "out"
        corpus["reader"].write_to_folder(str(out), metadata_style="grouped", thread_count=1)

        meta = json.loads(next(out.rglob("metadata.json")).read_text())
        df = pd.read_csv(out / "manifest.csv")
        from_manifest = df[df["modality"] == "RTDOSE"].iloc[0][f"{PLAN_NAME} cGy"]
        assert meta["doses"][0]["dose_max_cgy"] == pytest.approx(from_manifest, abs=0.005)


class TestExportedDoseGeometry:
    """The dose lands on the image grid -- native export and resampled alike."""

    def test_the_dose_really_does_start_on_a_different_grid(self, corpus):
        """Guards the premise of the tests below."""
        truth = dicom_truth_dose(corpus["dose_path"])
        image = dicom_truth_image(corpus["image_dir"])
        assert truth["spacing"] != pytest.approx(image["spacing"])

    def _assert_all_outputs_share_geometry(self, case: Path) -> sitk.Image:
        image = sitk.ReadImage(str(case / "image.nii.gz"))
        others = sorted((case / "masks").glob("*.nii.gz")) + sorted(
            (case / "doses").glob("*.nii.gz")
        )
        assert others, "expected masks and a dose alongside the image"
        for path in others:
            other = sitk.ReadImage(str(path))
            assert other.GetSize() == image.GetSize(), f"{path.name} size"
            assert other.GetSpacing() == pytest.approx(image.GetSpacing()), f"{path.name} spacing"
            assert other.GetOrigin() == pytest.approx(image.GetOrigin()), f"{path.name} origin"
            assert other.GetDirection() == pytest.approx(image.GetDirection()), f"{path.name} direction"
        return image

    def test_native_export_puts_the_dose_on_the_ct_grid(self, corpus):
        out = corpus["root"] / "out"
        corpus["reader"].write_to_folder(str(out), output_spacing=None, thread_count=1)

        case = next(out.rglob("image.nii.gz")).parent
        image = self._assert_all_outputs_share_geometry(case)

        # ...and that grid is the CT's own, unresampled.
        truth = dicom_truth_image(corpus["image_dir"])
        assert image.GetSpacing() == pytest.approx(truth["spacing"])
        assert image.GetSize()[2] == truth["n_slices"]

    @pytest.mark.parametrize("output_spacing", [(1.0, 1.0, 1.0), (3.0, 3.0, 5.0)])
    def test_resampled_export_puts_everything_on_the_new_grid(
        self, corpus, output_spacing,
    ):
        out = corpus["root"] / f"out_{output_spacing[0]}_{output_spacing[2]}"
        corpus["reader"].write_to_folder(
            str(out), output_spacing=output_spacing, thread_count=1,
        )

        case = next(out.rglob("image.nii.gz")).parent
        image = self._assert_all_outputs_share_geometry(case)
        assert image.GetSpacing() == pytest.approx(output_spacing)

    def test_exported_dose_values_stay_physical(self, corpus):
        """Resampling moves dose onto a new grid; it must not rescale it.

        The exported dose is in the DICOM's own units (Gy here), so its peak
        should approach -- never exceed -- the native maximum.
        """
        truth = dicom_truth_dose(corpus["dose_path"])
        out = corpus["root"] / "out"
        corpus["reader"].write_to_folder(str(out), output_spacing=(1.0, 1.0, 1.0), thread_count=1)

        dose = sitk.ReadImage(str(next(out.rglob("doses/*.nii.gz"))))
        peak_cgy = float(sitk.GetArrayViewFromImage(dose).max()) * 100.0
        assert peak_cgy <= truth["dmax_cgy"] + 0.05
        assert peak_cgy > truth["dmax_cgy"] * 0.9
