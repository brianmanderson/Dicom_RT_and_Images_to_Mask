"""Opt-in dose fidelity check against a real DICOM corpus.

The rest of the suite is hermetic and synthetic. This module runs the same
checks against **one real patient** — the last mile that synthetic data cannot
cover, because only real DICOM has the quirks (trailing spaces in descriptions,
plans withheld from a public release, dose grids unrelated to the image grid)
that the code has to survive.

It auto-skips unless you point it at a folder::

    # one patient's CT + RTSTRUCT + RTDOSE, in any folder layout
    export DICOMRTTOOL_DOSE_CORPUS=/path/to/patient
    pytest tests/test_real_corpus_dose.py

Verified against TCIA ``Pancreatic-CT-CBCT-SEG`` patient ``Pancreas-CT-CB_002``
(planning CT + 3 structure sets + 1 dose): Dmax **3912.09 cGy**, DoseUnits
``GY``, no RT Plan in the collection so the plan name falls back to the dose's
series description.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter

CORPUS = os.environ.get("DICOMRTTOOL_DOSE_CORPUS")

pytestmark = pytest.mark.skipif(
    not CORPUS or not Path(CORPUS).is_dir(),
    reason="Set DICOMRTTOOL_DOSE_CORPUS to a folder holding one patient's CT + RTSTRUCT + RTDOSE.",
)

_UNITS_TO_CGY = {"GY": 100.0, "CGY": 1.0}


def _peak_cgy(path: Path) -> float:
    """Maximum voxel of a dose NIfTI, in cGy.

    The image must be bound to a name that outlives the array view:
    ``GetArrayViewFromImage`` is zero-copy, so taking a view of an inline
    ``sitk.ReadImage(...)`` temporary reads freed memory once that temporary is
    collected — which segfaults rather than failing cleanly.
    """
    image = sitk.ReadImage(str(path))
    return float(sitk.GetArrayViewFromImage(image).max()) * 100.0


def _find_dose_files(root: Path) -> list[Path]:
    """Every RTDOSE instance under *root*, found by reading Modality."""
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "Modality", None) == "RTDOSE":
            found.append(path)
    return found


@pytest.fixture(scope="module")
def truth() -> dict:
    """Ground truth for the corpus's dose, from pydicom alone."""
    dose_files = _find_dose_files(Path(CORPUS))
    assert dose_files, f"no RTDOSE found under {CORPUS}"
    assert len(dose_files) == 1, f"expected one dose series, found {len(dose_files)}"

    ds = pydicom.dcmread(str(dose_files[0]))
    scaling = float(ds.DoseGridScaling)
    to_cgy = _UNITS_TO_CGY[ds.DoseUnits.upper()]
    offsets = [float(v) for v in ds.GridFrameOffsetVector]
    row_spacing, col_spacing = (float(v) for v in ds.PixelSpacing)
    return {
        "path": dose_files[0],
        "patient_id": ds.PatientID,
        "study_uid": ds.StudyInstanceUID,
        "series_uid": ds.SeriesInstanceUID,
        "dose_units": ds.DoseUnits,
        "summation": ds.DoseSummationType,
        "dmax_cgy": round(float(ds.pixel_array.max()) * scaling * to_cgy, 2),
        "spacing": (col_spacing, row_spacing, abs(offsets[1] - offsets[0])),
    }


@pytest.fixture(scope="module")
def exported(tmp_path_factory, truth) -> dict:
    """Walk the corpus once and export it resampled."""
    out = tmp_path_factory.mktemp("real_dose_export")
    reader = DicomReaderWriter(get_dose_output=True, require_all_contours=False, verbose=False)
    reader.walk_through_folders(CORPUS, thread_count=2)

    rois = [r.lower() for r in reader.return_rois(print_rois=False)]
    reader.set_contour_names_and_associations(contour_names=rois)

    manifest = out / "survey.csv"
    reader.create_manifest(str(manifest), anonymize=False)

    export_dir = out / "nifti"
    output_spacing = (1.5, 1.5, 3.0)
    reader.write_to_folder(
        str(export_dir), output_spacing=output_spacing,
        anonymize=False, thread_count=2,
    )
    return {
        "reader": reader,
        "manifest": pd.read_csv(manifest),
        "export_dir": export_dir,
        "output_spacing": output_spacing,
    }


def test_manifest_dose_row_matches_the_dicom(truth, exported):
    rows = exported["manifest"][exported["manifest"]["modality"] == "RTDOSE"]
    assert len(rows) == 1
    row = rows.iloc[0]

    assert row["patient_hash"] == truth["patient_id"]
    assert row["study_hash"] == truth["study_uid"]
    assert row["series_hash"] == truth["series_uid"]
    assert (row["spacing_x"], row["spacing_y"], row["spacing_z"]) == pytest.approx(
        truth["spacing"]
    )

    dose_columns = [c for c in exported["manifest"].columns if c.endswith(" cGy")]
    assert len(dose_columns) == 1, f"expected one plan column, got {dose_columns}"
    assert row[dose_columns[0]] == pytest.approx(truth["dmax_cgy"], abs=0.05)


def test_dose_row_shares_patient_and_study_with_the_images(truth, exported):
    manifest = exported["manifest"]
    dose_row = manifest[manifest["modality"] == "RTDOSE"].iloc[0]
    image_rows = manifest[manifest["modality"] != "RTDOSE"]
    assert not image_rows.empty, "expected at least one image series"

    assert (image_rows["patient_hash"] == dose_row["patient_hash"]).all()
    assert dose_row["series_hash"] not in set(image_rows["series_hash"])


def test_sidecar_dose_matches_the_dicom(truth, exported):
    sidecars = list(exported["export_dir"].rglob("metadata.json"))
    assert sidecars, "no metadata.json was written"

    checked = 0
    for sidecar in sidecars:
        meta = json.loads(sidecar.read_text())
        for dose in meta.get("doses", []):
            if dose["series_instance_uid"] != truth["series_uid"]:
                continue
            assert dose["dose_units"] == truth["dose_units"]
            assert dose["dose_summation_type"] == truth["summation"]
            assert dose["dose_max_cgy"] == pytest.approx(truth["dmax_cgy"], abs=0.05)
            assert dose["plan_name"], "a dose must always end up with some name"
            checked += 1
    assert checked, "the dose never appeared in any sidecar"


def test_exported_dose_shares_the_image_grid(exported):
    """The point of the export: image, masks, and dose on one grid."""
    cases = [p.parent for p in exported["export_dir"].rglob("image.nii.gz")]
    assert cases, "nothing was exported"

    checked = 0
    for case in cases:
        dose_files = sorted((case / "doses").glob("*.nii.gz")) if (case / "doses").exists() else []
        if not dose_files:
            continue
        image = sitk.ReadImage(str(case / "image.nii.gz"))
        assert image.GetSpacing() == pytest.approx(exported["output_spacing"])

        mask_files = sorted((case / "masks").glob("*.nii.gz")) if (case / "masks").exists() else []
        for path in mask_files + dose_files:
            other = sitk.ReadImage(str(path))
            assert other.GetSize() == image.GetSize(), f"{case.name}/{path.name} size"
            assert other.GetSpacing() == pytest.approx(image.GetSpacing()), f"{path.name} spacing"
            assert other.GetOrigin() == pytest.approx(image.GetOrigin()), f"{path.name} origin"
            assert other.GetDirection() == pytest.approx(image.GetDirection()), f"{path.name} direction"
        checked += 1
    assert checked, "no exported case carried a dose to check"


def test_exported_dose_keeps_physical_values(truth, exported):
    """Resampling relocates the dose; it must not rescale it.

    The peak can only fall — never rise — once interpolated onto a coarser grid.
    """
    dose_files = list(exported["export_dir"].rglob("doses/*.nii.gz"))
    assert dose_files, "no dose was exported"

    peak_cgy = max(_peak_cgy(p) for p in dose_files)
    assert peak_cgy <= truth["dmax_cgy"] + 0.05, "exported dose exceeds the native maximum"
    assert peak_cgy > truth["dmax_cgy"] * 0.85, "exported dose lost far too much of its peak"


def test_roi_volumes_match_the_exported_masks(exported):
    manifest = exported["manifest"]
    checked = 0
    for case in [p.parent for p in exported["export_dir"].rglob("image.nii.gz")]:
        masks_dir = case / "masks"
        if not masks_dir.exists():
            continue
        meta = json.loads((case / "metadata.json").read_text())
        series_uid = meta["case"]["series"]
        rows = manifest[manifest["series_hash"].astype(str).str.contains(series_uid, regex=False)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for mask_path in sorted(masks_dir.glob("*.nii.gz")):
            column = f"{mask_path.name.replace('.nii.gz', '')} cc"
            if column not in manifest.columns or pd.isna(row[column]):
                continue
            mask = sitk.ReadImage(str(mask_path))
            voxel_cc = float(np.prod(mask.GetSpacing())) / 1000.0
            counted = float(sitk.GetArrayViewFromImage(mask).sum()) * voxel_cc
            # The manifest volume is measured on the *native* grid, the mask on
            # the resampled one, so allow a small resampling difference.
            assert counted == pytest.approx(row[column], rel=0.05)
            checked += 1
    assert checked, "no ROI volume could be cross-checked"
