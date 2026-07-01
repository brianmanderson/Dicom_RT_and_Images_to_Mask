"""Opt-in parity test: DicomRTTool (Python) vs the C# DicomRtNifti.Cli tool.

This test is **skipped** unless both environment variables are set, so the
normal hermetic suite (and CI) never needs the external LCTSC corpus or the
built C# binary:

* ``DICOMRTTOOL_LCTSC_DIR`` — path to the LCTSC patient-root directory
  (``.../datasets/v2/lctsc``);
* ``DICOMRTTOOL_CSHARP_EXE`` — path to ``DicomRtNifti.Cli.exe``.

Optional: ``DICOMRTTOOL_PARITY_N`` (patient count, default 2).

The thresholds are intentionally tolerant: the two tools use different
polygon-boundary conventions, so masks agree closely but are not identical.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

LCTSC_DIR = os.environ.get("DICOMRTTOOL_LCTSC_DIR")
CSHARP_EXE = os.environ.get("DICOMRTTOOL_CSHARP_EXE")

pytestmark = pytest.mark.skipif(
    not (LCTSC_DIR and CSHARP_EXE and Path(LCTSC_DIR).is_dir() and Path(CSHARP_EXE).is_file()),
    reason="Set DICOMRTTOOL_LCTSC_DIR and DICOMRTTOOL_CSHARP_EXE to run the C# parity test.",
)


@pytest.fixture(scope="module")
def parity_result(tmp_path_factory):
    from evaluation.csharp_eval import evaluate

    n = int(os.environ.get("DICOMRTTOOL_PARITY_N", "2"))
    work = tmp_path_factory.mktemp("csharp_parity")
    return evaluate(
        dataset_dir=LCTSC_DIR,
        csharp_exe=CSHARP_EXE,
        work_dir=str(work),
        num_patients=n,
        target_spacing=(3.0, 3.0, 3.0),
    )


def test_masks_agree_within_boundary_tolerance(parity_result):
    assert parity_result.mask_rows, "no mask comparisons were produced"
    dices = [r["dice"] for r in parity_result.mask_rows]
    # Boundary-convention differences keep this below 1.0 but well above 0.9
    # for these large thoracic OARs.
    assert np.median(dices) > 0.95, f"median Dice too low: {np.median(dices):.4f}"
    assert np.min(dices) > 0.80, f"a mask diverged badly: min Dice {np.min(dices):.4f}"


def test_native_images_match_closely(parity_result):
    assert parity_result.image_rows, "no image comparisons were produced"
    # Both tools read the same DICOM via SimpleITK -> near-identical voxels.
    assert max(r["mean_abs_error"] for r in parity_result.image_rows) < 1.0


def test_resample_parity(parity_result):
    assert parity_result.resample_rows, "no resample comparisons were produced"
    # Both resample the same image to the same spacing with linear interp.
    assert max(r["mean_abs_error"] for r in parity_result.resample_rows) < 5.0
