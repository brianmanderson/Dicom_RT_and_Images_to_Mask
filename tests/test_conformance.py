"""Conformance test: DicomRTTool vs RTMaskConformanceTest analytical ground truth.

Generates the RTMaskConformanceTest fixture (synthetic CT + RTSTRUCT + analytic
per-ROI NIfTI ground truth), runs ``DicomReaderWriter`` to convert the
RTSTRUCT into a labeled mask, splits the labels into per-ROI binary NIfTIs,
and asserts each ROI passes the published conformance thresholds (Dice,
Surface DSC @ 1 mm, HD95, MSD, relative volume error).

Unlike ``test_synthetic_roundtrip.py``, the ground truth here is computed
*analytically* (sub-voxel quadrature against the closed-form shape
definitions) — independent of any rasterizer — so a Dice failure here is
a real accuracy regression, not a discretization artifact.

This module is opt-in: it imports the third-party ``rtmask_conformance``
package, which is installed via the ``conformance`` extra::

    pip install -e .[conformance]

If the package is not installed the entire module is skipped via
``pytest.importorskip``, so the default ``pytest`` run is unaffected.

Threshold overrides go in ``tests/conformance.yaml`` (set
``RTMASK_CONFORMANCE_CONFIG`` to use a different file). See
https://github.com/brianmanderson/RTMaskConformanceTest for the schema.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import SimpleITK as sitk

rtmask_conformance = pytest.importorskip(
    "rtmask_conformance",
    reason="install the `conformance` extra: pip install -e .[conformance]",
)

from rtmask_conformance import CONFORMANCE_ROIS, generate_fixture, load_config  # noqa: E402
from rtmask_conformance.generate import GenerateOptions  # noqa: E402
from rtmask_conformance.verify import Status, evaluate_one  # noqa: E402

from DicomRTTool.ReaderWriter import DicomReaderWriter  # noqa: E402

# n_quadrature=2 (8 sub-voxel samples) is enough to make the ground-truth
# masks stable to ~1 voxel of partial-volume disagreement on the boundary,
# which is well below our pass thresholds and an order of magnitude faster
# than the n=8 the published fixtures use.
_FIXTURE_QUADRATURE = 2

# Mapping from CONFORMANCE_ROIS index to the label value DicomReaderWriter
# emits in `reader.mask` when arg_max=True. By construction (Contour_Names
# is passed in CONFORMANCE_ROIS order), label k corresponds to roi
# CONFORMANCE_ROIS[k - 1].


@pytest.fixture(scope="session")
def conformance_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic CT + RTSTRUCT + analytic GT NIfTIs."""
    out = tmp_path_factory.mktemp("rtmask_conformance_fixture")
    generate_fixture(out, options=GenerateOptions(n_quadrature=_FIXTURE_QUADRATURE))
    return out


@pytest.fixture(scope="session")
def dicomrttool_predictions(
    conformance_fixture: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run DicomRTTool against the fixture and write per-ROI binary NIfTIs.

    The conformance verifier expects ``<predictions>/<roi>.nii.gz`` per ROI;
    DicomRTTool emits a single labeled mask, so we split it.
    """
    pred_dir = tmp_path_factory.mktemp("dicomrttool_preds")

    reader = DicomReaderWriter(
        description="conformance",
        Contour_Names=list(CONFORMANCE_ROIS),
        arg_max=True,
        verbose=False,
    )
    reader.walk_through_folders(str(conformance_fixture), thread_count=1)
    if not reader.indexes_with_contours:
        pytest.fail(
            f"DicomReaderWriter found no series with linked RTSTRUCT in {conformance_fixture}"
        )
    reader.set_index(reader.indexes_with_contours[0])
    reader.get_images_and_mask()

    geometry_template = reader.dicom_handle
    for label, roi in enumerate(CONFORMANCE_ROIS, start=1):
        binary = (reader.mask == label).astype("uint8")
        out_img = sitk.GetImageFromArray(binary)
        out_img.CopyInformation(geometry_template)
        sitk.WriteImage(out_img, str(pred_dir / f"{roi}.nii.gz"))
    return pred_dir


_DEFAULT_CONFIG_YAML = Path(__file__).with_name("conformance.yaml")


@pytest.fixture(scope="session")
def conformance_config():
    """Load thresholds.

    Resolution order:
      1. Explicit ``RTMASK_CONFORMANCE_CONFIG`` env var (any path).
      2. ``tests/conformance.yaml`` if it exists (DicomRTTool's calibrated
         relaxations vs the package defaults — see that file's header).
      3. The package-shipped defaults.
    """
    config_path = os.environ.get("RTMASK_CONFORMANCE_CONFIG")
    if config_path is None and _DEFAULT_CONFIG_YAML.is_file():
        config_path = str(_DEFAULT_CONFIG_YAML)
    return load_config(config_path)


@pytest.mark.parametrize("roi", CONFORMANCE_ROIS)
def test_dicomrttool_conformance(
    roi: str,
    conformance_fixture: Path,
    dicomrttool_predictions: Path,
    conformance_config,
):
    """Each ROI: DicomRTTool's mask must match analytic ground truth within
    the published thresholds (Dice, Surface DSC, HD95, MSD, volume error).
    """
    pred_path = dicomrttool_predictions / f"{roi}.nii.gz"
    gt_path = conformance_fixture / "groundtruth" / f"{roi}.nii.gz"
    assert gt_path.is_file(), f"fixture incomplete: {gt_path}"
    assert pred_path.is_file(), f"DicomRTTool produced no mask for {roi!r}"

    result = evaluate_one(roi, pred_path, gt_path, conformance_config)

    if result.status == Status.GEOMETRY_MISMATCH:
        pytest.fail(
            f"{roi}: geometry mismatch between DicomRTTool output and ground truth: "
            f"{result.geometry_diagnostic}"
        )
    if result.status != Status.PASS:
        pytest.fail(
            f"{roi}: {result.status.value}\n"
            f"  violations: {result.violations}\n"
            f"  metrics:    {result.metrics}\n"
            f"  thresholds: {result.thresholds}"
        )
