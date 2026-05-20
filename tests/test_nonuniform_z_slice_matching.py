"""Regression test for the non-uniform-Z slice-matching fix.

Background
==========

Clinical CT acquisitions sometimes have **non-uniform Z spacing** -- e.g.
mixed 3 mm and 6 mm slice gaps where the scanner skipped or doubled up
on certain regions. This is common on the NSCLC-Radiomics public cohort
(LUNG1-014/-021/-085/-095/-194/-246 all have it).

The legacy ``DicomReaderWriter.reshape_contour_data`` used SimpleITK's
``TransformPhysicalPointToIndex`` to map every contour point's physical
coordinate to a voxel index. On non-uniform-Z series, ITK's
``ImageSeriesReader`` compresses the per-slice positions into a single
*averaged* spacing[2]; the per-point index then rounds against that
averaged spacing and misses contour planes that sit on the irregular
side of the gap. Empirically that costs 11-14 of ~73 contour planes on
LUNG1-014's lung ROIs.

The fix caches the per-DICOM ``ImagePositionPatient[2]`` array during
``get_images()`` and resolves each contour plane's Z to the nearest
actual slice instead.

This test exercises three layers:

1. The per-slice Z array is populated on a uniform-Z synthetic dataset
   and matches the uniform grid (no regression).
2. The per-slice Z array is populated correctly when loading a
   non-uniform-Z CT series.
3. ``reshape_contour_data`` resolves contour Z's against the per-slice
   array when available, falling back to the legacy
   ``TransformPhysicalPointToIndex`` path otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter

from . import synthetic


# ---------------------------------------------------------------------------
# Layer 1 -- uniform-Z synthetic dataset (regression: fix is a no-op here)
# ---------------------------------------------------------------------------


def test_per_slice_z_populated_on_uniform_dataset(synthetic_dataset):
    """After loading the uniform-Z synthetic dataset that the rest of the
    test suite uses, the reader caches the per-DICOM IPP[2] array with
    the exact uniform spacing the dataset was built with.

    A regression in ``get_images()`` would leave the array unset (the
    fix would silently degrade to the legacy path) or populate it with
    the wrong values; both are caught here."""
    r = DicomReaderWriter(
        description="uniform_z_check",
        Contour_Names=[p.name for p in synthetic_dataset.primitives],
        arg_max=True,
        verbose=False,
    )
    r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
    r.set_index(r.indexes_with_contours[0])
    r.get_images_and_mask()

    assert r._dicom_slice_z_positions is not None
    z_arr = np.asarray(r._dicom_slice_z_positions)
    assert len(z_arr) == synthetic_dataset.geometry.size[2]
    diffs = np.diff(z_arr)
    sz = synthetic_dataset.geometry.spacing[2]
    np.testing.assert_allclose(diffs, sz, atol=1e-6)


# ---------------------------------------------------------------------------
# Layer 2 -- non-uniform-Z CT: per-slice Z array tracks the mixed gaps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nonuniform_z_dataset(tmp_path_factory) -> dict:
    """Build a synthetic CT then rewrite its per-slice IPP[2] to a
    mixed-3/6 mm pattern. The RTSTRUCT shipped alongside the CT is
    intentionally aligned to the *original* (uniform 1 mm) Z grid -- the
    test below only inspects the cached per-slice array, not the
    mask itself, so the contour positions don't need to match the new
    IPPs.

    Returns the dataset's filesystem layout and the expected per-slice
    Z array.
    """
    out = tmp_path_factory.mktemp("nonuniform_z")
    ct_dir = out / "CT"
    rt_path = out / "RT.dcm"

    # Build a 12-slice CT with the existing synthetic helper (uniform Z),
    # then we'll override the IPP[2] of each slice after the fact.
    geometry = synthetic.Geometry(
        size=(64, 64, 12),
        origin=(0.0, 0.0, 0.0),
        spacing=(2.0, 2.0, 1.0),
    )
    sphere = synthetic.Sphere(
        name="Sphere_nonuniform",
        center=(30.0, 30.0, 5.5),
        radius=8.0,
    )
    uids = synthetic.CTSeriesUIDs()
    sop_uids = synthetic.build_synthetic_ct(
        ct_dir, geometry, uids, [sphere], modality="CT",
    )
    synthetic.build_synthetic_rtstruct(
        rt_path, geometry, uids, sop_uids, [sphere], image_modality="CT",
    )

    # Override each CT slice's ImagePositionPatient[2] (and SliceLocation)
    # to a mixed 3 mm / 6 mm pattern. The slices are written in
    # ``slice_0000.dcm``-order, which is also Z-ascending in the
    # synthetic helper.
    gaps = [3.0, 3.0, 3.0, 6.0, 6.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    expected_zs = [0.0]
    for g in gaps:
        expected_zs.append(expected_zs[-1] + g)
    assert len(expected_zs) == 12

    for f, z in zip(sorted(ct_dir.glob("slice_*.dcm")), expected_zs):
        ds = pydicom.dcmread(str(f))
        ipp = list(ds.ImagePositionPatient)
        ipp[2] = float(z)
        ds.ImagePositionPatient = ipp
        ds.SliceLocation = float(z)
        ds.save_as(str(f), enforce_file_format=True, little_endian=True, implicit_vr=False)

    return {
        "walk_root": out,
        "ct_dir": ct_dir,
        "rt_path": rt_path,
        "expected_zs": np.asarray(expected_zs, dtype=np.float64),
        "sphere_name": sphere.name,
    }


def test_per_slice_z_tracks_nonuniform_gaps(nonuniform_z_dataset):
    """Loading a CT with mixed 3/6 mm Z gaps should cache the per-DICOM
    IPP[2] values *exactly* -- including the 6 mm jumps. Failure here
    indicates the read path silently truncates / smooths the per-slice
    Z information (the underlying problem the fix exists to solve)."""
    r = DicomReaderWriter(
        description="nonuniform_z_check",
        Contour_Names=[nonuniform_z_dataset["sphere_name"]],
        arg_max=True,
        verbose=False,
    )
    r.walk_through_folders(str(nonuniform_z_dataset["walk_root"]), thread_count=1)
    r.set_index(r.indexes_with_contours[0])
    r.get_images_and_mask()

    assert r._dicom_slice_z_positions is not None
    z_arr = np.asarray(r._dicom_slice_z_positions)
    np.testing.assert_allclose(
        z_arr, nonuniform_z_dataset["expected_zs"], atol=1e-6,
        err_msg="cached per-slice Z array does not match the per-DICOM IPP[2]",
    )

    # The SimpleITK image's spacing[2] should NOT match any individual
    # gap -- ITK averages, so non-uniform input collapses to one number.
    # This is the precise reason the fix is needed.
    avg_spacing = r.dicom_handle.GetSpacing()[2]
    assert avg_spacing != pytest.approx(3.0), (
        "SimpleITK averaged spacing happened to equal 3 mm; the test "
        "fixture isn't exercising the non-uniform case."
    )
    assert avg_spacing != pytest.approx(6.0), (
        "SimpleITK averaged spacing happened to equal 6 mm; the test "
        "fixture isn't exercising the non-uniform case."
    )


# ---------------------------------------------------------------------------
# Layer 3 -- reshape_contour_data uses the array when available
# ---------------------------------------------------------------------------


def test_reshape_contour_data_uses_nearest_ipp_when_available(synthetic_dataset):
    """``reshape_contour_data`` should resolve each contour Z to the
    nearest cached per-slice Z (and ignore SimpleITK's uniform-spacing
    answer) whenever ``_dicom_slice_z_positions`` is populated.

    We inject a custom non-uniform per-slice array and a tiny contour
    whose physical Z lands closer to a slice the legacy round-via-
    spacing path would not pick. Verifies the array is *actually used*
    rather than only stored."""
    r = DicomReaderWriter(
        description="reshape_unit_test",
        Contour_Names=[p.name for p in synthetic_dataset.primitives],
        arg_max=True,
        verbose=False,
    )
    r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
    r.set_index(r.indexes_with_contours[0])
    r.get_images_and_mask()

    # Override the cached per-slice Z array with a synthetic mixed-gap
    # pattern. dicom_handle's averaged spacing stays whatever the
    # uniform synthetic dataset built.
    n_slices = r.dicom_handle.GetSize()[2]
    # Mixed 3 mm / 6 mm pattern, anchored at the synthetic origin so
    # X/Y indexing stays sensible.
    origin_z = r.dicom_handle.GetOrigin()[2]
    spacing_z = r.dicom_handle.GetSpacing()[2]
    gaps = [3.0 if i % 4 != 2 else 6.0 for i in range(n_slices - 1)]
    custom_zs = np.cumsum([0.0] + gaps) + origin_z
    r._dicom_slice_z_positions = custom_zs

    # Pick a Z position that lands between two slices on the custom
    # array. The nearest-IPP lookup should pick the closer of the two,
    # which is generally NOT what the uniform-spacing rounding picks.
    target_z = float(custom_zs[3] - 0.4)  # 0.4 mm below slice 3
    expected_idx = int(np.argmin(np.abs(custom_zs - target_z)))
    # x and y at the volume center (legal indices)
    cx = r.dicom_handle.GetOrigin()[0] + r.dicom_handle.GetSpacing()[0] * 8
    cy = r.dicom_handle.GetOrigin()[1] + r.dicom_handle.GetSpacing()[1] * 8
    contour = np.array([cx, cy, target_z], dtype=np.float64)
    out = r.reshape_contour_data(contour)
    assert out.shape == (1, 3)
    assert out[0, 2] == expected_idx, (
        f"reshape_contour_data did not use the per-slice array. "
        f"Expected slice index {expected_idx} (nearest to z={target_z}); "
        f"got {int(out[0, 2])}. custom_zs={custom_zs.tolist()}"
    )


def test_reshape_contour_data_falls_back_when_array_missing(synthetic_dataset):
    """When ``_dicom_slice_z_positions`` is None (e.g. IPP read failed),
    ``reshape_contour_data`` must fall back to SimpleITK's
    ``TransformPhysicalPointToIndex``. Without this fallback the fix
    would crash on legitimate but unusual inputs (anonymized series
    with stripped IPP tags, etc.)."""
    r = DicomReaderWriter(
        description="fallback_test",
        Contour_Names=[p.name for p in synthetic_dataset.primitives],
        arg_max=True,
        verbose=False,
    )
    r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
    r.set_index(r.indexes_with_contours[0])
    r.get_images_and_mask()

    # Force the per-slice array to None and re-run a contour through
    # the reshape path; output should match what
    # TransformPhysicalPointToIndex returns directly.
    r._dicom_slice_z_positions = None
    origin = r.dicom_handle.GetOrigin()
    spacing = r.dicom_handle.GetSpacing()
    target = np.array([
        origin[0] + spacing[0] * 8,
        origin[1] + spacing[1] * 8,
        origin[2] + spacing[2] * 5,
    ], dtype=np.float64)
    out = r.reshape_contour_data(target)
    expected = np.asarray(r.dicom_handle.TransformPhysicalPointToIndex(target))
    np.testing.assert_array_equal(out[0], expected)
