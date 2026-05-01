"""End-to-end tests on a fully synthetic CT + RTSTRUCT dataset.

The data is built fresh in a tmp dir from analytical primitives (Sphere,
Box, Cylinder) — see ``tests/synthetic.py``. Because the ground truth is
analytically known, these tests can assert volume / shape correctness with
meaningful tolerances rather than just "did anything come out".

Coverage shape:

* ``TestSyntheticRead`` — DicomRTTool can walk the synthetic corpus, find
  every primitive's ROI, and build a mask whose shape matches the CT.
* ``TestSyntheticVolumeAccuracy`` — voxel-counted mask volumes match the
  analytical mm³ volumes to within a discretization tolerance.
* ``TestSyntheticNiftiRoundTrip`` — ``write_images_annotations`` produces
  NIfTI files that re-read losslessly.
* ``TestSyntheticRTRoundTrip`` — ``prediction_array_to_RT`` writes a new
  RTSTRUCT from the loaded mask, and re-reading that RTSTRUCT recovers the
  same ROI names and contour data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter

# Voxel-counted volumes vs analytical truth. The polygon-to-mask rasterizer
# (cv2.fillPoly inside StaticScripts.poly2mask) and the boundary-inclusive
# z-slice selection in our primitives both bias *upward* — boundary points
# count as "inside". On 2mm isotropic voxels with primitives of radius
# 10-15 mm this typically lands at +20-35% relative error. The test only
# needs to catch *gross* mistakes (e.g. a primitive that fails to rasterize
# at all) so we use a generous bracket: voxel volume must be within
# [0.6x, 1.6x] of analytical. The C# verification project at
# Dicom_RT_Images_Csharp owns the tighter analytical-vs-rasterized
# accuracy story; here we just round-trip.
_VOLUME_LOWER_FRAC = 0.6
_VOLUME_UPPER_FRAC = 1.6


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

class TestSyntheticRead:
    @pytest.fixture(scope="class")
    def reader(self, synthetic_dataset) -> DicomReaderWriter:
        roi_names = [p.name for p in synthetic_dataset.primitives]
        r = DicomReaderWriter(
            description="synth",
            Contour_Names=roi_names,
            arg_max=True,
            verbose=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        r.set_index(r.indexes_with_contours[0])
        r.get_images_and_mask()
        return r

    def test_walk_finds_one_indexed_series(self, reader: DicomReaderWriter):
        assert len(reader.series_instances_dictionary) == 1

    def test_all_primitives_appear_as_rois(
        self, reader: DicomReaderWriter, synthetic_dataset,
    ):
        found = set(reader.return_rois(print_rois=False))
        expected = {p.name for p in synthetic_dataset.primitives}
        assert expected <= found

    def test_image_array_has_expected_shape(
        self, reader: DicomReaderWriter, synthetic_dataset,
    ):
        cols, rows, slices = synthetic_dataset.geometry.size
        assert reader.ArrayDicom.shape == (slices, rows, cols)

    def test_mask_array_has_expected_shape(
        self, reader: DicomReaderWriter, synthetic_dataset,
    ):
        cols, rows, slices = synthetic_dataset.geometry.size
        # arg_max=True collapses the channel dimension — mask is 3D.
        assert reader.mask.shape == (slices, rows, cols)

    def test_mask_label_set(
        self, reader: DicomReaderWriter, synthetic_dataset,
    ):
        # Labels: 0 = background, 1..N = each primitive in order.
        n_prims = len(synthetic_dataset.primitives)
        labels = set(np.unique(reader.mask).tolist())
        # Background must be present; every primitive must contribute at
        # least one labeled voxel.
        assert 0 in labels
        for k in range(1, n_prims + 1):
            assert k in labels, f"label {k} missing — primitive may not have rasterized"


# ---------------------------------------------------------------------------
# Volume accuracy
# ---------------------------------------------------------------------------

class TestSyntheticVolumeAccuracy:
    """Voxel-counted mask volumes should match the analytical formulas."""

    @pytest.fixture(scope="class")
    def reader(self, synthetic_dataset) -> DicomReaderWriter:
        roi_names = [p.name for p in synthetic_dataset.primitives]
        r = DicomReaderWriter(
            description="vol",
            Contour_Names=roi_names,
            arg_max=True,
            verbose=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        r.set_index(r.indexes_with_contours[0])
        r.get_images_and_mask()
        return r

    @pytest.mark.parametrize("primitive_index", [0, 1, 2])
    def test_voxel_count_brackets_analytical_volume(
        self,
        reader: DicomReaderWriter,
        synthetic_dataset,
        primitive_index: int,
    ):
        primitive = synthetic_dataset.primitives[primitive_index]
        # Class labels are 1-indexed in arg_max output.
        label = primitive_index + 1
        n_voxels = int(np.sum(reader.mask == label))
        measured_mm3 = n_voxels * synthetic_dataset.geometry.voxel_volume_mm3
        analytical_mm3 = primitive.analytical_volume_mm3()
        ratio = measured_mm3 / analytical_mm3
        assert _VOLUME_LOWER_FRAC <= ratio <= _VOLUME_UPPER_FRAC, (
            f"{primitive.name}: voxel-count volume {measured_mm3:.1f} mm^3 "
            f"vs analytical {analytical_mm3:.1f} mm^3 "
            f"(ratio {ratio:.2f} outside [{_VOLUME_LOWER_FRAC}, {_VOLUME_UPPER_FRAC}])"
        )

    def test_volume_ordering_matches_analytical(
        self, reader: DicomReaderWriter, synthetic_dataset,
    ):
        """If the rasterizer mis-handles one primitive, ordering by volume
        usually breaks before any individual ratio leaves the bracket."""
        analytical = [p.analytical_volume_mm3() for p in synthetic_dataset.primitives]
        measured = [
            int(np.sum(reader.mask == (i + 1))) * synthetic_dataset.geometry.voxel_volume_mm3
            for i in range(len(synthetic_dataset.primitives))
        ]
        assert np.argsort(measured).tolist() == np.argsort(analytical).tolist()


# ---------------------------------------------------------------------------
# NIfTI round-trip
# ---------------------------------------------------------------------------

class TestSyntheticNiftiRoundTrip:
    """``write_images_annotations`` -> SimpleITK ReadImage round-trips losslessly."""

    @pytest.fixture
    def reader_and_outdir(self, synthetic_dataset, tmp_path: Path):
        roi_names = [p.name for p in synthetic_dataset.primitives]
        r = DicomReaderWriter(
            description="nifti_rt",
            Contour_Names=roi_names,
            arg_max=True,
            verbose=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        r.set_index(r.indexes_with_contours[0])
        r.get_images_and_mask()
        r.write_images_annotations(str(tmp_path))
        return r, tmp_path

    def test_files_written(self, reader_and_outdir):
        _, out = reader_and_outdir
        assert list(out.glob("Overall_Data_*.nii.gz"))
        assert list(out.glob("Overall_mask_*.nii.gz"))

    def test_image_round_trip_shape(self, reader_and_outdir):
        reader, out = reader_and_outdir
        img_path = next(out.glob("Overall_Data_*.nii.gz"))
        re_img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        assert re_img.shape == reader.ArrayDicom.shape

    def test_mask_round_trip_label_set_preserved(self, reader_and_outdir):
        reader, out = reader_and_outdir
        mask_path = next(out.glob("Overall_mask_*.nii.gz"))
        re_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        assert set(np.unique(re_mask).tolist()) == set(np.unique(reader.mask).tolist())


# ---------------------------------------------------------------------------
# RT struct round-trip
# ---------------------------------------------------------------------------

class TestSyntheticRTRoundTrip:
    """Mask -> RTSTRUCT -> re-read -> mask must preserve ROI names + non-empty
    contour data per ROI.
    """

    @pytest.fixture(scope="class")
    def round_tripped(self, synthetic_dataset, tmp_path_factory):
        roi_names = [p.name for p in synthetic_dataset.primitives]
        r = DicomReaderWriter(
            description="rt_rt",
            Contour_Names=roi_names,
            arg_max=True,
            verbose=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        r.set_index(r.indexes_with_contours[0])
        r.get_images_and_mask()

        # Build a one-hot prediction matching the loaded mask: shape
        # (z, rows, cols, n_classes+1) where channel 0 is background.
        n_classes = len(roi_names)
        pred = np.zeros((*r.mask.shape, n_classes + 1), dtype=np.float32)
        for c in range(n_classes + 1):
            pred[..., c] = (r.mask == c).astype(np.float32)

        out_dir = tmp_path_factory.mktemp("rt_round_trip")
        r.prediction_array_to_RT(
            prediction_array=pred,
            output_dir=str(out_dir),
            ROI_Names=roi_names,
        )
        return out_dir, roi_names

    def test_dcm_was_written(self, round_tripped):
        out_dir, _ = round_tripped
        files = list(out_dir.glob("RS_*.dcm"))
        assert len(files) == 1

    def test_round_tripped_rt_has_all_roi_names(self, round_tripped):
        out_dir, roi_names = round_tripped
        dcm = next(out_dir.glob("RS_*.dcm"))
        ds = pydicom.dcmread(str(dcm))
        names = {s.ROIName for s in ds.StructureSetROISequence}
        for n in roi_names:
            assert n in names

    def test_round_tripped_rt_has_contour_data_for_each_roi(self, round_tripped):
        out_dir, roi_names = round_tripped
        dcm = next(out_dir.glob("RS_*.dcm"))
        ds = pydicom.dcmread(str(dcm))
        roi_number_for_name = {s.ROIName: s.ROINumber for s in ds.StructureSetROISequence}
        for name in roi_names:
            roi_number = roi_number_for_name[name]
            contour_seqs = [
                seq for seq in ds.ROIContourSequence
                if seq.ReferencedROINumber == roi_number
            ]
            assert contour_seqs, f"no ROIContourSequence entry for {name}"
            (contour_seq,) = contour_seqs
            assert hasattr(contour_seq, "ContourSequence")
            assert len(contour_seq.ContourSequence) > 0
            for c in contour_seq.ContourSequence:
                assert len(c.ContourData) % 3 == 0
                assert len(c.ContourData) > 0

    def test_round_tripped_rt_modality_is_rtstruct(self, round_tripped):
        out_dir, _ = round_tripped
        ds = pydicom.dcmread(str(next(out_dir.glob("RS_*.dcm"))))
        assert ds.Modality == "RTSTRUCT"
