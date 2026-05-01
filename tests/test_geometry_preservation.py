"""Geometry round-trip tests on a synthetic dataset with non-trivial origin
and anisotropic voxel spacing.

This replaces the geometry-preservation coverage that lived in the
AnonDICOM-based ``test_all.py`` (Size / Spacing / Direction / Origin
preservation through the SimpleITK pipeline). With AnonDICOM gone we drive
the same checks against a synthetic dataset whose geometry is known
exactly — the assertions can therefore be tighter than the rounding-
tolerant ones the old tests used.

Coverage shape:

* ``TestImageHandleGeometry`` — the loaded ``dicom_handle`` reflects the
  spec exactly (no rounding, no axis swaps).
* ``TestAnnotationHandleGeometry`` — the produced ``annotation_handle``
  matches the image grid.
* ``TestNiftiPreservesGeometry`` — write_images_annotations + SimpleITK
  ReadImage round-trip preserves origin/spacing/direction.
* ``TestModalityFlip`` — same series with ``Modality="MR"`` indexes
  cleanly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter

# Tolerances chosen to be tight: ITK preserves these exactly for our
# synthetic geometry, but pydicom DS-VR formatting introduces a few digits
# of float noise so 1e-3 mm is a comfortable bound.
_FLOAT_ABS = 1e-3


def _loaded_reader(dataset) -> DicomReaderWriter:
    r = DicomReaderWriter(
        description="geom",
        Contour_Names=[p.name for p in dataset.primitives],
        arg_max=True,
        verbose=False,
    )
    r.walk_through_folders(str(dataset.walk_root), thread_count=1)
    r.set_index(r.indexes_with_contours[0])
    r.get_images_and_mask()
    return r


# ---------------------------------------------------------------------------
# Image handle geometry
# ---------------------------------------------------------------------------

class TestImageHandleGeometry:
    @pytest.fixture(scope="class")
    def reader(self, synthetic_dataset_anisotropic) -> DicomReaderWriter:
        return _loaded_reader(synthetic_dataset_anisotropic)

    def test_size(self, reader, synthetic_dataset_anisotropic):
        assert reader.dicom_handle.GetSize() == synthetic_dataset_anisotropic.geometry.size

    def test_spacing(self, reader, synthetic_dataset_anisotropic):
        for got, want in zip(
            reader.dicom_handle.GetSpacing(),
            synthetic_dataset_anisotropic.geometry.spacing,
            strict=True,
        ):
            assert abs(got - want) < _FLOAT_ABS

    def test_origin(self, reader, synthetic_dataset_anisotropic):
        for got, want in zip(
            reader.dicom_handle.GetOrigin(),
            synthetic_dataset_anisotropic.geometry.origin,
            strict=True,
        ):
            assert abs(got - want) < _FLOAT_ABS

    def test_direction_is_identity(self, reader):
        # Synthetic CT writer uses ImageOrientationPatient = [1,0,0, 0,1,0]
        # — i.e. axial / identity.
        identity = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        for got, want in zip(reader.dicom_handle.GetDirection(), identity, strict=True):
            assert abs(got - want) < _FLOAT_ABS


# ---------------------------------------------------------------------------
# Annotation handle geometry
# ---------------------------------------------------------------------------

class TestAnnotationHandleGeometry:
    """Mask handle should share image-handle geometry exactly so they
    overlay correctly in any viewer."""

    @pytest.fixture(scope="class")
    def reader(self, synthetic_dataset_anisotropic) -> DicomReaderWriter:
        return _loaded_reader(synthetic_dataset_anisotropic)

    def test_mask_handle_size_matches_image(self, reader):
        assert reader.annotation_handle.GetSize() == reader.dicom_handle.GetSize()

    def test_mask_handle_spacing_matches_image(self, reader):
        for got, want in zip(
            reader.annotation_handle.GetSpacing(),
            reader.dicom_handle.GetSpacing(),
            strict=True,
        ):
            assert abs(got - want) < _FLOAT_ABS

    def test_mask_handle_origin_matches_image(self, reader):
        for got, want in zip(
            reader.annotation_handle.GetOrigin(),
            reader.dicom_handle.GetOrigin(),
            strict=True,
        ):
            assert abs(got - want) < _FLOAT_ABS

    def test_mask_handle_direction_matches_image(self, reader):
        for got, want in zip(
            reader.annotation_handle.GetDirection(),
            reader.dicom_handle.GetDirection(),
            strict=True,
        ):
            assert abs(got - want) < _FLOAT_ABS


# ---------------------------------------------------------------------------
# NIfTI preserves geometry
# ---------------------------------------------------------------------------

class TestNiftiPreservesGeometry:
    @pytest.fixture
    def re_read_handles(
        self, synthetic_dataset_anisotropic, tmp_path: Path,
    ) -> tuple[sitk.Image, sitk.Image, sitk.Image]:
        reader = _loaded_reader(synthetic_dataset_anisotropic)
        reader.write_images_annotations(str(tmp_path))
        img = sitk.ReadImage(str(next(tmp_path.glob("Overall_Data_*.nii.gz"))))
        mask = sitk.ReadImage(str(next(tmp_path.glob("Overall_mask_*.nii.gz"))))
        return reader.dicom_handle, img, mask

    def test_image_size_round_trip(self, re_read_handles):
        original, re_read, _ = re_read_handles
        assert re_read.GetSize() == original.GetSize()

    def test_image_spacing_round_trip(self, re_read_handles):
        original, re_read, _ = re_read_handles
        for got, want in zip(re_read.GetSpacing(), original.GetSpacing(), strict=True):
            assert abs(got - want) < _FLOAT_ABS

    def test_image_origin_round_trip(self, re_read_handles):
        original, re_read, _ = re_read_handles
        for got, want in zip(re_read.GetOrigin(), original.GetOrigin(), strict=True):
            assert abs(got - want) < _FLOAT_ABS

    def test_mask_geometry_matches_image(self, re_read_handles):
        _, img, mask = re_read_handles
        assert mask.GetSize() == img.GetSize()
        for got, want in zip(mask.GetSpacing(), img.GetSpacing(), strict=True):
            assert abs(got - want) < _FLOAT_ABS
        for got, want in zip(mask.GetOrigin(), img.GetOrigin(), strict=True):
            assert abs(got - want) < _FLOAT_ABS


# ---------------------------------------------------------------------------
# MR modality flip
# ---------------------------------------------------------------------------

class TestModalityFlip:
    """Same code path as CT, but with ``Modality="MR"`` and the alternate
    SOP class UID. Replaces AnonDICOM's MR-007 / MR-009 coverage."""

    @pytest.fixture(scope="class")
    def reader(self, synthetic_dataset_mr) -> DicomReaderWriter:
        return _loaded_reader(synthetic_dataset_mr)

    def test_indexed_one_series(self, reader):
        assert len(reader.series_instances_dictionary) == 1

    def test_image_handle_built(self, reader):
        assert reader.dicom_handle is not None

    def test_annotation_handle_built(self, reader):
        assert reader.annotation_handle is not None

    def test_mask_shape_matches_image_shape(self, reader):
        assert reader.mask.shape == reader.ArrayDicom.shape
