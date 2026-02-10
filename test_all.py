"""Test suite for DicomRTTool.

Tests verify that masks and images produced by the reader match
pre-computed reference NIfTI files shipped with the test data.

Requires the ``AnonDICOM.zip`` test archive to be present in the
repository root (or a parent directory).
"""
from __future__ import annotations

import os
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk

from src.DicomRTTool.ReaderWriter import DicomReaderWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_base_dir() -> str:
    """Locate the directory containing AnonDICOM.zip."""
    base = "."
    for _ in range(4):
        if "AnonDICOM.zip" in os.listdir(base):
            return base
        base = os.path.join(base, "..")
    raise FileNotFoundError("Cannot find AnonDICOM.zip in any parent directory")


# ---------------------------------------------------------------------------
# Module-level setup: unzip test data once
# ---------------------------------------------------------------------------

_BASE = _find_base_dir()
_ANON_DIR = os.path.join(_BASE, "AnonDICOM")
if not os.path.exists(_ANON_DIR):
    with zipfile.ZipFile(os.path.join(_BASE, "AnonDICOM.zip"), "r") as zf:
        zf.extractall(_BASE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def path() -> str:
    return _ANON_DIR


@pytest.fixture
def base_mask(path: str) -> sitk.Image:
    return sitk.ReadImage(os.path.join(path, "Mask.nii.gz"))


@pytest.fixture
def base_mask007(path: str) -> sitk.Image:
    return sitk.ReadImage(os.path.join(path, "Mask_007.nii.gz"))


@pytest.fixture
def base_mask009(path: str) -> sitk.Image:
    return sitk.ReadImage(os.path.join(path, "Mask_009.nii.gz"))


@pytest.fixture
def base_image(path: str) -> sitk.Image:
    return sitk.ReadImage(os.path.join(path, "Image.nii.gz"))


@pytest.fixture
def main_reader(path: str) -> DicomReaderWriter:
    """Reader with spinalcord + body contours loaded."""
    reader = DicomReaderWriter(
        description="Examples",
        Contour_Names=["spinalcord", "body"],
        arg_max=True,
        verbose=True,
    )
    reader.walk_through_folders(path, thread_count=1)
    reader.set_index(reader.indexes_with_contours[0])
    reader.get_mask()
    return reader


@pytest.fixture
def main_reader007(main_reader: DicomReaderWriter) -> DicomReaderWriter:
    """Reader reconfigured for brainstem + dose contours (first index)."""
    main_reader.set_contour_names_and_associations(
        contour_names=["brainstem", "dose 1200[cgy]", "dose 500[cgy]"]
    )
    main_reader.set_index(main_reader.indexes_with_contours[0])
    main_reader.get_images_and_mask()
    return main_reader


@pytest.fixture
def main_reader009(main_reader007: DicomReaderWriter) -> DicomReaderWriter:
    """Reader switched to the second index with the same contour names."""
    main_reader007.set_index(main_reader007.indexes_with_contours[1])
    main_reader007.get_images_and_mask()
    return main_reader007


# ---------------------------------------------------------------------------
# Helper for floating-point comparison
# ---------------------------------------------------------------------------

def _round_tuple(t, decimals=6):
    """Round each float in a tuple for approximate comparison."""
    return tuple(round(x, decimals) if isinstance(x, float) else x for x in t)


# ---------------------------------------------------------------------------
# CT mask tests
# ---------------------------------------------------------------------------

class TestMaskCT:
    """Verify mask generation against the CT reference."""

    @pytest.fixture(autouse=True)
    def _setup(self, main_reader: DicomReaderWriter, base_mask: sitk.Image):
        self.reader = main_reader
        self.expected = base_mask

    def test_annotation_handle_exists(self):
        assert self.reader.annotation_handle is not None

    def test_size(self):
        assert self.expected.GetSize() == self.reader.annotation_handle.GetSize()

    def test_spacing(self):
        assert self.expected.GetSpacing() == self.reader.annotation_handle.GetSpacing()

    def test_direction(self):
        assert self.expected.GetDirection() == self.reader.annotation_handle.GetDirection()

    def test_origin(self):
        assert self.expected.GetOrigin() == self.reader.annotation_handle.GetOrigin()

    def test_array_values(self):
        actual = sitk.GetArrayFromImage(self.reader.annotation_handle)
        expected = sitk.GetArrayFromImage(self.expected)
        agreement = np.mean(actual == expected)
        assert agreement >= 0.999, (
            f"Only {agreement:.2%} of voxels match (need >99.9%)"
        )


# ---------------------------------------------------------------------------
# MR mask tests (index 007)
# ---------------------------------------------------------------------------

class TestMaskMR007:
    """Verify mask generation against the MR-007 reference."""

    @pytest.fixture(autouse=True)
    def _setup(self, main_reader007: DicomReaderWriter, base_mask007: sitk.Image):
        self.reader = main_reader007
        self.expected = base_mask007

    def test_annotation_handle_exists(self):
        assert self.reader.annotation_handle is not None

    def test_size(self):
        assert self.expected.GetSize() == self.reader.annotation_handle.GetSize()

    def test_spacing(self):
        assert (
            _round_tuple(self.expected.GetSpacing())
            == self.reader.annotation_handle.GetSpacing()
        )

    def test_direction(self):
        assert self.expected.GetDirection() == self.reader.annotation_handle.GetDirection()

    def test_origin(self):
        assert (
            _round_tuple(self.expected.GetOrigin(), 3)
            == self.reader.annotation_handle.GetOrigin()
        )

    def test_array_values(self):
        actual = sitk.GetArrayFromImage(self.reader.annotation_handle)
        expected = sitk.GetArrayFromImage(self.expected)
        agreement = np.mean(actual == expected)
        assert agreement >= 0.999, (
            f"Only {agreement:.2%} of voxels match (need >99.9%)"
        )


# ---------------------------------------------------------------------------
# MR mask tests (index 009)
# ---------------------------------------------------------------------------

class TestMaskMR009:
    """Verify mask generation against the MR-009 reference."""

    @pytest.fixture(autouse=True)
    def _setup(self, main_reader009: DicomReaderWriter, base_mask009: sitk.Image):
        self.reader = main_reader009
        self.expected = base_mask009

    def test_annotation_handle_exists(self):
        assert self.reader.annotation_handle is not None

    def test_size(self):
        assert self.expected.GetSize() == self.reader.annotation_handle.GetSize()

    def test_spacing(self):
        assert (
            _round_tuple(self.expected.GetSpacing())
            == self.reader.annotation_handle.GetSpacing()
        )

    def test_direction(self):
        assert self.expected.GetDirection() == self.reader.annotation_handle.GetDirection()

    def test_origin(self):
        assert (
            _round_tuple(self.expected.GetOrigin(), 3)
            == self.reader.annotation_handle.GetOrigin()
        )

    def test_array_values(self):
        actual = sitk.GetArrayFromImage(self.reader.annotation_handle)
        expected = sitk.GetArrayFromImage(self.expected)
        agreement = np.mean(actual == expected)
        assert agreement >= 0.999, (
            f"Only {agreement:.2%} of voxels match (need >99.9%)"
        )
