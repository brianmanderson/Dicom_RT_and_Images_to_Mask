"""Tests for the spacing-tuple resampler ``resample_to_spacing``."""
from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from DicomRTTool import resample_to_reference, resample_to_spacing


def _make_image(
    size=(20, 16, 10),          # (x, y, z)
    spacing=(1.0, 1.0, 2.0),
    origin=(3.0, -5.0, 7.0),
    binary=False,
) -> sitk.Image:
    # numpy array is (z, y, x).
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.float32)
    if binary:
        arr[2:6, 4:10, 5:15] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


class TestResampleSizeMath:
    def test_new_size_is_ceil_of_physical_extent(self):
        img = _make_image()
        out = resample_to_spacing(img, (2.0, 2.0, 2.0), "Linear")
        # x: ceil(20*1/2)=10, y: ceil(16*1/2)=8, z: ceil(10*2/2)=10
        assert out.GetSize() == (10, 8, 10)

    def test_upsample_size(self):
        img = _make_image(size=(10, 10, 5), spacing=(2.0, 2.0, 2.0))
        out = resample_to_spacing(img, (1.0, 1.0, 1.0))
        assert out.GetSize() == (20, 20, 10)

    def test_degenerate_axis_clamped_to_one(self):
        img = _make_image()
        out = resample_to_spacing(img, (1000.0, 1000.0, 1000.0))
        assert out.GetSize() == (1, 1, 1)

    def test_output_spacing_applied(self):
        img = _make_image()
        out = resample_to_spacing(img, (2.5, 2.5, 3.0))
        assert out.GetSpacing() == pytest.approx((2.5, 2.5, 3.0))


class TestResampleGeometryPreserved:
    def test_origin_and_direction_preserved(self):
        img = _make_image()
        out = resample_to_spacing(img, (2.0, 2.0, 2.0))
        assert out.GetOrigin() == pytest.approx(img.GetOrigin())
        assert out.GetDirection() == pytest.approx(img.GetDirection())


class TestInterpolatorBehaviour:
    def test_nearest_keeps_mask_binary(self):
        img = _make_image(binary=True)
        out = resample_to_spacing(img, (0.7, 0.7, 1.3), "Nearest")
        values = set(np.unique(sitk.GetArrayFromImage(out)).tolist())
        assert values.issubset({0.0, 1.0})

    def test_linear_may_blend(self):
        # Linear interpolation of a binary edge can produce intermediate values.
        img = _make_image(binary=True)
        out = resample_to_spacing(img, (0.5, 0.5, 0.9), "Linear")
        arr = sitk.GetArrayFromImage(out)
        assert arr.max() <= 1.0 + 1e-6

    def test_unknown_interpolator_raises(self):
        img = _make_image()
        with pytest.raises(ValueError):
            resample_to_spacing(img, (2.0, 2.0, 2.0), "Cubic")

    def test_bad_spacing_raises(self):
        img = _make_image()
        with pytest.raises(ValueError):
            resample_to_spacing(img, (2.0, -1.0, 2.0))
        with pytest.raises(ValueError):
            resample_to_spacing(img, (2.0, 2.0))  # type: ignore[arg-type]


class TestResampleToReference:
    def test_output_matches_reference_grid(self):
        moving = _make_image(size=(40, 40, 20), spacing=(0.5, 0.5, 1.0))
        reference = _make_image(size=(20, 16, 10), spacing=(1.0, 2.0, 3.0), origin=(1.0, 2.0, 3.0))
        out = resample_to_reference(moving, reference, "Linear")
        assert out.GetSize() == reference.GetSize()
        assert out.GetSpacing() == pytest.approx(reference.GetSpacing())
        assert out.GetOrigin() == pytest.approx(reference.GetOrigin())
        assert out.GetDirection() == pytest.approx(reference.GetDirection())

    def test_nearest_keeps_mask_binary(self):
        moving = _make_image(binary=True)
        reference = _make_image(size=(15, 12, 8), spacing=(1.3, 1.4, 1.5))
        out = resample_to_reference(moving, reference, "Nearest")
        values = set(np.unique(sitk.GetArrayFromImage(out)).tolist())
        assert values.issubset({0.0, 1.0})

    def test_unknown_interpolator_raises(self):
        img = _make_image()
        with pytest.raises(ValueError):
            resample_to_reference(img, img, "Cubic")
