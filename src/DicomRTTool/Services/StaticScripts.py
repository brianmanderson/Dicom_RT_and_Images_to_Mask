"""Geometry utilities for polygon-to-mask conversion.

Provides functions to convert RT Structure contour polygons into binary
mask arrays and to interpolate non-planar contour segments onto a voxel grid.
Also provides a spacing-tuple resampler used by the NIfTI export helpers.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import SimpleITK as sitk


def poly2mask(
    vertex_row_coords: np.ndarray,
    vertex_col_coords: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Convert polygon vertex coordinates to a filled boolean mask.

    Args:
        vertex_row_coords: Row (Y) image coordinates of the polygon vertices.
        vertex_col_coords: Column (X) image coordinates of the polygon vertices.
        shape: ``(rows, cols)`` dimensions of the output mask.

    Returns:
        A boolean ``np.ndarray`` of *shape* with ``True`` inside the polygon.
    """
    coords = np.stack([vertex_col_coords, vertex_row_coords], axis=1)
    coords = np.expand_dims(coords, 0).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, coords, 1)
    return mask.astype(bool)


def add_to_mask(
    mask: np.ndarray,
    z_value: float,
    r_value: float,
    c_value: float,
    mask_value: int = 1,
) -> None:
    """Mark all eight neighbouring voxels around a continuous (z, r, c) point.

    This ensures thin non-planar contour lines are adequately captured on
    the discrete voxel grid.

    Args:
        mask: 3-D ``int8`` array ``[slices, rows, cols]`` to modify in-place.
        z_value: Continuous slice index.
        r_value: Continuous row index.
        c_value: Continuous column index.
        mask_value: Value written into the mask (default ``1``).
    """
    z_lo, z_hi = math.floor(z_value), math.ceil(z_value)
    r_lo, r_hi = math.floor(r_value), math.ceil(r_value)
    c_lo, c_hi = math.floor(c_value), math.ceil(c_value)

    for z in (z_lo, z_hi):
        for r in (r_lo, r_hi):
            for c in (c_lo, c_hi):
                mask[z, r, c] = mask_value


# Interpolator string -> SimpleITK enum. ``"Linear"`` for images and dose,
# ``"Nearest"`` for label masks (so labels are never blended).
_INTERPOLATORS = {
    "Linear": sitk.sitkLinear,
    "Nearest": sitk.sitkNearestNeighbor,
    "NearestNeighbor": sitk.sitkNearestNeighbor,
    "BSpline": sitk.sitkBSpline,
}


def resample_to_spacing(
    handle: sitk.Image,
    output_spacing: tuple[float, float, float],
    interpolator: str = "Linear",
) -> sitk.Image:
    """Resample a SimpleITK image to a target voxel spacing.

    The output covers the same physical extent as the input: the new size
    along each axis is ``ceil(in_size * in_spacing / out_spacing)`` (clamped
    to a minimum of 1), while the origin and direction cosines are preserved.
    This mirrors the C# ``ResampleToSpacing`` used by the reference tool.

    Args:
        handle: Input image.
        output_spacing: Desired ``(x, y, z)`` spacing in mm.
        interpolator: ``"Linear"`` (images / dose) or ``"Nearest"`` (masks).
            Also accepts ``"NearestNeighbor"`` / ``"BSpline"``.

    Returns:
        A resampled :class:`SimpleITK.Image`.
    """
    if interpolator not in _INTERPOLATORS:
        raise ValueError(
            f"Unknown interpolator '{interpolator}'. "
            f"Expected one of {sorted(_INTERPOLATORS)}."
        )
    output_spacing = tuple(float(s) for s in output_spacing)
    if len(output_spacing) != 3 or any(s <= 0 for s in output_spacing):
        raise ValueError(f"output_spacing must be three positive numbers; got {output_spacing}")

    in_size = handle.GetSize()
    in_spacing = handle.GetSpacing()
    new_size = [
        max(1, math.ceil(in_size[i] * in_spacing[i] / output_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(handle.GetOrigin())
    resampler.SetOutputDirection(handle.GetDirection())
    resampler.SetInterpolator(_INTERPOLATORS[interpolator])
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(handle)
