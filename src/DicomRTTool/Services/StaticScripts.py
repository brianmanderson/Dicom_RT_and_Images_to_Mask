"""Geometry utilities for polygon-to-mask conversion.

Provides functions to convert RT Structure contour polygons into binary
mask arrays and to interpolate non-planar contour segments onto a voxel grid.
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def poly2mask(
    vertex_row_coords: np.ndarray,
    vertex_col_coords: np.ndarray,
    shape: Tuple[int, int],
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
    z_lo, z_hi = int(math.floor(z_value)), int(math.ceil(z_value))
    r_lo, r_hi = int(math.floor(r_value)), int(math.ceil(r_value))
    c_lo, c_hi = int(math.floor(c_value)), int(math.ceil(c_value))

    for z in (z_lo, z_hi):
        for r in (r_lo, r_hi):
            for c in (c_lo, c_hi):
                mask[z, r, c] = mask_value
