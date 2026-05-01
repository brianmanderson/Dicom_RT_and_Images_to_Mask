"""Contour-extraction helpers used by the RT-structure writer.

Extracted from the original ``ReaderWriter.py`` god-class.

This module is not part of the public API. Import from
``DicomRTTool.ReaderWriter`` for the supported surface.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from skimage.measure import find_contours, label, regionprops


class PointOutputMaker:
    """Converts a binary annotation slice into physical-space contour points.

    The writer pipeline (``DicomReaderWriter._mask_to_contours``) walks each
    slice of a multi-class prediction, hands each binary slice to
    :meth:`make_output`, and collects the resulting per-slice physical
    polygons in ``contour_dict`` keyed by slice index.
    """

    def __init__(
        self,
        image_size_rows: int,
        image_size_cols: int,
        pixel_size: tuple[float, ...],
        contour_dict: dict[int, list[np.ndarray]],
    ) -> None:
        self.image_size_rows = image_size_rows
        self.image_size_cols = image_size_cols
        self.pixel_size = pixel_size
        self.contour_dict = contour_dict

    def make_output(
        self,
        annotation: np.ndarray,
        slice_index: int,
        dicom_handle: sitk.Image,
    ) -> None:
        self.contour_dict[slice_index] = []
        regions = regionprops(label(annotation))
        for region in regions:
            temp_image = np.zeros(
                (self.image_size_rows, self.image_size_cols), dtype=np.uint8
            )
            rows, cols = region.coords[:, 0], region.coords[:, 1]
            temp_image[rows, cols] = 1

            contours = find_contours(
                temp_image, level=0.5, fully_connected="low", positive_orientation="high"
            )
            for contour in contours:
                contour = np.squeeze(contour)
                # Remove co-linear points (same slope -> redundant).
                with np.errstate(divide="ignore"):
                    slope = (contour[1:, 1] - contour[:-1, 1]) / (
                        contour[1:, 0] - contour[:-1, 0]
                    )
                prev_slope = None
                out: list[list[float]] = []
                for idx in range(len(slope)):
                    if slope[idx] != prev_slope:
                        out.append(contour[idx].tolist())
                    prev_slope = slope[idx]
                # Convert index -> physical coordinates.
                physical = [
                    [float(c[1]), float(c[0]), float(slice_index)] for c in out
                ]
                physical_pts = np.array(
                    [dicom_handle.TransformContinuousIndexToPhysicalPoint(p) for p in physical]
                )
                self.contour_dict[slice_index].append(physical_pts)
