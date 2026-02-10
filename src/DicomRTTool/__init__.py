"""DicomRTTool – DICOM image, RT structure, and dose file reader/writer.

Quick start::

    from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass

    reader = DicomReaderWriter(description='Example', arg_max=True)
    reader.walk_through_folders('/path/to/dicom')
    all_rois = reader.return_rois(print_rois=True)

See :mod:`DicomRTTool.ReaderWriter` for the full API.
"""
from __future__ import annotations

import SimpleITK as sitk  # noqa: F401 – re-export for backward compat

from .ReaderWriter import DicomReaderWriter, ROIAssociationClass  # noqa: F401
from .Viewer import plot_scroll_Image  # noqa: F401

__all__ = [
    "DicomReaderWriter",
    "ROIAssociationClass",
    "plot_scroll_Image",
    "sitk",
]
