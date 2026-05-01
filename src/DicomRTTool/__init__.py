"""DicomRTTool - DICOM image, RT structure, and dose file reader/writer.

Quick start::

    from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass

    reader = DicomReaderWriter(description='Example', arg_max=True)
    reader.walk_through_folders('/path/to/dicom')
    all_rois = reader.return_rois(print_rois=True)

See :mod:`DicomRTTool.ReaderWriter` for the full API.
"""
from __future__ import annotations

import SimpleITK as sitk

from .ReaderWriter import DicomReaderWriter, ROIAssociationClass
from .Viewer import plot_scroll_Image

__all__ = [
    "DicomReaderWriter",
    "ROIAssociationClass",
    "plot_scroll_Image",
    "sitk",
]
