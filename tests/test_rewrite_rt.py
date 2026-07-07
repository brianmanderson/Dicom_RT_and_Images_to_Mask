"""Tests for ``DicomReaderWriter.rewrite_RT``.

Regression coverage: the method previously indexed ``self.associations`` as
the legacy ``{old: new}`` dict, so with the current
``list[ROIAssociationClass]`` API it silently renamed nothing (or raised
``TypeError`` with the default ``associations=None``).
"""
from __future__ import annotations

import shutil

import pydicom

from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass


def test_rewrite_rt_renames_via_associations(synthetic_dataset, tmp_path):
    rt_copy = tmp_path / "RT_copy.dcm"
    shutil.copyfile(synthetic_dataset.rt_path, rt_copy)

    reader = DicomReaderWriter(
        verbose=False,
        Contour_Names=["target"],
        associations=[ROIAssociationClass("target", ["sphere_r15"])],
    )
    reader.rewrite_RT(str(rt_copy))

    ds = pydicom.dcmread(str(rt_copy))
    names = [s.ROIName for s in ds.StructureSetROISequence]
    assert "target" in names
    assert "sphere_r15" not in names
    assert "organ at risk" in names  # non-associated ROIs untouched
    assert "target" in reader.rois_in_loaded_index


def test_rewrite_rt_without_associations_is_a_noop(synthetic_dataset, tmp_path):
    rt_copy = tmp_path / "RT_copy.dcm"
    shutil.copyfile(synthetic_dataset.rt_path, rt_copy)
    before = [
        s.ROIName
        for s in pydicom.dcmread(str(rt_copy)).StructureSetROISequence
    ]

    reader = DicomReaderWriter(verbose=False)
    reader.rewrite_RT(str(rt_copy))

    after = [
        s.ROIName
        for s in pydicom.dcmread(str(rt_copy)).StructureSetROISequence
    ]
    assert after == before
