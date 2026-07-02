"""Robustness tests for partial or unconventional DICOM inputs.

Covers: extension-less DICOM files (PACS exports named by SOP UID),
``write_to_folder(rois=...)`` without walk-time ``Contour_Names``, and
orphan RT structures whose referenced image series is missing.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from DicomRTTool.ReaderWriter import DicomReaderWriter

_LOGGER = "DicomRTTool.ReaderWriter"


def _copy_extensionless(dataset, dest: Path) -> None:
    """Copy a synthetic dataset, stripping the ``.dcm`` suffix everywhere."""
    ct_dir = dest / "CT"
    ct_dir.mkdir(parents=True, exist_ok=True)
    for src in dataset.image_dir.glob("*.dcm"):
        shutil.copyfile(src, ct_dir / src.stem)  # "slice_0000.dcm" -> "slice_0000"
    shutil.copyfile(dataset.rt_path, dest / "RT")


class TestExtensionlessDicom:
    def test_walk_finds_extensionless_series_and_struct(
        self, synthetic_dataset, tmp_path: Path,
    ):
        """Regression: the walk skipped folders without ``*.dcm`` names, and
        the folder loader only considered ``*.dcm`` files as RT candidates,
        so extension-less exports lost their structs (or everything)."""
        _copy_extensionless(synthetic_dataset, tmp_path)
        r = DicomReaderWriter(
            Contour_Names=[p.name for p in synthetic_dataset.primitives],
            verbose=False,
        )
        r.walk_through_folders(str(tmp_path), thread_count=1)

        assert r.images_dictionary, "extensionless image series must be discovered"
        assert r.rt_dictionary, "extensionless RTSTRUCT must be discovered"
        assert r.indexes_with_contours


class TestRoisArgumentFilters:
    def test_rois_argument_filters_without_contour_names(
        self, synthetic_dataset, tmp_path: Path,
    ):
        """Regression: ``rois=`` reused ``indexes_with_contours`` computed
        against the (empty) walk-time ``Contour_Names``, exporting every
        series regardless of the requested ROIs."""
        r = DicomReaderWriter(verbose=False)  # no Contour_Names at walk time
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)

        # A requested ROI that exists -> the series exports with its mask.
        roi = synthetic_dataset.primitives[0].name.lower()
        out_hit = tmp_path / "hit"
        r.write_to_folder(str(out_hit), rois=[roi], thread_count=1)
        masks = list(out_hit.rglob("masks/*.nii.gz"))
        assert len(masks) == 1

        # A requested ROI that exists nowhere -> nothing is exported.
        out_miss = tmp_path / "miss"
        r.write_to_folder(str(out_miss), rois=["not_a_real_roi"], thread_count=1)
        assert not list(out_miss.rglob("*.nii.gz"))
        assert not (out_miss / "manifest.csv").exists()


class TestOrphanRt:
    def test_orphan_rt_warns_and_manifest_message_is_accurate(
        self, synthetic_dataset, tmp_path: Path, caplog,
    ):
        """An RT whose referenced image series is absent must be reported,
        and create_manifest must not claim no RT structures were found."""
        shutil.copyfile(synthetic_dataset.rt_path, tmp_path / "RT.dcm")
        r = DicomReaderWriter(verbose=False)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            r.walk_through_folders(str(tmp_path), thread_count=1)
        assert any(
            "references image series" in rec.getMessage() for rec in caplog.records
        )

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            r.create_manifest(str(tmp_path / "manifest.csv"))
        assert not (tmp_path / "manifest.csv").exists()
        assert any(
            "none reference an image series" in rec.getMessage()
            for rec in caplog.records
        )
