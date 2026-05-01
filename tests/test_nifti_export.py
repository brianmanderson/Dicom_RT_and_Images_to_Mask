"""Tests for NIfTI export — single-series + parallel + CSV characterisation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter


@pytest.fixture
def reader_with_mask(synthetic_dataset) -> DicomReaderWriter:
    r = DicomReaderWriter(
        description="nifti-test",
        Contour_Names=[p.name for p in synthetic_dataset.primitives],
        arg_max=True,
        verbose=False,
    )
    r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
    r.set_index(r.indexes_with_contours[0])
    r.get_images_and_mask()
    return r


class TestWriteImagesAnnotations:
    def test_image_and_mask_files_are_written(
        self, reader_with_mask: DicomReaderWriter, tmp_path: Path,
    ):
        reader_with_mask.write_images_annotations(str(tmp_path))
        images = list(tmp_path.glob("Overall_Data_*.nii.gz"))
        masks = list(tmp_path.glob("Overall_mask_*.nii.gz"))
        assert len(images) == 1, f"expected one image file, got {images}"
        assert len(masks) == 1, f"expected one mask file, got {masks}"

    def test_round_trip_image_shape(
        self, reader_with_mask: DicomReaderWriter, tmp_path: Path,
    ):
        reader_with_mask.write_images_annotations(str(tmp_path))
        img_path = next(tmp_path.glob("Overall_Data_*.nii.gz"))
        re_read = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        assert re_read.shape == reader_with_mask.ArrayDicom.shape

    def test_round_trip_mask_shape_matches_image(
        self, reader_with_mask: DicomReaderWriter, tmp_path: Path,
    ):
        reader_with_mask.write_images_annotations(str(tmp_path))
        img_path = next(tmp_path.glob("Overall_Data_*.nii.gz"))
        mask_path = next(tmp_path.glob("Overall_mask_*.nii.gz"))
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        assert img.shape == mask.shape

    def test_mask_round_trip_preserves_label_values(
        self, reader_with_mask: DicomReaderWriter, tmp_path: Path,
    ):
        reader_with_mask.write_images_annotations(str(tmp_path))
        mask_path = next(tmp_path.glob("Overall_mask_*.nii.gz"))
        re_read = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        assert set(np.unique(re_read).tolist()) == set(
            np.unique(reader_with_mask.mask).tolist()
        )

    def test_writer_creates_output_dir(
        self, reader_with_mask: DicomReaderWriter, tmp_path: Path,
    ):
        target = tmp_path / "nested" / "deeper"
        reader_with_mask.write_images_annotations(str(target))
        assert target.is_dir()
        assert any(target.glob("Overall_Data_*.nii.gz"))


class TestWriteParallel:
    """Smoke test for the bulk-export entry point."""

    @pytest.mark.parametrize("thread_count", [1, 2])
    def test_write_parallel_produces_outputs(
        self,
        synthetic_dataset,
        tmp_path: Path,
        thread_count: int,
    ):
        r = DicomReaderWriter(
            description="parallel",
            Contour_Names=[p.name for p in synthetic_dataset.primitives],
            arg_max=True,
            verbose=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)

        out_dir = tmp_path / f"out_t{thread_count}"
        index_csv = out_dir / "MRN_Path_To_Iteration.csv"
        out_dir.mkdir()

        r.write_parallel(
            out_path=str(out_dir),
            index_file=str(index_csv),
            thread_count=thread_count,
        )

        assert index_csv.exists(), "bookkeeping CSV was not created"
        nii_images = list(out_dir.glob("Overall_Data_*.nii.gz"))
        nii_masks = list(out_dir.glob("Overall_mask_*.nii.gz"))
        assert nii_images, "no image NIfTIs were written"
        assert nii_masks, "no mask NIfTIs were written"
        assert len(nii_images) == len(nii_masks)


class TestCharacterizeDataToCSV:
    """Smoke test for the metadata-export helper."""

    def test_csv_files_written_and_loadable(
        self, reader_with_mask: DicomReaderWriter, tmp_path: Path,
    ):
        rois_csv = tmp_path / "summary.csv"
        reader_with_mask.characterize_data_to_csv(csv_path=str(rois_csv))

        # Two CSVs are produced: the ROIs table + an _images sibling.
        images_csv = tmp_path / "summary_images.csv"
        assert rois_csv.exists()
        assert images_csv.exists()

        df = pd.read_csv(rois_csv)
        df_images = pd.read_csv(images_csv)
        assert "PatientID" in df.columns
        assert "PatientID" in df_images.columns
