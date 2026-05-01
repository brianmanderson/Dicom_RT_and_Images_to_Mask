"""Tests for the prediction-array → RT-structure writer.

Build a multi-class prediction array, hand it to ``prediction_array_to_RT``,
re-read the resulting RT file with pydicom, and assert the structure is
well-formed and matches what we wrote.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest

from DicomRTTool.ReaderWriter import DicomReaderWriter


@pytest.fixture(scope="module")
def loaded_reader(synthetic_dataset) -> DicomReaderWriter:
    """Reader with images loaded for the synthetic CT."""
    reader = DicomReaderWriter(
        description="writer-test",
        Contour_Names=[p.name for p in synthetic_dataset.primitives],
        arg_max=True,
        verbose=False,
    )
    reader.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
    reader.set_index(reader.indexes_with_contours[0])
    reader.get_images_and_mask()
    return reader


def _build_prediction(image_shape: tuple[int, int, int], n_classes: int) -> np.ndarray:
    """Build a synthetic prediction array.

    Channel 0 is background. Subsequent channels each carry one rectangular
    blob in a known location so the writer has *something* to write.
    """
    pred = np.zeros((*image_shape, n_classes + 1), dtype=np.float32)
    z = image_shape[0] // 2
    rows = image_shape[1]
    cols = image_shape[2]
    pred[z, rows // 4 : rows // 4 + 20, cols // 4 : cols // 4 + 20, 1] = 1
    if n_classes >= 2:
        pred[z, 3 * rows // 4 - 20 : 3 * rows // 4, 3 * cols // 4 - 20 : 3 * cols // 4, 2] = 1
    fg = pred[..., 1:].sum(axis=-1)
    pred[..., 0] = (fg == 0).astype(np.float32)
    return pred


class TestPredictionArrayToRT:
    @pytest.fixture(scope="class")
    def written_rt(self, loaded_reader: DicomReaderWriter, tmp_path_factory) -> Path:
        out_dir = tmp_path_factory.mktemp("rt_out")
        image = loaded_reader.ArrayDicom
        pred = _build_prediction(image.shape, n_classes=2)
        loaded_reader.prediction_array_to_RT(
            prediction_array=pred,
            output_dir=str(out_dir),
            ROI_Names=["test_square_a", "test_square_b"],
        )
        return out_dir

    def test_dcm_file_was_written(self, written_rt: Path):
        dcms = list(written_rt.glob("RS_*.dcm"))
        assert len(dcms) == 1, f"expected exactly one RS_*.dcm, got {dcms}"

    def test_completed_marker_was_written(self, written_rt: Path):
        assert (written_rt / "Completed.txt").exists()

    def test_written_rt_is_readable_pydicom(self, written_rt: Path):
        dcm_path = next(written_rt.glob("RS_*.dcm"))
        ds = pydicom.dcmread(str(dcm_path))
        assert ds.Modality == "RTSTRUCT"

    def test_written_rt_has_expected_roi_names(self, written_rt: Path):
        dcm_path = next(written_rt.glob("RS_*.dcm"))
        ds = pydicom.dcmread(str(dcm_path))
        names = {s.ROIName for s in ds.StructureSetROISequence}
        assert "test_square_a" in names
        assert "test_square_b" in names

    def test_written_rt_has_contour_data(self, written_rt: Path):
        dcm_path = next(written_rt.glob("RS_*.dcm"))
        ds = pydicom.dcmread(str(dcm_path))
        new_roi_names = {"test_square_a", "test_square_b"}
        new_roi_numbers = {
            s.ROINumber
            for s in ds.StructureSetROISequence
            if s.ROIName in new_roi_names
        }
        new_contours = [
            seq for seq in ds.ROIContourSequence
            if seq.ReferencedROINumber in new_roi_numbers
        ]
        assert new_contours, "no ROIContourSequence entries for our written ROIs"
        for contour_seq in new_contours:
            assert hasattr(contour_seq, "ContourSequence")
            assert len(contour_seq.ContourSequence) > 0
            for c in contour_seq.ContourSequence:
                assert len(c.ContourData) % 3 == 0
                assert len(c.ContourData) > 0


class TestPredictionArrayToRTValidation:
    """Argument-validation early-returns should not raise; they log + return."""

    def test_wrong_channel_count_returns_without_writing(
        self, loaded_reader: DicomReaderWriter, tmp_path: Path,
    ):
        image = loaded_reader.ArrayDicom
        # 4 channels (bg + 3 classes) but we pass only 2 ROI names => the
        # writer's validation should reject this and return early.
        bad = np.zeros((*image.shape, 4), dtype=np.float32)
        loaded_reader.prediction_array_to_RT(
            prediction_array=bad,
            output_dir=str(tmp_path),
            ROI_Names=["a", "b"],
        )
        assert list(tmp_path.glob("RS_*.dcm")) == []

    def test_unset_index_returns_without_writing(self, synthetic_dataset, tmp_path: Path):
        r = DicomReaderWriter(description="bad", arg_max=True, verbose=False)
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        r.set_index(99999)  # nonexistent
        pred = np.zeros((1, 8, 8, 2), dtype=np.float32)
        r.prediction_array_to_RT(
            prediction_array=pred,
            output_dir=str(tmp_path),
            ROI_Names=["x"],
        )
        assert list(tmp_path.glob("RS_*.dcm")) == []
