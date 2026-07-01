"""Rasterisation correctness: parallel path must match the serial path."""
from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter, _mask_threads_for


def _load(dataset, mask_thread_count: int) -> DicomReaderWriter:
    r = DicomReaderWriter(
        description="raster",
        Contour_Names=[p.name for p in dataset.primitives],
        arg_max=True,
        verbose=False,
        mask_thread_count=mask_thread_count,
    )
    r.walk_through_folders(str(dataset.walk_root), thread_count=1)
    r.set_index(r.indexes_with_contours[0])
    r.get_images_and_mask()
    return r


class TestParallelMatchesSerial:
    def test_combined_mask_identical(self, synthetic_dataset):
        serial = _load(synthetic_dataset, 1)
        parallel = _load(synthetic_dataset, 4)
        assert np.array_equal(serial.mask, parallel.mask)

    def test_per_roi_handles_identical(self, synthetic_dataset):
        serial = _load(synthetic_dataset, 1)
        parallel = _load(synthetic_dataset, 4)
        assert set(serial.mask_dictionary) == set(parallel.mask_dictionary)
        for name in serial.mask_dictionary:
            a = sitk.GetArrayFromImage(serial.mask_dictionary[name])
            b = sitk.GetArrayFromImage(parallel.mask_dictionary[name])
            assert np.array_equal(a, b), f"mask for {name} differs between serial/parallel"

    def test_volumes_identical(self, synthetic_dataset):
        serial = _load(synthetic_dataset, 1)
        parallel = _load(synthetic_dataset, 4)
        vs = serial.series_instances_dictionary[serial.index].additional_tags["Volumes"]
        vp = parallel.series_instances_dictionary[parallel.index].additional_tags["Volumes"]
        assert np.array_equal(vs, vp)


class TestMaskThreadSplit:
    @pytest.mark.parametrize(
        "workers,n_series,expected",
        [
            (9, 1, 4),    # single series -> use spare cores, capped at 4
            (9, 10, 1),   # more series than cores -> serial masks
            (8, 2, 4),    # 8 // 2 == 4
            (9, 3, 3),    # 9 // 3 == 3
            (9, 0, 1),    # guard against div-by-zero
            (1, 1, 1),    # no spare cores
        ],
    )
    def test_split(self, workers, n_series, expected):
        assert _mask_threads_for(workers, n_series) == expected
