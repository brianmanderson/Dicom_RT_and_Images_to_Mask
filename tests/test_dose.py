"""Tests for ``DicomReaderWriter.get_dose``."""
from __future__ import annotations

import pytest

from DicomRTTool.ReaderWriter import DicomReaderWriter


@pytest.fixture
def reader_no_dose(synthetic_dataset) -> DicomReaderWriter:
    """Reader walking a corpus with NO RT-Dose files."""
    r = DicomReaderWriter(description="dose-no", arg_max=True, verbose=False)
    r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
    r.set_index(next(iter(r.series_instances_dictionary)))
    r.get_images()
    return r


@pytest.fixture
def reader_with_dose(synthetic_dataset_with_dose) -> DicomReaderWriter:
    """Reader walking a corpus that includes an RT-Dose grid."""
    r = DicomReaderWriter(description="dose-yes", arg_max=True, verbose=False)
    r.walk_through_folders(
        str(synthetic_dataset_with_dose.walk_root), thread_count=1,
    )
    # Pick the index whose entry actually carries the linked dose.
    target = next(
        idx for idx, entry in r.series_instances_dictionary.items() if entry.RDs
    )
    r.set_index(target)
    r.get_images()
    return r


def test_get_dose_with_no_rds_is_a_noop(reader_no_dose: DicomReaderWriter):
    """If the index has no RD entries, ``get_dose`` returns silently and
    leaves the reader without a populated ``dose`` array.
    """
    entry = reader_no_dose.series_instances_dictionary[reader_no_dose.index]
    assert not entry.RDs
    reader_no_dose.get_dose()  # must not raise
    assert reader_no_dose.dose is None


def test_get_dose_happy_path_builds_resampled_handle(
    reader_with_dose: DicomReaderWriter,
):
    reader_with_dose.get_dose()
    assert reader_with_dose.dose is not None
    assert reader_with_dose.dose_handle is not None
    # The dose grid is resampled onto the image grid -> sizes match exactly.
    assert reader_with_dose.dose_handle.GetSize() == reader_with_dose.dicom_handle.GetSize()


def test_get_dose_handle_is_non_zero(reader_with_dose: DicomReaderWriter):
    """Synthetic dose is a centred Gaussian — the centre voxel must be > 0."""
    reader_with_dose.get_dose()
    arr = reader_with_dose.dose
    assert arr is not None
    assert arr.max() > 0.0
