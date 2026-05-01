"""Tests for the DICOM folder indexer.

Covers:

* Multi-threaded ``walk_through_folders`` produces the same dictionary keys
  as the single-threaded path (regression test for the threading.Lock added
  in PR #2). Driven by the multi-series synthetic fixture so both paths
  have non-trivial work to do.
* ``struct_pydicom_string_keys`` are actually threaded through to the
  parsed ``RTBase.additional_tags`` (regression test for the latent bug
  fixed in PR #3).
* ``return_files_from_*`` happy paths.
* ``where_is_ROI`` matches case-insensitively and returns ``[]`` for
  unknown names without raising.
"""
from __future__ import annotations

import pytest
from pydicom.tag import Tag

from DicomRTTool.ReaderWriter import DicomReaderWriter

# ---------------------------------------------------------------------------
# Threading determinism (multi-series corpus)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indexed_single(synthetic_multi_series_dataset) -> DicomReaderWriter:
    reader = DicomReaderWriter(description="t1", arg_max=True, verbose=False)
    reader.walk_through_folders(
        str(synthetic_multi_series_dataset.walk_root), thread_count=1,
    )
    return reader


@pytest.fixture(scope="module")
def indexed_threaded(synthetic_multi_series_dataset) -> DicomReaderWriter:
    reader = DicomReaderWriter(description="t4", arg_max=True, verbose=False)
    reader.walk_through_folders(
        str(synthetic_multi_series_dataset.walk_root), thread_count=4,
    )
    return reader


class TestThreadingDeterminism:
    def test_image_series_uids_match(
        self, indexed_single: DicomReaderWriter, indexed_threaded: DicomReaderWriter,
    ):
        assert (
            set(indexed_single.images_dictionary.keys())
            == set(indexed_threaded.images_dictionary.keys())
        )

    def test_rt_uids_match(
        self, indexed_single: DicomReaderWriter, indexed_threaded: DicomReaderWriter,
    ):
        assert (
            set(indexed_single.rt_dictionary.keys())
            == set(indexed_threaded.rt_dictionary.keys())
        )

    def test_series_dict_lengths_match(
        self, indexed_single: DicomReaderWriter, indexed_threaded: DicomReaderWriter,
    ):
        assert (
            len(indexed_single.series_instances_dictionary)
            == len(indexed_threaded.series_instances_dictionary)
        )

    def test_rois_in_index_dict_matches(
        self, indexed_single: DicomReaderWriter, indexed_threaded: DicomReaderWriter,
    ):
        single = {
            i: set(rois) for i, rois in indexed_single.rois_in_index_dict.items()
        }
        threaded = {
            i: set(rois) for i, rois in indexed_threaded.rois_in_index_dict.items()
        }
        assert single == threaded

    def test_finds_two_series(self, indexed_threaded: DicomReaderWriter):
        # Multi-series fixture writes two pairs — both should index.
        assert len(indexed_threaded.series_instances_dictionary) == 2


# ---------------------------------------------------------------------------
# struct_pydicom_string_keys plumbing
# ---------------------------------------------------------------------------

class TestStructKeysPlumbing:
    """``struct_pydicom_string_keys`` was historically stored on the loader
    but never threaded through to ``_add_rt``. PR #3 wires it through; this
    test locks that contract in."""

    @pytest.fixture(scope="class")
    def reader_with_struct_keys(self, synthetic_dataset) -> DicomReaderWriter:
        reader = DicomReaderWriter(
            description="struct-keys",
            arg_max=True,
            verbose=False,
            # (0x0008, 0x1030) is StudyDescription. Our synthetic RTSTRUCTs
            # don't populate it, so use (0x0010, 0x0020) (PatientID) which
            # we always set.
            struct_pydicom_string_keys={"MyPatientID": Tag((0x0010, 0x0020))},
        )
        reader.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        return reader

    def test_at_least_one_rt_has_extra_tag(self, reader_with_struct_keys):
        rts_with_tag = [
            rt for rt in reader_with_struct_keys.rt_dictionary.values()
            if "MyPatientID" in rt.additional_tags
        ]
        assert rts_with_tag, (
            "struct_pydicom_string_keys is not being threaded through to "
            "_add_rt — RTBase.additional_tags is empty"
        )

    def test_extra_tag_value_matches_synthetic_patient_id(
        self, reader_with_struct_keys,
    ):
        for rt in reader_with_struct_keys.rt_dictionary.values():
            if "MyPatientID" in rt.additional_tags:
                assert rt.additional_tags["MyPatientID"] == "DICOMRTTOOL_TEST"
                return


# ---------------------------------------------------------------------------
# Query API surface
# ---------------------------------------------------------------------------

class TestQueryAPI:
    @pytest.fixture(scope="class")
    def reader(self, synthetic_dataset) -> DicomReaderWriter:
        r = DicomReaderWriter(description="query", arg_max=True, verbose=False)
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        return r

    def test_return_rois_returns_lowercase_list(self, reader):
        rois = reader.return_rois(print_rois=False)
        assert rois  # non-empty
        assert all(isinstance(r, str) for r in rois)
        assert all(r == r.lower() for r in rois), (
            "ROI names should be normalized to lowercase by the parser"
        )

    def test_return_rois_includes_special_chars(self, reader):
        # Locked-in coverage of names with brackets + spaces (replacing
        # AnonDICOM's "dose 1200[cgy]" coverage).
        rois = set(reader.return_rois(print_rois=False))
        assert "dose 1200[cgy]" in rois
        assert "organ at risk" in rois

    def test_return_files_from_index_includes_dcm_files(self, reader):
        index = next(iter(reader.series_instances_dictionary))
        files = reader.return_files_from_index(index)
        assert files, "should return at least one file path"
        assert any(str(f).lower().endswith(".dcm") for f in files)

    def test_return_files_from_uid_with_unknown_uid_returns_empty(self, reader):
        assert reader.return_files_from_UID("definitely-not-a-real-uid") == []

    def test_where_is_roi_known(self, reader, synthetic_dataset):
        first_name = synthetic_dataset.primitives[0].name.lower()
        paths = reader.where_is_ROI(first_name)
        assert paths
        assert all(isinstance(p, str) for p in paths)

    def test_where_is_roi_unknown_returns_empty(self, reader):
        assert reader.where_is_ROI("not_a_real_roi_xyz") == []

    def test_where_is_roi_is_case_insensitive(self, reader, synthetic_dataset):
        first_name = synthetic_dataset.primitives[0].name
        assert reader.where_is_ROI(first_name.upper()) == reader.where_is_ROI(first_name.lower())


# ---------------------------------------------------------------------------
# Walk on an empty directory
# ---------------------------------------------------------------------------

def test_walk_through_folders_empty_dir_is_noop(tmp_path):
    """An empty directory should produce no entries and no exceptions."""
    reader = DicomReaderWriter(description="empty", arg_max=True, verbose=False)
    reader.walk_through_folders(str(tmp_path), thread_count=1)
    assert reader.images_dictionary == {}
    assert reader.rt_dictionary == {}
    assert reader.series_instances_dictionary == {}
