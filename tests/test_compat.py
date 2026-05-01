"""Lock in the v4.0 breaking-change contract.

Asserts that:

* The deprecated v3 API names (``down_folder``, ``where_are_RTs``,
  ``with_annotations``, ``set_contour_names_and_assocations``,
  ``mask_to_contours``, plus the various ``__double_underscore__``
  back-compat aliases) are no longer present on ``DicomReaderWriter``.
* The new public reset API (``reset``, ``reset_mask``, ``reset_rts``)
  is present and behaves as documented.
* ``set_iteration`` / ``set_description`` (renamed from the old
  ``__set_iteration__`` / ``__set_description__`` dunders) are present.
"""
from __future__ import annotations

import pytest

from DicomRTTool.ReaderWriter import DicomReaderWriter

# Names that existed in v3 and were removed in v4.0.
REMOVED_NAMES = [
    # Deprecated method aliases.
    "down_folder",
    "where_are_RTs",
    "with_annotations",
    "set_contour_names_and_assocations",  # historical typo
    "mask_to_contours",
    # __dunder__ private-as-public aliases.
    "__mask_empty_mask__",
    "__reset_mask__",
    "__reset__",
    "__reset_RTs__",
    "__compile__",
    "__check_contours_at_index__",
    "__check_if_all_contours_present__",
    "__characterize_RT__",
    "__return_mask_for_roi__",
    "__set_iteration__",
    "__set_description__",
]


@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_removed_v3_name_is_gone(name: str):
    assert not hasattr(DicomReaderWriter, name), (
        f"{name!r} should have been removed in v4.0 but is still present"
    )


# ---------------------------------------------------------------------------
# New public reset surface
# ---------------------------------------------------------------------------

class TestResetAPI:
    """The new public reset methods replace the old ``__reset__`` dunders."""

    @pytest.fixture
    def loaded(self, synthetic_dataset) -> DicomReaderWriter:
        r = DicomReaderWriter(
            description="reset",
            Contour_Names=[p.name for p in synthetic_dataset.primitives],
            arg_max=True,
            verbose=False,
        )
        r.walk_through_folders(str(synthetic_dataset.walk_root), thread_count=1)
        r.set_index(r.indexes_with_contours[0])
        r.get_images_and_mask()
        return r

    def test_reset_clears_dictionaries(self, loaded: DicomReaderWriter):
        assert loaded.images_dictionary, "precondition: should have loaded images"
        loaded.reset()
        assert loaded.images_dictionary == {}
        assert loaded.rt_dictionary == {}
        assert loaded.series_instances_dictionary == {}

    def test_reset_rts_keeps_images(self, loaded: DicomReaderWriter):
        before_images = dict(loaded.images_dictionary)
        loaded.reset_rts()
        # Image dictionary preserved...
        assert loaded.images_dictionary == before_images
        # ...but ROI bookkeeping is wiped.
        assert loaded.all_rois == []
        assert loaded.indexes_with_contours == []

    def test_reset_mask_reallocates(self, loaded: DicomReaderWriter):
        # Touch the existing mask so we can confirm a fresh one was allocated.
        loaded.mask[:] = 7  # dirty marker
        loaded.reset_mask()
        # Fresh empty mask should be all zeros again.
        assert (loaded.mask == 0).all()
        assert loaded.mask_dictionary == {}


# ---------------------------------------------------------------------------
# Renamed setters
# ---------------------------------------------------------------------------

def test_set_iteration_and_set_description_exist():
    r = DicomReaderWriter(description="x", verbose=False)
    r.set_iteration(7)
    assert r.iteration == "7"
    r.set_description("renamed")
    assert r.description == "renamed"
