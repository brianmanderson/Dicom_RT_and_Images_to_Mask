"""Tests for the RT Plan / dose linkage paths in ``_compile`` and ``get_dose``.

Covers the previously-untested compile steps: plan→struct grouping, dose
grouped via ``ReferencedRTPlanSequence`` (no struct reference), the
frame-of-reference fallback for entirely unreferenced doses, BEAM-only
``DoseSummationType`` filtering, and worker resilience to a corrupt series.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pydicom

from DicomRTTool.ReaderWriter import DicomReaderWriter
from tests.synthetic import (
    PRESET_DEFAULT,
    CTSeriesUIDs,
    build_synthetic_dataset,
    build_synthetic_dose,
    build_synthetic_multi_series,
    build_synthetic_plan,
)

_LOGGER = "DicomRTTool.ReaderWriter"


def _uids_from_ct(image_dir: Path) -> CTSeriesUIDs:
    """Reconstruct the shared UIDs from a written CT slice."""
    first = pydicom.dcmread(
        str(sorted(image_dir.glob("*.dcm"))[0]), stop_before_pixels=True,
    )
    return CTSeriesUIDs(
        study=first.StudyInstanceUID,
        series=first.SeriesInstanceUID,
        frame_of_reference=first.FrameOfReferenceUID,
    )


def _corpus(tmp_path: Path):
    """CT + RTSTRUCT plus the pieces needed to attach plans/doses."""
    image_dir, rt_path, geometry, primitives = build_synthetic_dataset(tmp_path)
    rt_sop = pydicom.dcmread(str(rt_path), stop_before_pixels=True).SOPInstanceUID
    return image_dir, rt_path, primitives, rt_sop, _uids_from_ct(image_dir)


def _reader(primitives, **kwargs) -> DicomReaderWriter:
    return DicomReaderWriter(
        Contour_Names=[p.name for p in primitives], verbose=False, **kwargs,
    )


class TestPlanLinkage:
    def test_plan_grouped_with_struct_and_in_grouped_metadata(self, tmp_path: Path):
        image_dir, rt_path, primitives, rt_sop, uids = _corpus(tmp_path)
        build_synthetic_plan(tmp_path / "RP.dcm", uids, rt_struct_sop_uid=rt_sop)

        r = _reader(primitives)
        r.walk_through_folders(str(tmp_path), thread_count=1)
        entry = r.series_instances_dictionary[r.indexes_with_contours[0]]
        assert entry.RPs, "plan must be grouped onto the image series via its struct"

        out = tmp_path / "out"
        r.write_to_folder(str(out), metadata_style="grouped", thread_count=1)
        meta = json.loads(next(out.rglob("metadata.json")).read_text())
        assert meta["plans"][0]["plan_label"] == "SYNTH_PLAN"
        assert meta["plans"][0]["plan_name"] == "Synthetic Plan"

    def test_dose_grouped_via_referenced_plan(self, tmp_path: Path):
        """Compile step: a dose with ONLY a plan reference (no struct ref)
        must reach the image series through plan -> struct -> image."""
        image_dir, rt_path, primitives, rt_sop, uids = _corpus(tmp_path)
        _, plan_sop = build_synthetic_plan(
            tmp_path / "RP.dcm", uids, rt_struct_sop_uid=rt_sop,
        )
        build_synthetic_dose(
            tmp_path / "RD.dcm", PRESET_DEFAULT, uids,
            rt_struct_sop_uid=None, rt_plan_sop_uid=plan_sop,
        )

        r = _reader(primitives, get_dose_output=True)
        r.walk_through_folders(str(tmp_path), thread_count=1)
        entry = r.series_instances_dictionary[r.indexes_with_contours[0]]
        assert entry.RDs, "dose must be grouped via its ReferencedRTPlanSequence"

        r.set_index(r.indexes_with_contours[0])
        r.get_images()
        r.get_dose()
        assert r.dose_handle is not None


class TestFrameOfReferenceFallback:
    def test_unreferenced_dose_groups_by_frame_of_reference(self, tmp_path: Path):
        image_dir, rt_path, primitives, rt_sop, uids = _corpus(tmp_path)
        build_synthetic_dose(
            tmp_path / "RD.dcm", PRESET_DEFAULT, uids,
            rt_struct_sop_uid=None, rt_plan_sop_uid=None,
        )

        r = _reader(primitives, get_dose_output=True)  # fallback on by default
        r.walk_through_folders(str(tmp_path), thread_count=1)
        entry = r.series_instances_dictionary[r.indexes_with_contours[0]]
        assert entry.RDs, "unreferenced dose must fall back to frame-of-reference"

    def test_unreferenced_dose_orphaned_when_fallback_disabled(
        self, tmp_path: Path, caplog,
    ):
        image_dir, rt_path, primitives, rt_sop, uids = _corpus(tmp_path)
        build_synthetic_dose(
            tmp_path / "RD.dcm", PRESET_DEFAULT, uids,
            rt_struct_sop_uid=None, rt_plan_sop_uid=None,
        )

        r = _reader(primitives, group_dose_by_frame_of_reference=False)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            r.walk_through_folders(str(tmp_path), thread_count=1)
        entry = r.series_instances_dictionary[r.indexes_with_contours[0]]
        assert not entry.RDs
        assert any(
            "could not be grouped" in rec.getMessage() for rec in caplog.records
        )


class TestBeamDoseFiltering:
    def test_beam_only_doses_warn_for_plan_filter_and_load_for_beam(
        self, tmp_path: Path, caplog,
    ):
        """Two BEAM doses + dose_type='PLAN' must warn and load nothing;
        dose_type='BEAM' must sum both."""
        image_dir, rt_path, primitives, rt_sop, uids = _corpus(tmp_path)
        for name in ("RD_beam1.dcm", "RD_beam2.dcm"):
            build_synthetic_dose(
                tmp_path / name, PRESET_DEFAULT, uids,
                rt_struct_sop_uid=rt_sop, summation_type="BEAM",
            )

        r = _reader(primitives, get_dose_output=True)
        r.walk_through_folders(str(tmp_path), thread_count=1)
        r.set_index(r.indexes_with_contours[0])
        r.get_images()
        entry = r.series_instances_dictionary[r.index]
        assert len(entry.RDs) == 2

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            r.get_dose(dose_type="PLAN")
        assert r.dose_handle is None
        assert any(
            "matched none of the available" in rec.getMessage()
            for rec in caplog.records
        )

        r.get_dose(dose_type="BEAM")
        assert r.dose_handle is not None
        assert r.dose.max() > 0.0


class TestWorkerResilience:
    def test_one_corrupt_series_does_not_lose_other_rows(
        self, tmp_path: Path, caplog,
    ):
        """A series that fails mid-export logs a warning; the other series'
        NIfTI output and manifest row still land."""
        raw = build_synthetic_multi_series(tmp_path, n_series=2)
        primitives = raw[0][3]

        r = _reader(primitives)
        r.walk_through_folders(str(tmp_path), thread_count=1)
        assert len(r.indexes_with_contours) == 2

        # Corrupt one slice of the first pair AFTER the walk, so the worker's
        # pixel read fails while the index still lists the series.
        victim = sorted(raw[0][0].glob("*.dcm"))[0]
        victim.write_bytes(b"\x00" * 64)

        out = tmp_path / "out"
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            r.write_to_folder(str(out), thread_count=1)

        df = pd.read_csv(out / "manifest.csv", dtype=str)
        assert len(df) == 1, "surviving series must still be exported"
        assert any("Failed on" in rec.getMessage() for rec in caplog.records)
        assert len(list(out.rglob("image.nii.gz"))) == 1
