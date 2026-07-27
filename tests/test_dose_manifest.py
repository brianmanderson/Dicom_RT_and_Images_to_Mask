"""Tests for dose rows in the manifest and dose values in the metadata sidecar.

Covers ``dose_max_cgy`` (unit conversion, native-grid peak), ``dose_plan_name``
(referenced plan -> plan label -> dose description fallback), the ``RTDOSE``
rows ``create_manifest`` and ``write_to_folder`` emit, and the ``dose_max_cgy``
field in the grouped ``metadata.json``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter, hash_series
from tests.synthetic import (
    PRESET_DEFAULT,
    CTSeriesUIDs,
    build_synthetic_dataset,
    build_synthetic_dose,
    build_synthetic_plan,
)

_LOGGER = "DicomRTTool.ReaderWriter"

# The SeriesDescription ``build_synthetic_dose`` writes -- the plan-name
# fallback when no RT Plan accompanies the dose.
_DOSE_DESCRIPTION = "DicomRTTool synthetic dose"


def _uids_from_ct(image_dir: Path) -> CTSeriesUIDs:
    first = pydicom.dcmread(
        str(sorted(image_dir.glob("*.dcm"))[0]), stop_before_pixels=True,
    )
    return CTSeriesUIDs(
        study=first.StudyInstanceUID,
        series=first.SeriesInstanceUID,
        frame_of_reference=first.FrameOfReferenceUID,
    )


def _corpus(tmp_path: Path, **dose_kwargs):
    """CT + RTSTRUCT + one RT Dose, with the dose knobs exposed.

    Returns ``(primitives, dose_path, reader)`` with the reader already walked.
    """
    image_dir, rt_path, _geometry, primitives = build_synthetic_dataset(tmp_path)
    rt_sop = pydicom.dcmread(str(rt_path), stop_before_pixels=True).SOPInstanceUID
    uids = _uids_from_ct(image_dir)
    dose_path = build_synthetic_dose(
        tmp_path / "RD.dcm", PRESET_DEFAULT, uids,
        rt_struct_sop_uid=rt_sop, **dose_kwargs,
    )
    reader = DicomReaderWriter(
        Contour_Names=[p.name for p in primitives],
        get_dose_output=True,
        verbose=False,
    )
    reader.walk_through_folders(str(tmp_path), thread_count=1)
    return primitives, dose_path, reader


def _only_dose(reader: DicomReaderWriter):
    """The single ``RDBase`` the walk found."""
    assert len(reader.rd_dictionary) == 1
    return next(iter(reader.rd_dictionary.values()))


class TestDoseMaxCgy:
    """``dose_max_cgy`` reads the native grid and converts DoseUnits to cGy."""

    def test_gy_units_are_multiplied_by_one_hundred(self, tmp_path: Path):
        _, _, reader = _corpus(tmp_path, dose_units="GY", peak_dose=50.0)
        assert reader.dose_max_cgy(_only_dose(reader)) == pytest.approx(5000.0, abs=0.05)

    def test_cgy_units_pass_through_unscaled(self, tmp_path: Path):
        _, _, reader = _corpus(tmp_path, dose_units="CGY", peak_dose=5000.0)
        assert reader.dose_max_cgy(_only_dose(reader)) == pytest.approx(5000.0, abs=0.05)

    def test_relative_units_have_no_cgy_equivalent(self, tmp_path: Path, caplog):
        _, _, reader = _corpus(tmp_path, dose_units="RELATIVE", peak_dose=1.0)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert reader.dose_max_cgy(_only_dose(reader)) is None
        assert "RELATIVE" in caplog.text

    def test_peak_survives_an_export_that_resamples_the_dose(self, tmp_path: Path):
        """Dmax must come off the native grid.

        Exporting to a coarser grid interpolates the peak away -- here 5000 cGy
        drops to roughly 4970 in the written dose. The manifest must still
        report the real maximum, not the one left in the resampled volume.
        """
        _, _, reader = _corpus(tmp_path, dose_units="GY", peak_dose=50.0)
        out = tmp_path / "out"
        reader.write_to_folder(str(out), output_spacing=(2.5, 2.5, 7.0), thread_count=1)

        exported = sitk.ReadImage(str(next(out.rglob("doses/*.nii.gz"))))
        resampled_peak = float(sitk.GetArrayViewFromImage(exported).max()) * 100.0
        assert resampled_peak < 4999.0, "expected resampling to blunt the peak"

        df = pd.read_csv(out / "manifest.csv")
        reported = df[df["modality"] == "RTDOSE"].iloc[0][f"{_DOSE_DESCRIPTION} cGy"]
        assert reported == pytest.approx(5000.0, abs=0.05)


class TestDosePlanName:
    """The dose's manifest column label, and its fallback chain."""

    def test_prefers_plan_name_from_the_referenced_plan(self, tmp_path: Path):
        image_dir, rt_path, _geom, primitives = build_synthetic_dataset(tmp_path)
        rt_sop = pydicom.dcmread(str(rt_path), stop_before_pixels=True).SOPInstanceUID
        uids = _uids_from_ct(image_dir)
        _, plan_sop = build_synthetic_plan(
            tmp_path / "RP.dcm", uids, rt_struct_sop_uid=rt_sop,
        )
        build_synthetic_dose(
            tmp_path / "RD.dcm", PRESET_DEFAULT, uids,
            rt_struct_sop_uid=rt_sop, rt_plan_sop_uid=plan_sop,
        )
        reader = DicomReaderWriter(
            Contour_Names=[p.name for p in primitives], verbose=False,
        )
        reader.walk_through_folders(str(tmp_path), thread_count=1)
        assert reader.dose_plan_name(_only_dose(reader)) == "Synthetic Plan"

    def test_falls_back_to_plan_label_when_name_is_empty(self, tmp_path: Path):
        image_dir, rt_path, _geom, primitives = build_synthetic_dataset(tmp_path)
        rt_sop = pydicom.dcmread(str(rt_path), stop_before_pixels=True).SOPInstanceUID
        uids = _uids_from_ct(image_dir)
        _, plan_sop = build_synthetic_plan(
            tmp_path / "RP.dcm", uids, rt_struct_sop_uid=rt_sop, plan_name="",
        )
        build_synthetic_dose(
            tmp_path / "RD.dcm", PRESET_DEFAULT, uids,
            rt_struct_sop_uid=rt_sop, rt_plan_sop_uid=plan_sop,
        )
        reader = DicomReaderWriter(
            Contour_Names=[p.name for p in primitives], verbose=False,
        )
        reader.walk_through_folders(str(tmp_path), thread_count=1)
        assert reader.dose_plan_name(_only_dose(reader)) == "SYNTH_PLAN"

    def test_falls_back_to_series_description_without_a_plan(self, tmp_path: Path):
        """The common public-collection case: dose shipped, plan withheld."""
        _, _, reader = _corpus(tmp_path)
        assert not reader.rp_dictionary
        assert reader.dose_plan_name(_only_dose(reader)) == _DOSE_DESCRIPTION


class TestManifestDoseRows:
    """``create_manifest`` emits one row per dose series."""

    def test_dose_row_is_keyed_by_the_dose_series_own_identifiers(self, tmp_path: Path):
        _, dose_path, reader = _corpus(tmp_path, dose_units="GY", peak_dose=50.0)
        dose_ds = pydicom.dcmread(str(dose_path), stop_before_pixels=True)

        manifest_path = tmp_path / "manifest.csv"
        reader.create_manifest(str(manifest_path), anonymize=True, salt="unit-test")
        df = pd.read_csv(manifest_path)

        dose_rows = df[df["modality"] == "RTDOSE"]
        assert len(dose_rows) == 1
        row = dose_rows.iloc[0]
        assert row["series_hash"] == hash_series(dose_ds.SeriesInstanceUID, "unit-test")
        # ...and it is a different series from any image row.
        assert row["series_hash"] not in set(df[df["modality"] != "RTDOSE"]["series_hash"])

    def test_dose_row_carries_dmax_under_the_plan_name_column(self, tmp_path: Path):
        _, _, reader = _corpus(tmp_path, dose_units="GY", peak_dose=50.0)
        manifest_path = tmp_path / "manifest.csv"
        reader.create_manifest(str(manifest_path))
        df = pd.read_csv(manifest_path)

        column = f"{_DOSE_DESCRIPTION} cGy"
        assert column in df.columns
        dose_row = df[df["modality"] == "RTDOSE"].iloc[0]
        assert dose_row[column] == pytest.approx(5000.0, abs=0.05)

    def test_dose_and_image_rows_leave_each_others_columns_blank(self, tmp_path: Path):
        primitives, _, reader = _corpus(tmp_path)
        manifest_path = tmp_path / "manifest.csv"
        reader.create_manifest(str(manifest_path))
        df = pd.read_csv(manifest_path)

        roi_column = f"{primitives[0].name} cc"
        dose_column = f"{_DOSE_DESCRIPTION} cGy"
        dose_row = df[df["modality"] == "RTDOSE"].iloc[0]
        image_row = df[df["modality"] == "CT"].iloc[0]

        assert pd.isna(dose_row[roi_column]), "a dose row has no ROI volumes"
        assert pd.isna(image_row[dose_column]), "an image row has no plan dose"
        assert not pd.isna(image_row[roi_column])
        assert not pd.isna(dose_row[dose_column])

    def test_rerunning_upserts_the_dose_row_rather_than_duplicating_it(
        self, tmp_path: Path,
    ):
        _, _, reader = _corpus(tmp_path)
        manifest_path = tmp_path / "manifest.csv"
        reader.create_manifest(str(manifest_path))
        first = pd.read_csv(manifest_path)
        reader.create_manifest(str(manifest_path))
        second = pd.read_csv(manifest_path)

        assert len(second) == len(first)
        assert (second["modality"] == "RTDOSE").sum() == 1

    def test_manifest_is_written_for_a_dose_only_corpus(self, tmp_path: Path):
        """Dose rows stand on their own -- no ROIs required."""
        image_dir, _rt, _geom, _prims = build_synthetic_dataset(tmp_path)
        uids = _uids_from_ct(image_dir)
        build_synthetic_dose(tmp_path / "RD.dcm", PRESET_DEFAULT, uids)

        reader = DicomReaderWriter(verbose=False)
        reader.walk_through_folders(str(tmp_path), thread_count=1)
        manifest_path = tmp_path / "manifest.csv"
        reader.create_manifest(str(manifest_path), rois=[])

        df = pd.read_csv(manifest_path)
        assert (df["modality"] == "RTDOSE").sum() == 1


class TestWriteToFolderManifest:
    def test_export_manifest_includes_the_dose_row(self, tmp_path: Path):
        _, _, reader = _corpus(tmp_path, dose_units="GY", peak_dose=50.0)
        out = tmp_path / "out"
        reader.write_to_folder(str(out), thread_count=1)

        df = pd.read_csv(out / "manifest.csv")
        dose_rows = df[df["modality"] == "RTDOSE"]
        assert len(dose_rows) == 1
        assert dose_rows.iloc[0][f"{_DOSE_DESCRIPTION} cGy"] == pytest.approx(
            5000.0, abs=0.05,
        )

    def test_no_dose_rows_when_dose_is_not_exported(self, tmp_path: Path):
        """Without ``get_dose_output`` no dose lands on disk, so the export
        manifest -- which describes the export -- must not claim one."""
        image_dir, rt_path, _geom, primitives = build_synthetic_dataset(tmp_path)
        rt_sop = pydicom.dcmread(str(rt_path), stop_before_pixels=True).SOPInstanceUID
        uids = _uids_from_ct(image_dir)
        build_synthetic_dose(
            tmp_path / "RD.dcm", PRESET_DEFAULT, uids, rt_struct_sop_uid=rt_sop,
        )
        reader = DicomReaderWriter(
            Contour_Names=[p.name for p in primitives],
            get_dose_output=False,
            verbose=False,
        )
        reader.walk_through_folders(str(tmp_path), thread_count=1)
        out = tmp_path / "out"
        reader.write_to_folder(str(out), thread_count=1)

        df = pd.read_csv(out / "manifest.csv")
        assert "RTDOSE" not in set(df["modality"])


class TestMetadataSidecar:
    def test_grouped_metadata_carries_dose_max_and_plan_name(self, tmp_path: Path):
        _, _, reader = _corpus(tmp_path, dose_units="GY", peak_dose=50.0)
        out = tmp_path / "out"
        reader.write_to_folder(str(out), metadata_style="grouped", thread_count=1)

        meta = json.loads(next(out.rglob("metadata.json")).read_text())
        dose = meta["doses"][0]
        assert dose["plan_name"] == _DOSE_DESCRIPTION
        assert dose["dose_max_cgy"] == pytest.approx(5000.0, abs=0.05)
        assert dose["dose_units"] == "GY"

    def test_dose_max_is_omitted_for_relative_units(self, tmp_path: Path):
        _, _, reader = _corpus(tmp_path, dose_units="RELATIVE", peak_dose=1.0)
        out = tmp_path / "out"
        reader.write_to_folder(str(out), metadata_style="grouped", thread_count=1)

        meta = json.loads(next(out.rglob("metadata.json")).read_text())
        assert "dose_max_cgy" not in meta["doses"][0]
        assert meta["doses"][0]["plan_name"] == _DOSE_DESCRIPTION
