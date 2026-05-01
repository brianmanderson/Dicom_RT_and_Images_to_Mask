"""DICOM folder walking and per-series classification.

Extracted from the original ``ReaderWriter.py`` god-class. Provides:

* :class:`DicomFolderLoader` — walks a single directory, classifies each
  DICOM file as image / RT struct / RT dose / RT plan, and populates the
  caller's shared dictionaries. Thread-safe (uses an internal
  ``threading.Lock``) so multiple instances or repeated ``load()`` calls
  can run inside a ``ThreadPoolExecutor``.
* :func:`add_image`, :func:`add_rt`, :func:`add_rd`, :func:`add_rp`,
  :func:`add_sops` — the per-record dictionary helpers used by both the
  loader and the higher-level :class:`DicomReaderWriter` façade.

These names are not part of the public API. Import from
``DicomRTTool.ReaderWriter`` for the supported surface.
"""
from __future__ import annotations

import logging
import os
import threading

import pydicom
import SimpleITK as sitk

from ..Services.DicomBases import (
    ImageBase,
    PlanBase,
    PyDicomKeys,
    RDBase,
    RTBase,
    SitkDicomKeys,
    dcmread,
)
from . import NULL_CTX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Folder loading helper
# ---------------------------------------------------------------------------
class DicomFolderLoader:
    """Loads DICOM files from a single directory into shared dictionaries.

    A single :class:`threading.Lock` guards every mutation of the shared
    ``images_dict`` / ``rt_dict`` / ``rd_dict`` / ``rp_dict`` so that
    multiple worker threads can safely call :meth:`load` concurrently.
    """

    def __init__(
        self,
        plan_keys: PyDicomKeys | None,
        struct_keys: PyDicomKeys | None,
        image_keys: SitkDicomKeys | None,
        dose_keys: SitkDicomKeys | None,
    ) -> None:
        self.plan_keys = plan_keys
        self.struct_keys = struct_keys
        self.image_keys = image_keys
        self.dose_keys = dose_keys
        # Guards every mutation of the shared *_dict arguments to load().
        # CPython's GIL makes individual dict ops atomic, but read-modify-write
        # patterns (e.g. add_rd's "create or call .add_beam()") need an
        # explicit lock to be race-free across threads.
        self._lock = threading.Lock()

    def load(
        self,
        dicom_path: str,
        images_dict: dict[str, ImageBase],
        rt_dict: dict[str, RTBase],
        rd_dict: dict[str, RDBase],
        rp_dict: dict[str, PlanBase],
        verbose: bool = False,
    ) -> None:
        if verbose:
            logger.info("Loading from %s", dicom_path)

        reader = sitk.ImageSeriesReader()
        reader.GlobalWarningDisplayOff()
        img_reader = sitk.ImageFileReader()
        img_reader.LoadPrivateTagsOn()

        series_ids = reader.GetGDCMSeriesIDs(dicom_path)
        file_list = [f for f in os.listdir(dicom_path) if f.lower().endswith(".dcm")]
        all_series_names: list[str] = []

        for series_id in series_ids:
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path, series_id)
            all_series_names.extend(os.path.basename(n) for n in dicom_names)

            img_reader.SetFileName(dicom_names[0])
            img_reader.ReadImageInformation()
            modality = img_reader.GetMetaData("0008|0060").lower()

            if "rtdose" in modality:
                for name in dicom_names:
                    img_reader.SetFileName(name)
                    img_reader.Execute()
                    add_rd(img_reader, rd_dict, self.dose_keys, self._lock)
            else:
                img_reader.Execute()
                add_image(
                    images_dict, dicom_names, img_reader, dicom_path,
                    self.image_keys, self._lock,
                )

        # Remaining files (RT structs, plans, etc.).
        rt_files = [f for f in file_list if f not in all_series_names]
        for file_name in rt_files:
            full_path = os.path.join(dicom_path, file_name)
            try:
                ds = dcmread(full_path)
            except (pydicom.errors.InvalidDicomError, OSError, AttributeError):
                logger.warning("Could not read %s", full_path)
                continue
            modality = ds.Modality.lower()
            if "struct" in modality:
                add_rt(ds, full_path, rt_dict, self.struct_keys, self._lock)
            elif "plan" in modality:
                add_rp(ds, full_path, rp_dict, self.plan_keys, self._lock)


# ---------------------------------------------------------------------------
# Dictionary population helpers
# ---------------------------------------------------------------------------
def add_image(
    images_dict: dict[str, ImageBase],
    dicom_names: list[str],
    reader: sitk.ImageFileReader,
    path: str,
    sitk_string_keys: SitkDicomKeys | None = None,
    lock: threading.Lock | None = None,
) -> None:
    uid = reader.GetMetaData("0020|000e")
    # Read-then-write requires the lock to avoid a TOCTOU race across threads.
    with (lock or NULL_CTX):
        if uid not in images_dict:
            img = ImageBase()
            img.load_info(dicom_names, reader, path, sitk_string_keys)
            images_dict[uid] = img


def add_rt(
    ds: pydicom.Dataset,
    path: str,
    rt_dict: dict[str, RTBase],
    pydicom_string_keys: PyDicomKeys | None = None,
    lock: threading.Lock | None = None,
) -> None:
    try:
        uid = ds.SeriesInstanceUID
        with (lock or NULL_CTX):
            if uid not in rt_dict:
                rt = RTBase()
                rt.load_info(ds, path, pydicom_string_keys)
                rt_dict[uid] = rt
    except (AttributeError, KeyError, IndexError, ValueError) as exc:
        logger.warning("Error loading RT from %s: %s", path, exc, exc_info=True)


def add_rd(
    reader: sitk.ImageFileReader,
    rd_dict: dict[str, RDBase],
    sitk_string_keys: SitkDicomKeys | None = None,
    lock: threading.Lock | None = None,
) -> None:
    try:
        uid = reader.GetMetaData("0020|000e")
        with (lock or NULL_CTX):
            if uid not in rd_dict:
                rd = RDBase()
                rd.load_info(reader, sitk_string_keys)
                rd_dict[uid] = rd
            else:
                rd_dict[uid].add_beam(reader)
    except (AttributeError, KeyError, IndexError, RuntimeError, OSError) as exc:
        logger.warning("Error loading RD from %s: %s", reader.GetFileName(), exc, exc_info=True)


def add_rp(
    ds: pydicom.Dataset,
    path: str,
    rp_dict: dict[str, PlanBase],
    pydicom_string_keys: PyDicomKeys | None = None,
    lock: threading.Lock | None = None,
) -> None:
    try:
        uid = ds.SeriesInstanceUID
        with (lock or NULL_CTX):
            if uid not in rp_dict:
                plan = PlanBase()
                plan.load_info(ds, path, pydicom_string_keys)
                rp_dict[uid] = plan
    except (AttributeError, KeyError, IndexError, ValueError) as exc:
        logger.warning("Error loading RP from %s: %s", path, exc, exc_info=True)


def add_sops(
    reader: sitk.ImageSeriesReader,
    series_dict: dict[int, ImageBase],
) -> None:
    """Populate SOPInstanceUIDs for the series that *reader* just loaded."""
    uid = reader.GetMetaData(0, "0020|000e")
    for entry in series_dict.values():
        if entry.SeriesInstanceUID == uid:
            entry.SOPs = [
                reader.GetMetaData(i, "0008|0018")
                for i in range(len(reader.GetFileNames()))
            ]
            return
