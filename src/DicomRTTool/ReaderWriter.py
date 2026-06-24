"""Core reader/writer for DICOM images, RT structures, and dose files.

This module provides :class:`DicomReaderWriter`, the primary public API for:
* Walking a folder tree to discover and catalogue DICOM series.
* Loading images, RT structure masks, and dose grids as NumPy arrays
  and SimpleITK image handles.
* Converting NumPy prediction masks back into DICOM RT Structure files.

Typical usage::

    from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass

    reader = DicomReaderWriter(description="Example", arg_max=True)
    reader.walk_through_folders("/path/to/dicom")
    reader.set_contour_names_and_associations(
        contour_names=["tumor"],
        associations=[ROIAssociationClass("tumor", ["tumor_mr", "tumor_ct"])],
    )
    reader.get_images_and_mask()
    image = reader.ArrayDicom
    mask  = reader.mask
"""
from __future__ import annotations

import copy
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from NiftiResampler.ResampleTools import ImageResampler
from pydicom.tag import Tag
from tqdm import tqdm

from ._internal import DEFAULT_WORKERS as _DEFAULT_WORKERS
from ._internal.anonymizer import (
    DEFAULT_SALT,
    AnonymizationKey,
    hash_patient,
    hash_series,
    hash_study,
)
from ._internal.indexer import (
    DicomFolderLoader,
    add_sops,
)
from ._internal.rt_contours import PointOutputMaker
from .Services.DicomBases import (
    ImageBase,
    PathLike,
    PlanBase,
    PyDicomKeys,
    RDBase,
    ROIClass,
    RTBase,
    SitkDicomKeys,
    dcmread,
)
from .Services.StaticScripts import add_to_mask, poly2mask, resample_to_spacing
from .Viewer import plot_scroll_Image  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility.
dcmread_function = dcmread

# Filename sanitisation for the per-ROI export layout (Windows-safe).
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_FILENAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _sanitize_filename(name: str) -> str:
    """Return *name* stripped of characters illegal in Windows filenames."""
    cleaned = _INVALID_FILENAME_CHARS.sub("_", name or "").strip().rstrip(". ")
    if cleaned.upper() in _RESERVED_FILENAMES:
        cleaned = f"_{cleaned}"
    return cleaned


# Cap per-ROI rasterisation threads: past ~4 the GIL-bound parts dominate and
# returns diminish (8 threads measured slower than 4).
_MAX_MASK_THREADS = 4


def _mask_threads_for(total_workers: int, n_series: int) -> int:
    """Split a thread budget between series-level and per-ROI parallelism.

    When many series are processed at once, series-level parallelism already
    saturates the cores, so each series rasterises serially (1). When only a
    few series are processed (e.g. a single multi-ROI series), the leftover
    cores are handed to per-ROI rasterisation inside ``get_mask``.
    """
    if n_series <= 0:
        return 1
    return max(1, min(_MAX_MASK_THREADS, total_workers // n_series))


# ---------------------------------------------------------------------------
# ROI association helper (public API — kept here intentionally)
# ---------------------------------------------------------------------------
class ROIAssociationClass:
    """Maps a canonical ROI name to one or more alternative names.

    Args:
        roi_name: The canonical (target) ROI name.
        other_names: Alternative names that should be treated as equivalent.
    """

    def __init__(self, roi_name: str, other_names: list[str]) -> None:
        self.roi_name: str = roi_name.lower()
        self.other_names: list[str] = list({name.lower() for name in other_names})

    def add_name(self, roi_name: str) -> None:
        """Register an additional alternative name."""
        lower = roi_name.lower()
        if lower not in self.other_names:
            self.other_names.append(lower)


# ===========================  MAIN CLASS  ==================================


class DicomReaderWriter:
    """Read DICOM images / RT structures / dose files and write RT predictions.

    This is the primary entry-point for the package.  See the module-level
    docstring and the project README for usage examples.

    Args:
        description: Tag written into output NIfTI filenames.
        Contour_Names: list of ROI names to extract masks for.
        associations: list of :class:`ROIAssociationClass` for name mapping.
        arg_max: If ``True``, collapse the multi-channel mask via argmax.
        verbose: Print / log progress information.
        create_new_RT: Create a fresh RT struct when writing predictions.
        template_dir: Path to a template ``template_RS.dcm``.
        delete_previous_rois: Remove existing ROIs when writing predictions.
        require_all_contours: Require *every* listed contour to be present.
        iteration: Integer tag for NIfTI filename versioning.
        get_dose_output: Also load dose grids when calling
            :meth:`get_images_and_mask`.
        flip_axes: 3-tuple of booleans - axes to flip after loading.
        index: Initial series-dictionary index.
        series_instances_dictionary: Pre-populated series dictionary.
        plan_pydicom_string_keys: Extra pydicom keys to read from RT Plans.
        struct_pydicom_string_keys: Extra pydicom keys to read from RT Structs.
        image_sitk_string_keys: Extra SITK keys to read from images.
        dose_sitk_string_keys: Extra SITK keys to read from dose files.
        group_dose_by_frame_of_reference: Fall back to frame-of-reference
            matching when dose cannot be associated via plan/structure refs.
    """

    # -- Type annotations for instance attributes ---------------------------
    images_dictionary: dict[str, ImageBase]
    rt_dictionary: dict[str, RTBase]
    rd_dictionary: dict[str, RDBase]
    rp_dictionary: dict[str, PlanBase]
    rois_in_index_dict: dict[int, list[str]]
    dicom_handle: sitk.Image | None
    dose_handle: sitk.Image | None
    annotation_handle: sitk.Image | None
    all_rois: list[str]
    roi_class_list: list[ROIClass]
    rois_in_loaded_index: list[str]
    indexes_with_contours: list[int]
    roi_groups: dict[str, list[str]]
    all_RTs: dict[str, list[str]]
    RTs_with_ROI_Names: dict[str, list[str]]
    series_instances_dictionary: dict[int, ImageBase]
    mask_dictionary: dict[str, sitk.Image]
    mask: np.ndarray | None

    def __init__(
        self,
        description: str = "",
        Contour_Names: list[str] | None = None,
        associations: list[ROIAssociationClass] | None = None,
        arg_max: bool = True,
        verbose: bool = True,
        create_new_RT: bool = True,
        template_dir: str | None = None,
        delete_previous_rois: bool = True,
        require_all_contours: bool = True,
        iteration: int = 0,
        get_dose_output: bool = False,
        flip_axes: tuple[bool, bool, bool] = (False, False, False),
        index: int = 0,
        series_instances_dictionary: dict[int, ImageBase] | None = None,
        plan_pydicom_string_keys: PyDicomKeys | None = None,
        struct_pydicom_string_keys: PyDicomKeys | None = None,
        image_sitk_string_keys: SitkDicomKeys | None = None,
        dose_sitk_string_keys: SitkDicomKeys | None = None,
        group_dose_by_frame_of_reference: bool = True,
        mask_thread_count: int = 1,
    ) -> None:
        # Logging setup
        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        # Internal state
        self.roi_class_list: list[ROIClass] = []
        self.dose: np.ndarray | None = None
        self.group_dose_by_frame_of_reference = group_dose_by_frame_of_reference
        # Per-ROI rasterisation threads inside get_mask(). 1 = serial (default).
        # The bulk writers auto-tune this so per-ROI threads only spin up when
        # there are spare cores not already used by series-level parallelism.
        self.mask_thread_count = mask_thread_count
        self.verbose = verbose
        self.annotation_handle: sitk.Image | None = None
        self.dicom_handle: sitk.Image | None = None
        self.dose_handle: sitk.Image | None = None
        self.rois_in_index_dict: dict[int, list[str]] = {}
        self.rt_dictionary: dict[str, RTBase] = {}
        self.mask_dictionary: dict[str, sitk.Image] = {}
        self._dicom_handle_uid: str | None = None
        self._dicom_info_uid: str | None = None
        self._rs_struct_uid: str | None = None
        # Per-slice ImagePositionPatient[2] (mm) from the source DICOM
        # series, in the same order as ``self.dicom_handle``'s Z axis.
        # Populated by ``get_images()``; used by ``reshape_contour_data`` to
        # map each contour plane's physical z-mm to a slice index by
        # nearest-neighbour lookup against the actual per-DICOM IPP[2],
        # avoiding the uniform-spacing assumption baked into
        # ``TransformPhysicalPointToIndex``. Critical for non-uniform-Z
        # CT acquisitions (mixed 3 mm / 6 mm slice gaps, common on
        # NSCLC-Radiomics): the uniform-spacing path mis-rounds the z
        # index for several contour planes per ROI and the resulting mask
        # is missing slices. ``None`` when no series has been loaded yet
        # or the IPP read failed; ``reshape_contour_data`` falls back to
        # the legacy uniform-spacing path in that case.
        self._dicom_slice_z_positions: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self._rd_study_instance_uid: str | None = None
        self.index = index
        self.all_RTs: dict[str, list[str]] = {}
        self.RTs_with_ROI_Names: dict[str, list[str]] = {}
        self.all_rois: list[str] = []
        self.roi_groups: dict[str, list[str]] = {}
        self.indexes_with_contours: list[int] = []
        self.plan_pydicom_string_keys = plan_pydicom_string_keys
        self.struct_pydicom_string_keys = struct_pydicom_string_keys
        self.image_sitk_string_keys = image_sitk_string_keys
        self.dose_sitk_string_keys = dose_sitk_string_keys
        self.images_dictionary: dict[str, ImageBase] = {}
        self.rd_dictionary: dict[str, RDBase] = {}
        self.rp_dictionary: dict[str, PlanBase] = {}
        self.series_instances_dictionary: dict[int, ImageBase] = (
            series_instances_dictionary if series_instances_dictionary is not None else {}
        )
        self.get_dose_output = get_dose_output
        self.require_all_contours = require_all_contours
        self.flip_axes = flip_axes
        self.create_new_RT = create_new_RT
        self.arg_max = arg_max
        if template_dir is None or not os.path.exists(template_dir):
            template_dir = os.path.join(os.path.dirname(__file__), "template_RS.dcm")
        self.template_dir = template_dir
        self.template = True
        self.delete_previous_rois = delete_previous_rois
        self.associations = associations
        self.Contour_Names: list[str] = [c.lower() for c in Contour_Names] if Contour_Names else []

        # Backward-compatibility aliases
        self.RS_struct_uid = property(lambda self: self._rs_struct_uid)
        self.dicom_handle_uid = property(lambda self: self._dicom_handle_uid)

        self._init_readers()
        self.set_contour_names_and_associations(
            contour_names=Contour_Names, associations=associations, check_contours=False
        )
        self.description = description
        self.iteration = str(iteration)

    # -- Private helpers (readers) -----------------------------------------

    def _init_readers(self) -> None:
        self.reader = sitk.ImageSeriesReader()
        self.image_reader = sitk.ImageFileReader()
        self.image_reader.LoadPrivateTagsOn()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.reader.SetOutputPixelType(sitk.sitkFloat32)

    # -- Backward-compatibility aliases for private attrs -------------------------
    # These used to be public; keep property access working.

    @property
    def RS_struct_uid(self) -> str | None:  # type: ignore[override]
        return self._rs_struct_uid

    @RS_struct_uid.setter
    def RS_struct_uid(self, value: str | None) -> None:
        self._rs_struct_uid = value

    @property
    def dicom_handle_uid(self) -> str | None:  # type: ignore[override]
        return self._dicom_handle_uid

    @dicom_handle_uid.setter
    def dicom_handle_uid(self, value: str | None) -> None:
        self._dicom_handle_uid = value

    @property
    def rd_study_instance_uid(self) -> str | None:
        return self._rd_study_instance_uid

    @rd_study_instance_uid.setter
    def rd_study_instance_uid(self, value: str | None) -> None:
        self._rd_study_instance_uid = value

    @property
    def dicom_info_uid(self) -> str | None:
        return self._dicom_info_uid

    @dicom_info_uid.setter
    def dicom_info_uid(self, value: str | None) -> None:
        self._dicom_info_uid = value

    # -- Index management ---------------------------------------------------

    def set_index(self, index: int) -> None:
        """Select the active series by integer index."""
        self.index = index
        self.rois_in_loaded_index = self.rois_in_index_dict.get(index, [])

    # -- Setters ------------------------------------------------------------

    def set_description(self, description: str) -> None:
        """Set the description string used in NIfTI output filenames."""
        self.description = description

    def set_iteration(self, iteration: int = 0) -> None:
        """Set the iteration index used in NIfTI output filenames."""
        self.iteration = str(iteration)

    # -- State reset --------------------------------------------------------
    #
    # These public methods let callers wipe state between uses or targets —
    # for example, when reusing one ``DicomReaderWriter`` to process multiple
    # patient folders, or to switch contour-name targets and rebuild masks
    # from a known-clean slate. ``set_index`` deliberately does *not* reset
    # implicitly; it just selects an index against the existing in-memory
    # caches.

    def reset(self) -> None:
        """Wipe all loaded state.

        Clears the series, image, RT, and mask dictionaries plus the cached
        UIDs. Use this between unrelated walks (e.g. before
        ``walk_through_folders`` on a fresh patient corpus).
        """
        self.reset_rts()
        self._rd_study_instance_uid = None
        self._dicom_handle_uid = None
        self._dicom_info_uid = None
        self.series_instances_dictionary = {}
        self.rt_dictionary = {}
        self.images_dictionary = {}
        self.mask_dictionary = {}

    def reset_mask(self) -> None:
        """Re-allocate an empty mask sized to the currently-loaded image and
        clear ``mask_dictionary``.

        Call this after changing ``Contour_Names`` (e.g. via
        ``set_contour_names_and_associations``) and before ``get_mask()`` so
        the freshly-allocated mask has the right number of channels for the
        new contour set.
        """
        self._make_empty_mask()
        self.mask_dictionary = {}

    def reset_rts(self) -> None:
        """Clear ROI bookkeeping (names, groups, indexes_with_contours).

        Lighter-weight than :meth:`reset`: leaves loaded images intact, only
        clears the contour-related state. Useful between
        ``set_contour_names_and_associations`` calls.
        """
        self.all_rois = []
        self.roi_class_list = []
        self.roi_groups = {}
        self.indexes_with_contours = []
        self._rs_struct_uid = None
        self.RTs_with_ROI_Names = {}

    # -- Mask management ----------------------------------------------------

    def _make_empty_mask(self) -> None:
        if self.dicom_handle is not None:
            cols, rows, slices = self.dicom_handle.GetSize()
            self.image_size_cols = cols
            self.image_size_rows = rows
            self.image_size_z = slices
            self.mask = np.zeros(
                (slices, rows, cols, len(self.Contour_Names) + 1), dtype=np.int8
            )
            self.annotation_handle = sitk.GetImageFromArray(self.mask)

    # -- Compilation (linking images ↔ RTs ↔ doses ↔ plans) ----------------

    def _compile(self) -> None:
        """Link image, RT, dose, and plan dictionaries by SeriesInstanceUID."""
        if self.verbose:
            logger.info("Compiling dictionaries together...")

        existing_uids = [v.SeriesInstanceUID for v in self.series_instances_dictionary.values()]
        next_idx = max(self.series_instances_dictionary.keys(), default=-1) + 1

        # --- 1. Images ---
        for uid in sorted(self.images_dictionary.keys()):
            if uid not in existing_uids:
                self.series_instances_dictionary[next_idx] = self.images_dictionary[uid]
                existing_uids.append(uid)
                next_idx += 1

        # --- 2. RT Structures → images (by referenced series UID) ---
        for rt_uid, rt in self.rt_dictionary.items():
            ref_uid = rt.SeriesInstanceUID
            self.all_RTs[rt.path] = rt.ROI_Names
            for roi in rt.ROI_Names:
                self.RTs_with_ROI_Names.setdefault(roi, []).append(rt.path)

            if ref_uid in existing_uids:
                idx = list(self.series_instances_dictionary.keys())[existing_uids.index(ref_uid)]
                self.series_instances_dictionary[idx].RTs[rt_uid] = rt
            else:
                tmpl = ImageBase()
                tmpl.RTs[rt_uid] = rt
                self.series_instances_dictionary[next_idx] = tmpl
                next_idx += 1

        # --- 3. Doses → RT structures (by referenced structure SOP) ---
        for rd_uid, rd in self.rd_dictionary.items():
            if rd.ReferencedStructureSetSOPInstanceUID is None:
                continue
            for _img_key, img in self.series_instances_dictionary.items():
                for _rt_key, rt_entry in img.RTs.items():
                    if rd.ReferencedStructureSetSOPInstanceUID == rt_entry.SOPInstanceUID:
                        rt_entry.Doses[rd_uid] = rd
                        img.RDs[rd_uid] = rd
                        rd.Grouped = True

        # --- 4. Plans → RT structures ---
        for rp_uid, rp in self.rp_dictionary.items():
            added = False
            ref = rp.ReferencedStructureSetSOPInstanceUID
            for _img_key, img in self.series_instances_dictionary.items():
                for _rt_key, rt_entry in img.RTs.items():
                    if ref == rt_entry.SOPInstanceUID:
                        rt_entry.Plans[rp_uid] = rp
                        img.RPs[rp_uid] = rp
                        added = True
            if not added:
                tmpl = ImageBase()
                tmpl.RPs[rp_uid] = rp
                self.series_instances_dictionary[next_idx] = tmpl
                next_idx += 1

        # --- 5. Doses → Plans (when no struct ref, but plan ref exists) ---
        for rd_uid, rd in self.rd_dictionary.items():
            if rd.ReferencedStructureSetSOPInstanceUID is not None:
                continue
            plan_ref = rd.ReferencedPlanSOPInstanceUID
            for _img_key, img in self.series_instances_dictionary.items():
                for _rp_key, rp_entry in img.RPs.items():
                    if plan_ref == rp_entry.SOPInstanceUID:
                        # Find the associated RT via plan's struct ref
                        rt_sop = rp_entry.ReferencedStructureSetSOPInstanceUID
                        for _rt_key, rt_entry in img.RTs.items():
                            if rt_entry.SOPInstanceUID == rt_sop:
                                rt_entry.Doses[rd_uid] = rd
                        img.RDs[rd_uid] = rd
                        rd.Grouped = True

        # --- 6. Ungrouped doses → frame of reference fallback ---
        for rd_uid, rd in self.rd_dictionary.items():
            if rd.Grouped:
                continue
            added = False
            if self.group_dose_by_frame_of_reference:
                for _img_key, img in self.series_instances_dictionary.items():
                    if img.StudyInstanceUID != rd.StudyInstanceUID:
                        continue
                    if img.FrameOfReference == rd.ReferencedFrameOfReference:
                        img.RDs[rd_uid] = rd
                        added = True
                        if self.verbose:
                            logger.info(
                                "Dose files %s grouped with images at %s via Frame of Reference",
                                rd.Dose_Files, img.path,
                            )
            if not added:
                tmpl = ImageBase()
                tmpl.RDs[rd_uid] = rd
                self.series_instances_dictionary[next_idx] = tmpl
                next_idx += 1

    # -- Contour name management -------------------------------------------

    def set_contour_names_and_associations(
        self,
        contour_names: list[str] | None = None,
        associations: list[ROIAssociationClass] | None = None,
        check_contours: bool = True,
    ) -> None:
        """Set ROI names to extract and optional name associations.

        Args:
            contour_names: Canonical ROI names (case-insensitive).
            associations: Name-mapping rules.
            check_contours: Re-scan indexes for contour availability.
        """
        if contour_names is not None:
            self.reset_rts()
            self.Contour_Names = [n.lower() for n in contour_names]

        if associations is not None:
            self.associations = associations
            self.hierarchy: dict = {}

        if check_contours:
            self._check_all_contours()

        if contour_names is not None or self.associations is not None:
            if self.verbose:
                logger.info("Contour names or associations changed, resetting mask")
            self.reset_mask()


    def _check_contours_at_index(
        self, index: int, RTs: dict[str, RTBase] | None = None
    ) -> None:
        """Check which requested contours exist at a given index."""
        self.rois_in_loaded_index = []
        entry = self.series_instances_dictionary[index]
        if entry.path is None:
            return

        if RTs is None:
            RTs = entry.RTs

        true_rois: list[str] = []
        for _rt_key, rt in RTs.items():
            # Collect code associations
            for code, names in rt.CodeAssociations.items():
                existing = self.roi_groups.get(code, [])
                self.roi_groups[code] = list(set(existing + names))

            for roi in rt.ROIs_In_Structure.values():
                name = roi.ROIName
                if name not in self.RTs_with_ROI_Names:
                    self.RTs_with_ROI_Names[name] = [rt.path]
                elif rt.path not in self.RTs_with_ROI_Names[name]:
                    self.RTs_with_ROI_Names[name].append(rt.path)

                if name not in self.rois_in_loaded_index:
                    self.rois_in_loaded_index.append(name)
                if name not in self.all_rois:
                    self.all_rois.append(name)
                    self.roi_class_list.append(roi)

                if self.Contour_Names:
                    if name in self.Contour_Names:
                        true_rois.append(name)
                    elif self.associations:
                        for assoc in self.associations:
                            if name in assoc.other_names:
                                true_rois.append(assoc.roi_name)
                            elif name in self.Contour_Names:
                                true_rois.append(name)

        # Determine what's missing
        lacking = [r for r in self.Contour_Names if r not in true_rois]
        some_exist = any(r in true_rois for r in self.Contour_Names)

        if lacking and self.verbose:
            logger.info(
                "Index %d at %s lacks %s. Found: %s",
                index, entry.path, lacking, self.rois_in_loaded_index,
            )

        if index not in self.indexes_with_contours and (
            not lacking or (some_exist and not self.require_all_contours)
        ):
            self.indexes_with_contours.append(index)

    def _check_all_contours(self) -> None:
        self.indexes_with_contours = []
        for index in self.series_instances_dictionary:
            self._check_contours_at_index(index)
            self.rois_in_index_dict[index] = self.rois_in_loaded_index

    # -- Public query methods -----------------------------------------------

    def return_rois(self, print_rois: bool = True) -> list[str]:
        """Return all ROI names found across loaded series."""
        if print_rois:
            logger.info("The following ROIs were found:")
            for roi in self.all_rois:
                logger.info("  %s", roi)
        return self.all_rois

    def return_found_rois_with_same_code(self, print_rois: bool = True) -> dict[str, list[str]]:
        """Return ROIs grouped by their DICOM structure code."""
        if print_rois:
            for code, names in self.roi_groups.items():
                logger.info("Code %s: %s", code, ", ".join(names))
        return self.roi_groups

    def return_files_from_UID(self, UID: str) -> list[str]:
        """Return all file paths associated with an image series UID."""
        if UID not in self.images_dictionary:
            logger.warning("%s not found in images dictionary", UID)
            return []

        entry = self.images_dictionary[UID]
        reader = sitk.ImageSeriesReader()
        reader.GlobalWarningDisplayOff()
        out = list(reader.GetGDCMSeriesFileNames(entry.path, UID))
        for rt in entry.RTs.values():
            out.append(rt.path)
        for rd in entry.RDs.values():
            out.append(rd.path)
        return out

    def return_files_from_index(self, index: int) -> list[str]:
        """Return all file paths associated with a series-dictionary index."""
        entry = self.series_instances_dictionary[index]
        reader = sitk.ImageSeriesReader()
        reader.GlobalWarningDisplayOff()
        out = list(reader.GetGDCMSeriesFileNames(entry.path, entry.SeriesInstanceUID))
        for rt in entry.RTs.values():
            out.append(rt.path)
        for rp in entry.RPs.values():
            out.append(rp.path)
        for rd in entry.RDs.values():
            out.append(rd.path)
        return out

    def return_files_from_patientID(self, patientID: str) -> list[str]:
        """Return all file paths associated with a patient ID."""
        out: list[str] = []
        for index, entry in self.series_instances_dictionary.items():
            if entry.PatientID == patientID:
                out.extend(self.return_files_from_index(index))
        return out

    def where_is_ROI(self, ROIName: str) -> list[str]:
        """Return the RT file paths that contain the given ROI name."""
        key = ROIName.lower()
        if key in self.RTs_with_ROI_Names:
            return list(self.RTs_with_ROI_Names[key])
        logger.warning("%s was not found; check spelling or list all ROIs", ROIName)
        return []

    def which_indexes_have_all_rois(self) -> list[int]:
        """Return indexes where all requested ROIs are present."""
        if not self.Contour_Names:
            logger.warning("No contour names set. Use set_contour_names_and_associations().")
            return []
        if self.verbose:
            for idx in self.indexes_with_contours:
                logger.info("Index %d at %s", idx, self.series_instances_dictionary[idx].path)
        return list(self.indexes_with_contours)

    def which_indexes_lack_all_rois(self) -> list[int]:
        """Return indexes where one or more requested ROIs are missing."""
        if not self.Contour_Names:
            logger.warning("No contour names set. Use set_contour_names_and_associations().")
            return []
        lacking = [i for i in self.series_instances_dictionary if i not in self.indexes_with_contours]
        if self.verbose:
            for idx in lacking:
                logger.info("Index %d at %s", idx, self.series_instances_dictionary[idx].path)
        return lacking

    # -- Folder walking -----------------------------------------------------

    def walk_through_folders(
        self,
        input_path: PathLike,
        thread_count: int = _DEFAULT_WORKERS,
    ) -> None:
        """Recursively discover and catalogue all DICOM files under *input_path*.

        Args:
            input_path: Root directory to search.
            thread_count: Number of parallel threads for loading.
        """
        paths_with_dicom: list[str] = []
        for root, _dirs, files in os.walk(input_path):
            if any(f.lower().endswith(".dcm") for f in files):
                paths_with_dicom.append(root)

        if not paths_with_dicom:
            logger.info("No DICOM files found under %s", input_path)
            return

        loader = DicomFolderLoader(
            self.plan_pydicom_string_keys,
            self.struct_pydicom_string_keys,
            self.image_sitk_string_keys,
            self.dose_sitk_string_keys,
        )

        pbar = tqdm(total=len(paths_with_dicom), desc="Loading DICOM files")

        # Use thread_count=1 for deterministic/test runs.
        # Broad except is intentional here: this is a top-level "process
        # every folder, skip and log on any failure" loop. We do not want a
        # corrupt or surprising file to abort the whole walk.
        if thread_count <= 1:
            for path in paths_with_dicom:
                try:
                    loader.load(
                        path, self.images_dictionary, self.rt_dictionary,
                        self.rd_dictionary, self.rp_dictionary, self.verbose,
                    )
                except Exception:
                    logger.warning("Failed on %s", path, exc_info=True)
                pbar.update()
        else:
            with ThreadPoolExecutor(max_workers=thread_count) as pool:
                futures = {
                    pool.submit(
                        loader.load, path, self.images_dictionary,
                        self.rt_dictionary, self.rd_dictionary,
                        self.rp_dictionary, self.verbose,
                    ): path
                    for path in paths_with_dicom
                }
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        future.result()
                    except Exception:
                        logger.warning("Failed on %s", path, exc_info=True)
                    pbar.update()

        pbar.close()
        self._compile()

        if self.verbose or len(self.series_instances_dictionary) > 1:
            for key, entry in self.series_instances_dictionary.items():
                logger.info("Index %d, description %s at %s", key, entry.Description, entry.path)
            logger.info(
                "%d unique series IDs found. Default is index 0; change with set_index(index).",
                len(self.series_instances_dictionary),
            )
            self.set_index(0)

        self._check_all_contours()

    # -- Image loading ------------------------------------------------------

    def get_images_and_mask(self) -> None:
        """Load images and masks (and optionally dose) for the current index."""
        if self.index not in self.series_instances_dictionary:
            logger.error("Index %d not in dictionary. Use set_index().", self.index)
            return
        self.get_images()
        self.get_mask()
        if self.get_dose_output:
            self.get_dose()

    def get_all_info(self) -> None:
        """Log every DICOM metadata key/value for the current index."""
        self.load_key_information_only()
        for key in self.image_reader.GetMetaDataKeys():
            logger.info("%s = %s", key, self.image_reader.GetMetaData(key))

    def return_key_info(self, key: str) -> str | None:
        """Return a specific DICOM metadata value by key string."""
        self.load_key_information_only()
        if not self.image_reader.HasMetaDataKey(key):
            logger.warning("%s is not present in the reader", key)
            return None
        return self.image_reader.GetMetaData(key)

    def load_key_information_only(self) -> None:
        """Load only DICOM header info (no pixel data) for the current index."""
        if self.index not in self.series_instances_dictionary:
            logger.error("Index not in dictionary. Use set_index().")
            return
        entry = self.series_instances_dictionary[self.index]
        uid = entry.SeriesInstanceUID
        if self._dicom_info_uid != uid:
            self.image_reader.SetFileName(entry.files[0])
            self.image_reader.ReadImageInformation()
            self._dicom_info_uid = uid

    def get_images(self) -> None:
        """Load pixel data for the current index into ``self.ArrayDicom``."""
        if self.index not in self.series_instances_dictionary:
            logger.error("Index not in dictionary. Use set_index().")
            return
        entry = self.series_instances_dictionary[self.index]
        uid = entry.SeriesInstanceUID
        if uid is None:
            logger.warning("Index %d has no associated image series", self.index)
            return
        if self._dicom_handle_uid == uid:
            return  # Already loaded

        if self.verbose:
            logger.info("Loading images for '%s' at %s", entry.Description, entry.path)

        self.ds = dcmread(entry.files[0])
        self.reader.SetFileNames(entry.files)
        self.dicom_handle = self.reader.Execute()

        if self.verbose:
            logger.info("Resetting mask for new image set")
        self.reset_mask()

        add_sops(self.reader, self.series_instances_dictionary)

        # Read each source DICOM's ImagePositionPatient[2] in the same order
        # SimpleITK's ImageSeriesReader used (it sorts by IPP projection along
        # the slice axis). On non-uniform-Z CT acquisitions (mixed 3 mm / 6 mm
        # slice gaps), this per-slice array is the only correct way to map a
        # contour z-mm to a CT slice index; SimpleITK's
        # ``TransformPhysicalPointToIndex`` collapses the per-slice positions
        # into a single averaged spacing and rounds away from the real slice.
        # See ``reshape_contour_data`` for the consumer of this array.
        try:
            sorted_files = list(self.reader.GetFileNames())
            slice_zs = []
            for f in sorted_files:
                sub = dcmread(f, stop_before_pixels=True)
                ipp = getattr(sub, "ImagePositionPatient", None)
                if ipp is None or len(ipp) < 3:
                    raise ValueError(f"{f} has no ImagePositionPatient")
                slice_zs.append(float(ipp[2]))
            self._dicom_slice_z_positions = np.asarray(slice_zs, dtype=np.float64)
        except Exception as exc:  # pragma: no cover - logged for diagnosis
            logger.warning(
                "Could not build per-slice Z lookup; falling back to "
                "uniform-spacing TransformPhysicalPointToIndex (may misalign "
                "contours on non-uniform-Z CTs). Reason: %s",
                exc,
            )
            self._dicom_slice_z_positions = None

        if any(self.flip_axes):
            flip_filter = sitk.FlipImageFilter()
            flip_filter.SetFlipAxes(self.flip_axes)
            self.dicom_handle = flip_filter.Execute(self.dicom_handle)
            # FlipImageFilter on the Z axis reverses the in-memory slice
            # order. Keep the per-slice Z lookup in lockstep so
            # ``reshape_contour_data`` resolves contour z-mm against the
            # post-flip indices.
            if (
                self._dicom_slice_z_positions is not None
                and len(self.flip_axes) >= 3
                and self.flip_axes[2]
            ):
                self._dicom_slice_z_positions = self._dicom_slice_z_positions[::-1].copy()

        self.ArrayDicom = sitk.GetArrayFromImage(self.dicom_handle)
        self.image_size_cols, self.image_size_rows, self.image_size_z = (
            self.dicom_handle.GetSize()
        )
        self._dicom_handle_uid = uid

    # -- Dose loading -------------------------------------------------------

    def get_dose(self, dose_type: str = "PLAN") -> None:
        """Load radiation dose grids for the current index.

        Args:
            dose_type: Filter by DoseSummationType (e.g. ``"PLAN"``, ``"BEAM"``).
        """
        if self.index not in self.series_instances_dictionary:
            logger.error("Index not in dictionary. Use set_index().")
            return
        entry = self.series_instances_dictionary[self.index]
        if self._dicom_handle_uid != entry.SeriesInstanceUID:
            logger.info("Loading images first for index %d", self.index)
            self.get_images()
        if (
            self._rd_study_instance_uid is not None
            and self._rd_study_instance_uid == entry.StudyInstanceUID
        ):
            return  # Already loaded

        self._rd_study_instance_uid = entry.StudyInstanceUID
        self.dose = None
        resampler = ImageResampler()
        reader = sitk.ImageFileReader()
        output: sitk.Image | None = None
        filter_rds = len(entry.RDs) > 1

        for _rd_uid, rd in entry.RDs.items():
            if filter_rds and rd.DoseSummationType != dose_type:
                if self.verbose:
                    logger.info("Skipping dose type %s (loading %s)", rd.DoseSummationType, dose_type)
                continue
            for dose_file in rd.Dose_Files:
                reader.SetFileName(dose_file)
                reader.ReadImageInformation()
                dose_handle = reader.Execute()
                resampled = resampler.resample_image(
                    input_image_handle=dose_handle,
                    ref_resampling_handle=self.dicom_handle,
                    interpolator="Linear",
                    empty_value=0,
                )
                resampled = sitk.Cast(resampled, sitk.sitkFloat32)
                scale = float(reader.GetMetaData("3004|000e"))
                resampled *= scale
                output = resampled if output is None else output + resampled

        if output is not None:
            self.dose = sitk.GetArrayFromImage(output)
            self.dose_handle = output

    # -- Mask loading -------------------------------------------------------

    def _characterize_rt(self, rt: RTBase) -> None:
        """Load and index an RT structure set if not already loaded."""
        if self._rs_struct_uid != rt.SeriesInstanceUID:
            self.structure_references: dict[int, int] = {}
            self.RS_struct = dcmread(rt.path)
            self._rs_struct_uid = rt.SeriesInstanceUID
            for i, contour_seq in enumerate(self.RS_struct.ROIContourSequence):
                self.structure_references[contour_seq.ReferencedROINumber] = i

    def _return_mask_for_roi(self, rt: RTBase, roi_name: str) -> np.ndarray:
        self._characterize_rt(rt)
        struct_idx = self.structure_references[rt.ROIs_In_Structure[roi_name].ROINumber]
        return self._contours_to_mask(struct_idx, roi_name)

    def get_mask(self) -> None:
        """Build the annotation mask for the current index and contour names."""
        if self.index not in self.series_instances_dictionary:
            logger.error("Index not in dictionary. Use set_index().")
            return
        if not self.Contour_Names:
            logger.warning(
                "No contour names set. Use set_contour_names_and_associations() "
                "or call get_images() instead."
            )
            return

        entry = self.series_instances_dictionary[self.index]
        if self._dicom_handle_uid != entry.SeriesInstanceUID:
            logger.info("Loading images for index %d (mask requested)", self.index)
            self.get_images()

        channel_of = {name: i + 1 for i, name in enumerate(self.Contour_Names)}
        for _rt_key, rt in entry.RTs.items():
            # Warm the RT-struct cache once (serial) so the per-ROI workers below
            # only ever *read* self.RS_struct / self.structure_references.
            self._characterize_rt(rt)
            jobs: list[tuple[int, str, int]] = []
            for roi_name in rt.ROIs_In_Structure:
                true_name = self._resolve_roi_name(roi_name)
                if true_name and true_name in self.Contour_Names:
                    struct_idx = self.structure_references[rt.ROIs_In_Structure[roi_name].ROINumber]
                    jobs.append((struct_idx, roi_name, channel_of[true_name]))
            if not jobs:
                continue

            # Rasterise each ROI into its own array. The ROIs are independent and
            # only read shared state, so this parallelises across ROIs; the heavy
            # inner ops (SimpleITK transforms, cv2.fillPoly, numpy) release the
            # GIL. ``mask_thread_count == 1`` keeps the legacy serial path.
            def _raster(job: tuple[int, str, int]) -> tuple[int, np.ndarray]:
                struct_idx, roi_name, ch = job
                return ch, self._contours_to_mask(struct_idx, roi_name)

            if self.mask_thread_count > 1 and len(jobs) > 1:
                with ThreadPoolExecutor(max_workers=self.mask_thread_count) as pool:
                    results = list(pool.map(_raster, jobs))
            else:
                results = [_raster(job) for job in jobs]

            for ch, roi_mask in results:
                # Union the ROI into its channel, keeping it binary. Operate only
                # on the (z, rows, cols) channel view -- the previous
                # ``self.mask[self.mask > 1] = 1`` rescanned the entire 4-D array
                # once per ROI (O(n_rois) passes over ~all voxels).
                channel = self.mask[..., ch]
                np.maximum(channel, roi_mask, out=channel)

        # Build per-ROI sitk handles
        for name in self.Contour_Names:
            ch = self.Contour_Names.index(name) + 1
            mask_img = sitk.GetImageFromArray(self.mask[..., ch].astype(np.uint8))
            mask_img.SetSpacing(self.dicom_handle.GetSpacing())
            mask_img.SetDirection(self.dicom_handle.GetDirection())
            mask_img.SetOrigin(self.dicom_handle.GetOrigin())
            self.mask_dictionary[name] = mask_img

        # Flip if needed
        if self.flip_axes[0]:
            self.mask = self.mask[:, :, ::-1, ...]
        if self.flip_axes[1]:
            self.mask = self.mask[:, ::-1, ...]
        if self.flip_axes[2]:
            self.mask = self.mask[::-1, ...]

        # Compute volumes
        voxel_cc = np.prod(self.dicom_handle.GetSpacing()) / 1000.0
        volumes = np.sum(self.mask[..., 1:], axis=(0, 1, 2)) * voxel_cc
        entry.additional_tags["Volumes"] = volumes

        if self.arg_max:
            self.mask = np.argmax(self.mask, axis=-1)

        self.annotation_handle = sitk.GetImageFromArray(self.mask.astype(np.int8))
        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())

    def _resolve_roi_name(self, roi_name: str) -> str | None:
        """Map an ROI name to its canonical contour name, or None."""
        lower = roi_name.lower()
        if lower in self.Contour_Names:
            return lower
        if self.associations:
            for assoc in self.associations:
                if lower in assoc.other_names:
                    return assoc.roi_name
        return None

    # -- Contour geometry ---------------------------------------------------

    def reshape_contour_data(self, contour_data: np.ndarray) -> np.ndarray:
        """Convert flat contour data to Nx3 matrix of voxel indices.

        Uses SimpleITK's ``TransformPhysicalPointToIndex`` for the in-plane
        (col, row) coordinates -- those are not sensitive to Z-spacing.
        For the slice index, prefers a nearest-neighbour lookup against
        the per-DICOM ``ImagePositionPatient[2]`` array cached in
        ``self._dicom_slice_z_positions`` (correct on any Z-spacing
        pattern including the non-uniform CT case). Falls back to the
        ``TransformPhysicalPointToIndex`` Z when that array isn't
        available -- which preserves the legacy behaviour on synthetic
        test images that bypass ``get_images()``.
        """
        pts = np.asarray(contour_data).reshape(-1, 3)
        indices = np.array(
            [self.dicom_handle.TransformPhysicalPointToIndex(pts[i]) for i in range(len(pts))]
        )
        if self._dicom_slice_z_positions is not None and len(self._dicom_slice_z_positions) > 0:
            # Replace SimpleITK's uniform-spacing Z with the actual
            # nearest-IPP slice index for every contour point. Identical
            # output on uniform-Z CTs (rounding to the nearest of N
            # uniform positions and the nearest-IPP lookup against those
            # same N positions resolve to the same index); meaningfully
            # different only when the per-slice IPPs are non-uniform.
            slice_zs = self._dicom_slice_z_positions
            for i in range(len(pts)):
                z_mm = pts[i][2]
                indices[i, 2] = int(np.argmin(np.abs(slice_zs - z_mm)))
        return indices

    def return_mask(
        self, mask: np.ndarray, matrix_points: np.ndarray, geometric_type: str
    ) -> np.ndarray:
        """Rasterise contour points onto *mask*."""
        col_val = matrix_points[:, 0]
        row_val = matrix_points[:, 1]
        z_vals = matrix_points[:, 2]

        if geometric_type != "OPEN_NONPLANAR":
            temp = poly2mask(row_val, col_val, (self.image_size_rows, self.image_size_cols))
            mask[z_vals[0], temp] += 1
        else:
            self._rasterise_nonplanar(mask, col_val, row_val, z_vals)
        return mask

    @staticmethod
    def _rasterise_nonplanar(
        mask: np.ndarray,
        col_val: np.ndarray,
        row_val: np.ndarray,
        z_vals: np.ndarray,
    ) -> None:
        """Interpolate and rasterise an OPEN_NONPLANAR contour segment."""
        for i in range(len(z_vals) - 1, 0, -1):
            z0, z1 = int(z_vals[i]), int(z_vals[i - 1])
            r0, r1 = int(row_val[i]), int(row_val[i - 1])
            c0, c1 = int(col_val[i]), int(col_val[i - 1])
            dz, dr, dc = z1 - z0, r1 - r0, c1 - c0
            step = 1

            if dz != 0:
                r_slope, c_slope = dr / dz, dc / dz
                step = -1 if dz < 0 else 1
                for z in range(z0, z1 + step, step):
                    r = r0 + r_slope * (z - z0)
                    c = c0 + c_slope * (z - z0)
                    add_to_mask(mask, z, r, c)
            if dr != 0:
                c_slope, z_slope = dc / dr, dz / dr
                step = -1 if dr < 0 else 1
                for r in range(r0, r1 + step, step):
                    c = c0 + c_slope * (r - r0)
                    z = z0 + z_slope * (r - r0)
                    add_to_mask(mask, z, r, c)
            if dc != 0:
                r_slope, z_slope = dr / dc, dz / dc
                step = -1 if dc < 0 else 1
                for c in range(c0, c1 + step, step):
                    r = r0 + r_slope * (c - c0)
                    z = z0 + z_slope * (c - c0)
                    add_to_mask(mask, z, r, c)

    def contour_points_to_mask(
        self, contour_points: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Convert a flat array of physical contour points to a 3-D mask."""
        if mask is None:
            mask = np.zeros(
                (self.dicom_handle.GetSize()[-1], self.image_size_rows, self.image_size_cols),
                dtype=np.int8,
            )
        matrix_points = self.reshape_contour_data(contour_points)
        return self.return_mask(mask, matrix_points, "CLOSED_PLANAR")

    def _contours_to_mask(self, index: int, true_name: str) -> np.ndarray:
        """Convert all contour slices for one ROI into a binary 3-D mask."""
        size = self.dicom_handle.GetSize()
        mask = np.zeros((size[-1], self.image_size_rows, self.image_size_cols), dtype=np.int8)

        if Tag((0x3006, 0x0039)) not in self.RS_struct:
            logger.warning("Structure set has no contour data. Returning blank mask.")
            return mask

        contour_seq = self.RS_struct.ROIContourSequence[index]
        if Tag((0x3006, 0x0040)) not in contour_seq:
            logger.warning("No contour data for '%s'. Returning blank mask.", true_name)
            return mask

        for contour in contour_seq.ContourSequence:
            pts = self.reshape_contour_data(contour.ContourData[:])
            mask = self.return_mask(mask, pts, contour.ContourGeometricType)

        return mask % 2

    # Backward compat
    contours_to_mask = _contours_to_mask

    # -- NIfTI I/O ----------------------------------------------------------

    def write_images_annotations(
        self,
        out_path: PathLike,
        output_spacing: tuple[float, float, float] | None = None,
    ) -> None:
        """Write the current image and mask to NIfTI files.

        Args:
            out_path: Directory for output files.
            output_spacing: Optional ``(x, y, z)`` spacing in mm to resample
                the outputs to. The image and dose are resampled with linear
                interpolation; the label mask uses nearest-neighbour so labels
                are never blended. When ``None`` (default) outputs keep the
                native DICOM geometry and behaviour is unchanged.
        """
        os.makedirs(out_path, exist_ok=True)
        img_path = os.path.join(out_path, f"Overall_Data_{self.description}_{self.iteration}.nii.gz")
        ann_path = os.path.join(out_path, f"Overall_mask_{self.description}_y{self.iteration}.nii.gz")

        handle = self.dicom_handle
        if output_spacing is not None:
            handle = resample_to_spacing(handle, output_spacing, "Linear")
        if handle.GetPixelIDTypeAsString().find("32-bit signed integer") != 0:
            handle = sitk.Cast(handle, sitk.sitkFloat32)
        sitk.WriteImage(handle, img_path)

        ann = self.annotation_handle
        if output_spacing is None:
            # Keep the legacy behaviour: copy geometry from the image handle.
            ann.SetSpacing(self.dicom_handle.GetSpacing())
            ann.SetOrigin(self.dicom_handle.GetOrigin())
            ann.SetDirection(self.dicom_handle.GetDirection())
        else:
            # The resampled mask already carries the new geometry; copying the
            # native spacing back would corrupt it.
            ann = resample_to_spacing(ann, output_spacing, "Nearest")
        if ann.GetPixelIDTypeAsString().find("int") == -1:
            ann = sitk.Cast(ann, sitk.sitkUInt8)
        sitk.WriteImage(ann, ann_path)

        if self.dose_handle:
            dose_path = os.path.join(out_path, f"Overall_dose_{self.description}_{self.iteration}.nii.gz")
            dose = self.dose_handle
            if output_spacing is not None:
                dose = resample_to_spacing(dose, output_spacing, "Linear")
            sitk.WriteImage(dose, dose_path)

        marker = os.path.join(
            self.series_instances_dictionary[self.index].path,
            f"{self.description}_Iteration_{self.iteration}.txt",
        )
        with open(marker, "w"):
            pass

    # -- Parallel NIfTI writing ---------------------------------------------

    def write_parallel(
        self,
        out_path: PathLike,
        index_file: PathLike,
        thread_count: int = _DEFAULT_WORKERS,
    ) -> None:
        """Write all indexed series to NIfTI files in parallel.

        Args:
            out_path: Output directory.
            index_file: CSV file tracking iterations and per-ROI volumes.
                Created on first call; updated in place on subsequent calls
                so re-runs skip already-converted series.
            thread_count: Number of parallel worker threads.
        """
        os.makedirs(out_path, exist_ok=True)

        if not os.path.exists(index_file):
            cols = {"PatientID": [], "Path": [], "Iteration": [], "Folder": [],
                    "SeriesInstanceUID": [], "Pixel_Spacing_X": [], "Pixel_Spacing_Y": [],
                    "Slice_Thickness": []}
            for roi in self.Contour_Names:
                cols[f"Volume_{roi} [cc]"] = []
            df = pd.DataFrame(cols)
            df.to_csv(index_file, index=False)
        else:
            df = pd.read_csv(index_file)

        # Ensure volume columns exist.
        for roi in self.Contour_Names:
            col = f"Volume_{roi} [cc]"
            if col not in df.columns:
                df[col] = np.nan
                df.to_csv(index_file, index=False)

        # Register new series.
        rewrite = False
        for index in self.indexes_with_contours:
            uid = self.series_instances_dictionary[index].SeriesInstanceUID
            if df.loc[df["SeriesInstanceUID"] == uid].shape[0] == 0:
                rewrite = True
                iteration = 0
                while iteration in df["Iteration"].values:
                    iteration += 1
                entry = self.series_instances_dictionary[index]
                new_row = pd.DataFrame({
                    "PatientID": [entry.PatientID], "Path": [entry.path],
                    "Iteration": [int(iteration)], "Folder": [None],
                    "SeriesInstanceUID": [uid],
                    "Pixel_Spacing_X": [entry.pixel_spacing_x],
                    "Pixel_Spacing_Y": [entry.pixel_spacing_y],
                    "Slice_Thickness": [entry.slice_thickness],
                })
                df = pd.concat([df, new_row], ignore_index=True)
        if rewrite:
            df.to_csv(index_file, index=False)

        # Build work items
        key_dict = {
            "series_instances_dictionary": self.series_instances_dictionary,
            "associations": self.associations, "arg_max": self.arg_max,
            "require_all_contours": self.require_all_contours,
            "Contour_Names": self.Contour_Names, "description": self.description,
            "get_dose_output": self.get_dose_output,
        }
        items = []
        for index in self.indexes_with_contours:
            uid = self.series_instances_dictionary[index].SeriesInstanceUID
            prev = df.loc[df["SeriesInstanceUID"] == uid]
            if prev.shape[0] == 0:
                continue
            iteration = int(prev["Iteration"].values[0])
            folder = prev["Folder"].values[0]
            if pd.isna(folder):
                folder = None
            wp = os.path.join(out_path, folder) if folder else out_path
            img_file = os.path.join(wp, f"Overall_Data_{self.description}_{iteration}.nii.gz")

            should_run = True
            if os.path.exists(img_file):
                should_run = False
                for roi in self.Contour_Names:
                    if pd.isna(prev[f"Volume_{roi} [cc]"].values[0]):
                        should_run = True
                        break
            if should_run:
                items.append((iteration, index, wp, key_dict))

        if not items:
            return

        pbar = tqdm(total=len(items), desc="Writing NIfTI files...")

        def _worker(item):
            iteration, idx, wp, kd = item
            base = DicomReaderWriter(**kd)
            try:
                base.set_index(idx)
                base.get_images_and_mask()
                base.set_iteration(iteration)
                base.write_images_annotations(wp)
            except Exception:
                entry = base.series_instances_dictionary[idx]
                logger.warning("Failed on %s", entry.path, exc_info=True)
                with open(os.path.join(entry.path, "failed.txt"), "w"):
                    pass

        if thread_count <= 1:
            for item in items:
                _worker(item)
                pbar.update()
        else:
            with ThreadPoolExecutor(max_workers=thread_count) as pool:
                futures = {pool.submit(_worker, item): item for item in items}
                for future in as_completed(futures):
                    future.result()
                    pbar.update()
        pbar.close()

        # Update volume data.
        for item in items:
            idx = item[1]
            iteration = item[0]
            tags = self.series_instances_dictionary[idx].additional_tags
            if "Volumes" not in tags:
                continue
            for ri, roi in enumerate(self.Contour_Names):
                col = f"Volume_{roi} [cc]"
                df.loc[df.Iteration == iteration, col] = tags["Volumes"][ri]
        df.to_csv(index_file, index=False)

    # -- Per-ROI NIfTI writing (C#-compatible layout) ----------------------

    def write_per_roi(
        self,
        out_path: PathLike,
        output_spacing: tuple[float, float, float] | None = None,
        anonymize: bool = False,
        salt: str = DEFAULT_SALT,
        thread_count: int = _DEFAULT_WORKERS,
        rois: list[str] | None = None,
        manifest_name: str = "manifest.csv",
        key_file_name: str = "anonymization_key.json",
        dose_type: str = "PLAN",
    ) -> None:
        """Export every series-with-contours to a per-ROI NIfTI layout.

        For each series an ``<case_id>/`` folder is written containing:

        * ``image.nii.gz`` — the image volume;
        * ``masks/<roi>.nii.gz`` — one binary mask per ROI;
        * ``doses/<desc>.nii.gz`` — the summed dose grid, only when the series
          carries dose *and* this reader was built with ``get_dose_output=True``.

        A single ``manifest.csv`` (no persistent iteration index) is written at
        *out_path* with one row per series: identifiers (raw and/or hashed),
        the output spacing, and per-ROI volume in cc (``-1`` when an ROI is
        absent for that series).

        When ``output_spacing`` is given the image and dose are resampled with
        linear interpolation and masks with nearest-neighbour. When
        ``anonymize`` is ``True`` the case folder is named by the series hash,
        only hashed identifiers are written to the manifest, and an
        ``anonymization_key.json`` reverse-lookup file is saved at *out_path*.

        Args:
            out_path: Output directory.
            output_spacing: Optional ``(x, y, z)`` mm spacing to resample to.
            anonymize: Hash identifiers + folder names and write a key file.
            salt: Salt for the deterministic hashes.
            thread_count: Number of parallel worker threads.
            rois: ROI names to export. Defaults to :attr:`Contour_Names`.
            manifest_name: Filename of the manifest CSV.
            key_file_name: Filename of the anonymization key JSON.
            dose_type: ``DoseSummationType`` filter used when loading dose.
        """
        os.makedirs(out_path, exist_ok=True)
        wanted_rois = [r.lower() for r in (rois if rois is not None else self.Contour_Names)]
        if not wanted_rois:
            logger.warning("No ROIs to export. Set Contour_Names or pass rois=...")
            return

        items = list(self.indexes_with_contours)
        if not items:
            logger.warning("No indexes with contours found; nothing to export.")
            return

        key_dict = {
            "series_instances_dictionary": self.series_instances_dictionary,
            "associations": self.associations, "arg_max": self.arg_max,
            "require_all_contours": self.require_all_contours,
            "Contour_Names": self.Contour_Names, "description": self.description,
            "get_dose_output": self.get_dose_output,
            # Spare cores go to per-ROI rasterisation when few series are written.
            "mask_thread_count": _mask_threads_for(thread_count, len(items)),
        }

        pbar = tqdm(total=len(items), desc="Writing per-ROI NIfTI files...")

        def _worker(index: int) -> dict | None:
            base = DicomReaderWriter(**key_dict)
            base.verbose = False
            try:
                base.set_index(index)
                base.get_images_and_mask()
                return self._export_one_series(
                    base, index, out_path, wanted_rois,
                    output_spacing, anonymize, salt, dose_type,
                )
            except Exception:
                entry = base.series_instances_dictionary.get(index)
                logger.warning("Failed on %s", getattr(entry, "path", index), exc_info=True)
                return None

        rows: list[dict] = []
        if thread_count <= 1:
            for index in items:
                result = _worker(index)
                if result is not None:
                    rows.append(result)
                pbar.update()
        else:
            with ThreadPoolExecutor(max_workers=thread_count) as pool:
                futures = {pool.submit(_worker, index): index for index in items}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        rows.append(result)
                    pbar.update()
        pbar.close()

        if not rows:
            logger.warning("No series exported; manifest not written.")
            return

        self._write_manifest(rows, wanted_rois, out_path, manifest_name, anonymize)

        if anonymize:
            key = AnonymizationKey(salt=salt)
            for row in rows:
                key.patients.setdefault(row["patient_hash"], row["_mrn"])
                key.studies.setdefault(row["study_hash"], row["_study_uid"])
                key.series.setdefault(row["series_hash"], row["_series_uid"])
            key.save(os.path.join(out_path, key_file_name))

    def _export_one_series(
        self,
        base: DicomReaderWriter,
        index: int,
        out_path: PathLike,
        wanted_rois: list[str],
        output_spacing: tuple[float, float, float] | None,
        anonymize: bool,
        salt: str,
        dose_type: str,
    ) -> dict:
        """Write one series' image/masks/dose tree and return its manifest data."""
        entry = base.series_instances_dictionary[index]
        mrn = entry.PatientID or ""
        study_uid = entry.StudyInstanceUID or ""
        series_uid = entry.SeriesInstanceUID or ""

        patient_hash = hash_patient(mrn, salt)
        study_hash = hash_study(study_uid, salt)
        series_hash = hash_series(series_uid, salt)

        case_id = (
            series_hash if anonymize
            else _sanitize_filename(f"{mrn}_{series_uid[-8:]}") or series_hash
        )

        case_dir = os.path.join(out_path, case_id)
        masks_dir = os.path.join(case_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        # ROIs genuinely present in this series' RT structure(s). ``get_mask``
        # allocates an all-zero mask for *every* ``Contour_Name``, so the mask
        # dictionary alone cannot distinguish "absent" from "empty"; resolve
        # the RT membership instead so absent ROIs become -1 in the manifest
        # and no empty mask file is written.
        present_rois: set[str] = set()
        for rt in entry.RTs.values():
            for raw_name in rt.ROIs_In_Structure:
                canonical = base._resolve_roi_name(raw_name)
                if canonical:
                    present_rois.add(canonical)

        # --- image (linear) ---
        image_handle = base.dicom_handle
        if output_spacing is not None:
            image_handle = resample_to_spacing(image_handle, output_spacing, "Linear")
        if image_handle.GetPixelIDTypeAsString().find("32-bit signed integer") != 0:
            image_handle = sitk.Cast(image_handle, sitk.sitkFloat32)
        sitk.WriteImage(image_handle, os.path.join(case_dir, "image.nii.gz"))

        # --- masks (nearest neighbour) ---
        native_spacing = base.dicom_handle.GetSpacing()
        out_spacing = tuple(float(s) for s in (output_spacing if output_spacing is not None else native_spacing))
        voxel_cc = float(np.prod(out_spacing)) / 1000.0
        roi_volumes: dict[str, float] = {}
        for roi in wanted_rois:
            if roi not in present_rois or roi not in base.mask_dictionary:
                continue
            mask_img = base.mask_dictionary[roi]
            if output_spacing is not None:
                mask_img = resample_to_spacing(mask_img, output_spacing, "Nearest")
            mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
            sitk.WriteImage(mask_img, os.path.join(masks_dir, f"{_sanitize_filename(roi)}.nii.gz"))
            n_vox = int(np.sum(sitk.GetArrayViewFromImage(mask_img)))
            roi_volumes[roi] = round(n_vox * voxel_cc, 3)

        # --- dose (linear), only when present ---
        if base.dose_handle is not None:
            doses_dir = os.path.join(case_dir, "doses")
            os.makedirs(doses_dir, exist_ok=True)
            dose_img = base.dose_handle
            if output_spacing is not None:
                dose_img = resample_to_spacing(dose_img, output_spacing, "Linear")
            desc = None
            for rd in entry.RDs.values():
                if rd.Description:
                    desc = rd.Description
                    break
            desc = _sanitize_filename(desc or dose_type or "dose") or "dose"
            sitk.WriteImage(dose_img, os.path.join(doses_dir, f"{desc}.nii.gz"))

        return {
            "case_id": case_id,
            "patient_hash": patient_hash,
            "study_hash": study_hash,
            "series_hash": series_hash,
            "_mrn": mrn,
            "_study_uid": study_uid,
            "_series_uid": series_uid,
            "spacing_x": out_spacing[0],
            "spacing_y": out_spacing[1],
            "spacing_z": out_spacing[2],
            "volumes": roi_volumes,
        }

    def _manifest_record(
        self,
        row: dict,
        wanted_rois: list[str],
        anonymize: bool,
        include_case_id: bool = True,
    ) -> dict:
        """Flatten one collected series ``row`` into a manifest CSV record.

        Columns: optional ``case_id``, the three hashes, the raw identifiers
        (omitted when *anonymize*), the output spacing, and one ``<roi> cc``
        column per wanted ROI (``-1`` when the ROI is absent for that series).
        """
        record: dict = {}
        if include_case_id and "case_id" in row:
            record["case_id"] = row["case_id"]
        record["patient_hash"] = row["patient_hash"]
        record["study_hash"] = row["study_hash"]
        record["series_hash"] = row["series_hash"]
        if not anonymize:
            record["PatientID"] = row["_mrn"]
            record["StudyInstanceUID"] = row["_study_uid"]
            record["SeriesInstanceUID"] = row["_series_uid"]
        record["spacing_x"] = row["spacing_x"]
        record["spacing_y"] = row["spacing_y"]
        record["spacing_z"] = row["spacing_z"]
        for roi in wanted_rois:
            record[f"{roi} cc"] = row["volumes"].get(roi, -1)
        return record

    def _write_manifest(
        self,
        rows: list[dict],
        wanted_rois: list[str],
        out_path: PathLike,
        manifest_name: str,
        anonymize: bool,
    ) -> None:
        """Write the single per-series manifest CSV (one row per series)."""
        records = [self._manifest_record(row, wanted_rois, anonymize) for row in rows]
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(out_path, manifest_name), index=False)

    def _collect_series_row(
        self,
        base: DicomReaderWriter,
        index: int,
        wanted_rois: list[str],
        salt: str,
    ) -> dict:
        """Compute identifiers, native image spacing, and per-ROI cc volumes.

        Does not write any NIfTI files -- used by :meth:`create_manifest`.
        Volumes are taken from the native-resolution masks; an ROI absent from
        the series' RT structure(s) is simply omitted (becomes ``-1`` at CSV
        write time).
        """
        entry = base.series_instances_dictionary[index]
        mrn = entry.PatientID or ""
        study_uid = entry.StudyInstanceUID or ""
        series_uid = entry.SeriesInstanceUID or ""

        spacing = tuple(float(s) for s in base.dicom_handle.GetSpacing())
        voxel_cc = float(np.prod(spacing)) / 1000.0

        present_rois: set[str] = set()
        for rt in entry.RTs.values():
            for raw_name in rt.ROIs_In_Structure:
                canonical = base._resolve_roi_name(raw_name)
                if canonical:
                    present_rois.add(canonical)

        volumes: dict[str, float] = {}
        for roi in wanted_rois:
            if roi in present_rois and roi in base.mask_dictionary:
                n_vox = int(np.sum(sitk.GetArrayViewFromImage(base.mask_dictionary[roi])))
                volumes[roi] = round(n_vox * voxel_cc, 3)

        return {
            "patient_hash": hash_patient(mrn, salt),
            "study_hash": hash_study(study_uid, salt),
            "series_hash": hash_series(series_uid, salt),
            "_mrn": mrn,
            "_study_uid": study_uid,
            "_series_uid": series_uid,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "volumes": volumes,
        }

    def create_manifest(
        self,
        output_path: PathLike,
        anonymize: bool = False,
        salt: str = DEFAULT_SALT,
        rois: list[str] | None = None,
        thread_count: int = _DEFAULT_WORKERS,
    ) -> None:
        """Build, or incrementally extend, a metadata manifest CSV.

        Writes one row per series-with-contours to *output_path*, mirroring the
        C# tool's ``export_manifest.csv``: identifiers (raw and/or hashed),
        the image spacing (``spacing_x/y/z``), and the mask volume in cc for
        every ROI name (one ``<roi> cc`` column each; ``-1`` when an ROI is
        absent from that series). Unlike :meth:`write_per_roi`, no NIfTI files
        are written -- this produces the manifest only.

        If *output_path* already exists it is read and **extended in place**:
        rows for series already recorded (matched by ``SeriesInstanceUID`` or
        ``series_hash``) are left untouched, only new series are appended, and
        any new ROI columns are added (filled with ``-1`` for pre-existing
        rows). This makes it safe to call repeatedly as more data is walked.

        Args:
            output_path: CSV path to create or extend.
            anonymize: Write only the hashed identifiers (no raw IDs).
            salt: Salt for the deterministic identifier hashes.
            rois: ROI names to record. Defaults to :attr:`Contour_Names` when
                set, otherwise every ROI discovered during the walk.
            thread_count: Number of parallel worker threads.
        """
        # ROIs to record: explicit ``rois=``, else ``Contour_Names``, else
        # *every* ROI discovered during the walk (matches the C# manifest, which
        # records all ROIs it finds). Defaulting to all ROIs means a plain
        # ``walk_through_folders`` -> ``create_manifest`` call "just works"
        # without first calling ``set_contour_names_and_associations``.
        if rois is not None:
            wanted_rois = [r.lower() for r in rois]
        elif self.Contour_Names:
            wanted_rois = list(self.Contour_Names)
        else:
            wanted_rois = [r.lower() for r in self.all_rois]
        if not wanted_rois:
            logger.warning("No ROIs found to record. Walk a folder containing RT structures first.")
            return

        # Every series that carries at least one RT structure (independent of
        # ``indexes_with_contours``, which is gated by ``Contour_Names``).
        candidates = [
            idx for idx, entry in self.series_instances_dictionary.items()
            if entry.SeriesInstanceUID is not None and entry.RTs
        ]
        if not candidates:
            logger.warning("No series with RT structures found; manifest not written.")
            return

        # Load any existing manifest so we only append genuinely new series.
        existing_df = None
        existing_keys: set[str] = set()
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            for col in ("SeriesInstanceUID", "series_hash"):
                if col in existing_df.columns:
                    existing_keys.update(existing_df[col].dropna().astype(str).tolist())

        # Skip series already present (cheap -- no image load needed).
        todo: list[int] = []
        for index in candidates:
            entry = self.series_instances_dictionary[index]
            series_uid = entry.SeriesInstanceUID or ""
            if series_uid in existing_keys or hash_series(series_uid, salt) in existing_keys:
                continue
            todo.append(index)

        # The worker rasterises masks for exactly the ROIs we record (not the
        # parent's possibly-empty ``Contour_Names``), so volumes are populated
        # even when the caller never set contour names.
        key_dict = {
            "series_instances_dictionary": self.series_instances_dictionary,
            "associations": self.associations, "arg_max": self.arg_max,
            "require_all_contours": self.require_all_contours,
            "Contour_Names": wanted_rois, "description": self.description,
            "get_dose_output": self.get_dose_output,
            # Hand leftover cores to per-ROI rasterisation when few series are
            # being processed (e.g. a single multi-ROI series).
            "mask_thread_count": _mask_threads_for(thread_count, len(todo)),
        }

        rows: list[dict] = []
        if todo:
            pbar = tqdm(total=len(todo), desc="Building manifest...")

            def _worker(index: int) -> dict | None:
                base = DicomReaderWriter(**key_dict)
                base.verbose = False
                try:
                    base.set_index(index)
                    base.get_images_and_mask()
                    return self._collect_series_row(base, index, wanted_rois, salt)
                except Exception:
                    entry = base.series_instances_dictionary.get(index)
                    logger.warning("Failed on %s", getattr(entry, "path", index), exc_info=True)
                    return None

            if thread_count <= 1:
                for index in todo:
                    result = _worker(index)
                    if result is not None:
                        rows.append(result)
                    pbar.update()
            else:
                with ThreadPoolExecutor(max_workers=thread_count) as pool:
                    futures = {pool.submit(_worker, index): index for index in todo}
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            rows.append(result)
                        pbar.update()
            pbar.close()

        new_df = pd.DataFrame(
            [self._manifest_record(row, wanted_rois, anonymize, include_case_id=False) for row in rows]
        )

        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True) if not new_df.empty else existing_df
        else:
            combined = new_df

        if combined is None or combined.empty:
            logger.warning("No manifest rows to write.")
            return

        # Absent ROI volume cells (new columns on old rows, or vice versa) -> -1.
        for col in [c for c in combined.columns if str(c).endswith(" cc")]:
            combined[col] = combined[col].fillna(-1)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        combined.to_csv(output_path, index=False)

    # -- CSV characterisation ---------------------------------------------

    def characterize_data_to_csv(
        self,
        wanted_rois: list[str] | None = None,
        csv_path: PathLike = "./Data.csv",
    ) -> None:
        """Scan all indexes and record per-ROI volume data to two CSV files.

        Two files are produced next to *csv_path*:

        * ``<csv_path>`` — one row per RT structure with per-ROI volumes
          in cc.
        * ``<csv_path-stem>_images.csv`` — one row per image series with
          spacing + thickness.

        Args:
            wanted_rois: ROI names to evaluate. Defaults to
                :attr:`Contour_Names` or all found ROIs.
            csv_path: Output CSV file path for the ROIs table.
        """
        self.verbose = False
        loading_rois = self._resolve_wanted_rois(wanted_rois)
        loading_rois = list(set(loading_rois))

        final = {"PatientID": [], "PixelSpacingX": [], "PixelSpacingY": [],
                 "SliceThickness": [], "zzzRTPath": [], "zzzImagePath": []}
        image_data = {"PatientID": [], "ImagePath": [], "PixelSpacingX": [],
                      "PixelSpacingY": [], "SliceThickness": []}
        temp_assoc: dict[str, str] = {}
        columns: list[str] = []

        for roi in loading_rois:
            if self.associations:
                for assoc in self.associations:
                    if roi in assoc.other_names:
                        temp_assoc[roi] = assoc.roi_name
            if roi not in final:
                final[f"{roi} cc"] = []
                columns.append(roi)

        pbar = tqdm(total=len(self.series_instances_dictionary), desc="Building data...")
        for index, entry in self.series_instances_dictionary.items():
            pbar.update()
            if entry.SeriesInstanceUID is None:
                continue
            self.set_index(index)
            if not any(roi in self.rois_in_loaded_index for roi in columns):
                continue

            image_data["PatientID"].append(entry.PatientID)
            image_data["ImagePath"].append(entry.path)
            image_data["PixelSpacingX"].append(entry.pixel_spacing_x)
            image_data["PixelSpacingY"].append(entry.pixel_spacing_y)
            image_data["SliceThickness"].append(entry.slice_thickness)

            self.get_images()
            dim = np.prod(self.dicom_handle.GetSpacing())

            for _rt_idx, rt_base in entry.RTs.items():
                self._check_contours_at_index(index)
                final["PatientID"].append(rt_base.PatientID)
                final["zzzRTPath"].append(rt_base.path)
                final["zzzImagePath"].append(entry.path)
                final["PixelSpacingX"].append(entry.pixel_spacing_x)
                final["PixelSpacingY"].append(entry.pixel_spacing_y)
                final["SliceThickness"].append(entry.slice_thickness)

                for roi in columns:
                    final[f"{roi} cc"].append(np.nan)
                for roi in columns:
                    if roi in rt_base.ROI_Names:
                        m = self._return_mask_for_roi(rt_base, roi)
                        vol = round(np.sum(m) * dim / 1000, 3)
                        final[f"{roi} cc"][-1] = vol
        pbar.close()

        for target in temp_assoc.values():
            if target not in final:
                final[target] = [np.nan] * len(final["PatientID"])

        df = pd.DataFrame(final)
        for key, target in temp_assoc.items():
            df[target] = df[f"{key} cc"] + df.fillna(0)[target]
        df = df.reindex(sorted(df.columns), axis=1)
        df_img = pd.DataFrame(image_data)

        rois_path = Path(csv_path)
        images_path = rois_path.with_name(f"{rois_path.stem}_images{rois_path.suffix or '.csv'}")
        rois_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(rois_path, index=False)
        df_img.to_csv(images_path, index=False)

    def _resolve_wanted_rois(self, wanted_rois: list[str] | None) -> list[str]:
        if wanted_rois is None:
            if self.Contour_Names:
                return list(self.Contour_Names)
            return list(self.all_rois)
        result: list[str] = []
        for roi in wanted_rois:
            if roi in self.all_rois:
                result.append(roi)
            elif self.associations:
                for assoc in self.associations:
                    if assoc.roi_name == roi:
                        result.extend(assoc.other_names)
        return result

    # -- RT Structure writing -----------------------------------------------

    def prediction_array_to_RT(
        self,
        prediction_array: np.ndarray,
        output_dir: PathLike,
        ROI_Names: list[str],
        ROI_Types: list[str] | None = None,
    ) -> None:
        """Convert a multi-class prediction array into an RT Structure file.

        Args:
            prediction_array: Shape ``(slices, rows, cols, num_classes+1)``.
                Channel 0 is background.
            output_dir: Directory for the output ``.dcm`` file.
            ROI_Names: Names for each class channel (excluding background).
            ROI_Types: DICOM ROI types (default ``"ORGAN"`` for each).
        """
        if ROI_Names is None:
            logger.error("ROI_Names is required")
            return
        if prediction_array.shape[-1] != len(ROI_Names) + 1:
            logger.error(
                "prediction_array last dimension (%d) must equal len(ROI_Names)+1 (%d)",
                prediction_array.shape[-1], len(ROI_Names) + 1,
            )
            return
        if self.index not in self.series_instances_dictionary:
            logger.error("Index not in dictionary. Use set_index().")
            return

        entry = self.series_instances_dictionary[self.index]
        if self._dicom_handle_uid != entry.SeriesInstanceUID:
            self.get_images()

        self.SOPInstanceUIDs = entry.SOPs
        if self.create_new_RT or not entry.RTs:
            self.use_template()
        elif self._rs_struct_uid != entry.SeriesInstanceUID:
            for _uid_key, rt in entry.RTs.items():
                self.RS_struct = dcmread(rt.path)
                self._rs_struct_uid = entry.SeriesInstanceUID
                break

        prediction_array = np.squeeze(prediction_array)

        # Remove empty channels
        max_per_channel = np.max(prediction_array, axis=tuple(range(prediction_array.ndim - 1)))
        max_per_channel[0] = 1  # Keep background
        active = max_per_channel > 0
        prediction_array = prediction_array[..., active]
        active_names = [n for n, a in zip(ROI_Names, active[1:], strict=False) if a]
        dropped = [n for n, a in zip(ROI_Names, active[1:], strict=False) if not a]
        if dropped:
            logger.info("No mask data for ROIs: %s", dropped)

        self.image_size_z, self.image_size_rows, self.image_size_cols = prediction_array.shape[:3]
        self.ROI_Names = active_names
        self.ROI_Types = ROI_Types or ["ORGAN"] * len(active_names)
        self.output_dir = output_dir

        if prediction_array.ndim == 3:
            prediction_array = np.expand_dims(prediction_array, axis=-1)

        for axis, flip in enumerate(self.flip_axes):
            if flip:
                prediction_array = np.flip(prediction_array, axis={0: 2, 1: 1, 2: 0}[axis])

        self.annotations = prediction_array
        self._mask_to_contours()

    def _mask_to_contours(self) -> None:
        """Convert the prediction annotations array to DICOM RT contours."""
        self.PixelSize = self.dicom_handle.GetSpacing()

        current_names = [s.ROIName for s in self.RS_struct.StructureSetROISequence]
        color_palette = [
            [128, 0, 0], [170, 110, 40], [0, 128, 128], [0, 0, 128],
            [230, 25, 75], [225, 225, 25], [0, 130, 200], [145, 30, 180],
            [255, 255, 255],
        ]
        available_colors = list(color_palette)
        struct_index = 0
        new_roi_number = 1000
        base_annotations = copy.deepcopy(self.annotations)

        for name, roi_type in zip(self.ROI_Names, self.ROI_Types, strict=False):
            new_roi_number -= 1
            if not available_colors:
                available_colors = list(color_palette)
            color_idx = np.random.randint(len(available_colors))
            logger.info("Writing contour data for %s", name)

            ann_channel = self.ROI_Names.index(name) + 1
            annotations = copy.deepcopy(base_annotations[..., ann_channel]).astype(int)

            if name not in current_names or self.delete_previous_rois:
                self.RS_struct.StructureSetROISequence.insert(
                    0, copy.deepcopy(self.RS_struct.StructureSetROISequence[0])
                )
            else:
                logger.info("ROI '%s' already exists in RT structure, skipping", name)
                continue

            seq = self.RS_struct.StructureSetROISequence
            seq[struct_index].ROINumber = new_roi_number
            seq[struct_index].ReferencedFrameOfReferenceUID = self.ds.FrameOfReferenceUID
            seq[struct_index].ROIName = name
            seq[struct_index].ROIVolume = 0
            seq[struct_index].ROIGenerationAlgorithm = "SEMIAUTOMATIC"

            obs = self.RS_struct.RTROIObservationsSequence
            obs.insert(0, copy.deepcopy(obs[0]))
            if "MaterialID" in obs[struct_index]:
                del obs[struct_index].MaterialID
            obs[struct_index].ObservationNumber = new_roi_number
            obs[struct_index].ReferencedROINumber = new_roi_number
            obs[struct_index].ROIObservationLabel = name
            obs[struct_index].RTROIInterpretedType = roi_type

            contour_seq = self.RS_struct.ROIContourSequence
            contour_seq.insert(0, copy.deepcopy(contour_seq[0]))
            contour_seq[struct_index].ReferencedROINumber = new_roi_number
            del contour_seq[struct_index].ContourSequence[1:]
            contour_seq[struct_index].ROIDisplayColor = available_colors[color_idx]
            del available_colors[color_idx]

            # Build contour points using thread pool
            contour_dict: dict[int, list[np.ndarray]] = {}
            maker = PointOutputMaker(
                self.image_size_rows, self.image_size_cols,
                self.PixelSize, contour_dict,
            )

            contour_num = 0
            if np.max(annotations) > 0:
                slice_maxes = np.max(annotations, axis=(1, 2))
                active_slices = np.where(slice_maxes > 0)[0]

                thread_count = min(_DEFAULT_WORKERS, len(active_slices))
                if thread_count <= 1:
                    for si in active_slices:
                        maker.make_output(annotations[si], si, self.dicom_handle)
                else:
                    with ThreadPoolExecutor(max_workers=thread_count) as pool:
                        futures = [
                            pool.submit(maker.make_output, annotations[si], si, self.dicom_handle)
                            for si in active_slices
                        ]
                        for f in as_completed(futures):
                            f.result()

                for i in sorted(contour_dict.keys()):
                    for points in contour_dict[i]:
                        output = np.asarray(points).flatten("C")
                        if contour_num > 0:
                            contour_seq[struct_index].ContourSequence.append(
                                copy.deepcopy(contour_seq[struct_index].ContourSequence[0])
                            )
                        cs = contour_seq[struct_index].ContourSequence[contour_num]
                        cs.ContourNumber = str(contour_num)
                        cs.ContourGeometricType = "CLOSED_PLANAR"
                        cs.ContourImageSequence[0].ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
                        cs.ContourData = list(output)
                        cs.NumberOfContourPoints = len(output) // 3
                        contour_num += 1

        # Clean up unused template entries
        if self.template or self.delete_previous_rois:
            n = len(self.ROI_Names)
            for seq_name in ("StructureSetROISequence", "RTROIObservationsSequence", "ROIContourSequence"):
                seq = getattr(self.RS_struct, seq_name)
                while len(seq) > n:
                    del seq[-1]
            for i in range(n):
                self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
                self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
                self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.RS_struct.SeriesInstanceUID = pydicom.uid.generate_uid(prefix="1.2.826.0.1.3680043.8.498.")
        out_path = out_dir / f"RS_MRN{self.RS_struct.PatientID}_{self.RS_struct.SeriesInstanceUID}.dcm"
        # If the target already exists (extremely unlikely given the UID
        # prefix), append a numeric suffix to the stem rather than naively
        # patching the extension — handles paths whose stem also contains
        # ".dcm" without corrupting them.
        if out_path.exists():
            out_path = out_path.with_stem(out_path.stem + "_1")

        logger.info("Writing RT structure to %s", out_dir)
        pydicom.dcmwrite(os.fspath(out_path), self.RS_struct)
        (out_dir / "Completed.txt").touch()
        logger.info("Finished writing RT structure")

    # -- Template management ------------------------------------------------

    def use_template(self) -> None:
        """Load and configure the template RT structure."""
        self.template = True
        if not self.template_dir or not os.path.exists(self.template_dir):
            self.template_dir = os.path.join(os.path.dirname(__file__), "template_RS.dcm")
        self.key_list = self.template_dir.replace("template_RS.dcm", "key_list.txt")
        self.RS_struct = dcmread(self.template_dir)
        self.change_template()

    def change_template(self) -> None:
        """Update template RT struct references to match the current image series."""
        ref_key = Tag(0x3006, 0x0010)
        if ref_key in self.RS_struct:
            ref = self.RS_struct[ref_key]._value[0]
            ref.FrameOfReferenceUID = self.ds.FrameOfReferenceUID
            study_seq = ref.RTReferencedStudySequence[0]
            study_seq.ReferencedSOPInstanceUID = self.ds.StudyInstanceUID
            series_seq = study_seq.RTReferencedSeriesSequence[0]
            series_seq.SeriesInstanceUID = self.ds.SeriesInstanceUID

            # Replace contour image sequence
            del series_seq.ContourImageSequence[1:]
            template_seg = copy.deepcopy(series_seq.ContourImageSequence[0])
            for sop_uid in self.SOPInstanceUIDs:
                seg = copy.deepcopy(template_seg)
                seg.ReferencedSOPInstanceUID = sop_uid
                series_seq.ContourImageSequence.append(seg)
            del series_seq.ContourImageSequence[0]

        # Copy patient/study keys from image dataset
        key_list_path = self.template_dir.replace("template_RS.dcm", "key_list.txt")
        if os.path.exists(key_list_path):
            with open(key_list_path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        tag = (parts[0].strip(), parts[1].strip())
                        try:
                            self.RS_struct[tag[0], tag[1]] = self.ds[tag[0], tag[1]]
                        except (KeyError, AttributeError):
                            continue

    # -- RT rewriting -------------------------------------------------------

    def rewrite_RT(self, lstRSFile: PathLike | None = None) -> None:
        """Rename ROIs in an existing RT structure file using associations.

        Args:
            lstRSFile: Path to the RT structure file to rewrite.
        """
        if lstRSFile is not None:
            self.RS_struct = dcmread(lstRSFile)

        roi_struct = (
            self.RS_struct.StructureSetROISequence
            if Tag((0x3006, 0x0020)) in self.RS_struct
            else []
        )
        obs_seq = (
            self.RS_struct.RTROIObservationsSequence
            if Tag((0x3006, 0x0080)) in self.RS_struct
            else []
        )

        self.rois_in_loaded_index = []
        for i, structure in enumerate(roi_struct):
            if structure.ROIName in self.associations:
                self.RS_struct.StructureSetROISequence[i].ROIName = self.associations[structure.ROIName]
            self.rois_in_loaded_index.append(self.RS_struct.StructureSetROISequence[i].ROIName)

        for i, obs in enumerate(obs_seq):
            if obs.ROIObservationLabel in self.associations:
                self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel = (
                    self.associations[obs.ROIObservationLabel]
                )

        self.RS_struct.save_as(lstRSFile)
