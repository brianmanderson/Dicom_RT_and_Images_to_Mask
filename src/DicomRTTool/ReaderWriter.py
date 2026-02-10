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
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from NiftiResampler.ResampleTools import ImageResampler
from pydicom.tag import Tag
from skimage.measure import find_contours, label, regionprops
from tqdm import tqdm

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
from .Services.StaticScripts import add_to_mask, poly2mask
from .Viewer import plot_scroll_Image  # noqa: F401 – re-export

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
dcmread_function = dcmread

# ---------------------------------------------------------------------------
# Max worker threads (leave ~10 % headroom for the OS)
# ---------------------------------------------------------------------------
_DEFAULT_WORKERS = max(1, int(cpu_count() * 0.9) - 1)


# ---------------------------------------------------------------------------
# ROI association helper
# ---------------------------------------------------------------------------
class ROIAssociationClass:
    """Maps a canonical ROI name to one or more alternative names.

    Args:
        roi_name: The canonical (target) ROI name.
        other_names: Alternative names that should be treated as equivalent.
    """

    def __init__(self, roi_name: str, other_names: List[str]) -> None:
        self.roi_name: str = roi_name.lower()
        self.other_names: List[str] = list({name.lower() for name in other_names})

    def add_name(self, roi_name: str) -> None:
        """Register an additional alternative name."""
        lower = roi_name.lower()
        if lower not in self.other_names:
            self.other_names.append(lower)


# ---------------------------------------------------------------------------
# Contour-to-point worker (used during RT writing)
# ---------------------------------------------------------------------------
class _PointOutputMaker:
    """Converts a binary annotation slice into physical-space contour points."""

    def __init__(
        self,
        image_size_rows: int,
        image_size_cols: int,
        pixel_size: Tuple[float, ...],
        contour_dict: Dict[int, List[np.ndarray]],
    ) -> None:
        self.image_size_rows = image_size_rows
        self.image_size_cols = image_size_cols
        self.pixel_size = pixel_size
        self.contour_dict = contour_dict

    def make_output(
        self,
        annotation: np.ndarray,
        slice_index: int,
        dicom_handle: sitk.Image,
    ) -> None:
        self.contour_dict[slice_index] = []
        regions = regionprops(label(annotation))
        for region in regions:
            temp_image = np.zeros(
                (self.image_size_rows, self.image_size_cols), dtype=np.uint8
            )
            rows, cols = region.coords[:, 0], region.coords[:, 1]
            temp_image[rows, cols] = 1

            contours = find_contours(
                temp_image, level=0.5, fully_connected="low", positive_orientation="high"
            )
            for contour in contours:
                contour = np.squeeze(contour)
                # Remove co-linear points (same slope → redundant)
                with np.errstate(divide="ignore"):
                    slope = (contour[1:, 1] - contour[:-1, 1]) / (
                        contour[1:, 0] - contour[:-1, 0]
                    )
                prev_slope = None
                out: List[List[float]] = []
                for idx in range(len(slope)):
                    if slope[idx] != prev_slope:
                        out.append(contour[idx].tolist())
                    prev_slope = slope[idx]
                # Convert index → physical coordinates
                physical = [
                    [float(c[1]), float(c[0]), float(slice_index)] for c in out
                ]
                physical_pts = np.array(
                    [dicom_handle.TransformContinuousIndexToPhysicalPoint(p) for p in physical]
                )
                self.contour_dict[slice_index].append(physical_pts)


# ---------------------------------------------------------------------------
# Folder loading helper (runs inside a thread)
# ---------------------------------------------------------------------------
class _DicomFolderLoader:
    """Loads DICOM files from a single directory into shared dictionaries."""

    def __init__(
        self,
        plan_keys: Optional[PyDicomKeys],
        struct_keys: Optional[PyDicomKeys],
        image_keys: Optional[SitkDicomKeys],
        dose_keys: Optional[SitkDicomKeys],
    ) -> None:
        self.plan_keys = plan_keys
        self.struct_keys = struct_keys
        self.image_keys = image_keys
        self.dose_keys = dose_keys

    def load(
        self,
        dicom_path: str,
        images_dict: Dict[str, ImageBase],
        rt_dict: Dict[str, RTBase],
        rd_dict: Dict[str, RDBase],
        rp_dict: Dict[str, PlanBase],
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
        all_series_names: List[str] = []

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
                    _add_rd(img_reader, rd_dict, self.dose_keys)
            else:
                img_reader.Execute()
                _add_image(images_dict, dicom_names, img_reader, dicom_path, self.image_keys)

        # Remaining files (RT structs, plans, etc.)
        rt_files = [f for f in file_list if f not in all_series_names]
        for file_name in rt_files:
            full_path = os.path.join(dicom_path, file_name)
            try:
                ds = dcmread(full_path)
            except Exception:
                logger.warning("Could not read %s", full_path)
                continue
            modality = ds.Modality.lower()
            if "struct" in modality:
                _add_rt(ds, full_path, rt_dict)
            elif "plan" in modality:
                _add_rp(ds, full_path, rp_dict, self.plan_keys)


# ---------------------------------------------------------------------------
# Dictionary population helpers
# ---------------------------------------------------------------------------
def _add_image(
    images_dict: Dict[str, ImageBase],
    dicom_names: List[str],
    reader: sitk.ImageFileReader,
    path: str,
    sitk_string_keys: Optional[SitkDicomKeys] = None,
) -> None:
    uid = reader.GetMetaData("0020|000e")
    if uid not in images_dict:
        img = ImageBase()
        img.load_info(dicom_names, reader, path, sitk_string_keys)
        images_dict[uid] = img


def _add_rt(
    ds: pydicom.Dataset,
    path: str,
    rt_dict: Dict[str, RTBase],
    pydicom_string_keys: Optional[PyDicomKeys] = None,
) -> None:
    try:
        uid = ds.SeriesInstanceUID
        if uid not in rt_dict:
            rt = RTBase()
            rt.load_info(ds, path, pydicom_string_keys)
            rt_dict[uid] = rt
    except Exception:
        logger.warning("Error loading RT from %s", path, exc_info=True)


def _add_rd(
    reader: sitk.ImageFileReader,
    rd_dict: Dict[str, RDBase],
    sitk_string_keys: Optional[SitkDicomKeys] = None,
) -> None:
    try:
        uid = reader.GetMetaData("0020|000e")
        if uid not in rd_dict:
            rd = RDBase()
            rd.load_info(reader, sitk_string_keys)
            rd_dict[uid] = rd
        else:
            rd_dict[uid].add_beam(reader)
    except Exception:
        logger.warning("Error loading RD from %s", reader.GetFileName(), exc_info=True)


def _add_rp(
    ds: pydicom.Dataset,
    path: str,
    rp_dict: Dict[str, PlanBase],
    pydicom_string_keys: Optional[PyDicomKeys] = None,
) -> None:
    try:
        uid = ds.SeriesInstanceUID
        if uid not in rp_dict:
            plan = PlanBase()
            plan.load_info(ds, path, pydicom_string_keys)
            rp_dict[uid] = plan
    except Exception:
        logger.warning("Error loading RP from %s", path, exc_info=True)


def _add_sops(
    reader: sitk.ImageSeriesReader,
    series_dict: Dict[int, ImageBase],
) -> None:
    """Populate SOPInstanceUIDs for the series that *reader* just loaded."""
    uid = reader.GetMetaData(0, "0020|000e")
    for key, entry in series_dict.items():
        if entry.SeriesInstanceUID == uid:
            entry.SOPs = [
                reader.GetMetaData(i, "0008|0018")
                for i in range(len(reader.GetFileNames()))
            ]
            return


# ===========================  MAIN CLASS  ==================================


class DicomReaderWriter:
    """Read DICOM images / RT structures / dose files and write RT predictions.

    This is the primary entry-point for the package.  See the module-level
    docstring and the project README for usage examples.

    Args:
        description: Tag written into output NIfTI filenames.
        Contour_Names: List of ROI names to extract masks for.
        associations: List of :class:`ROIAssociationClass` for name mapping.
        arg_max: If ``True``, collapse the multi-channel mask via argmax.
        verbose: Print / log progress information.
        create_new_RT: Create a fresh RT struct when writing predictions.
        template_dir: Path to a template ``template_RS.dcm``.
        delete_previous_rois: Remove existing ROIs when writing predictions.
        require_all_contours: Require *every* listed contour to be present.
        iteration: Integer tag for NIfTI filename versioning.
        get_dose_output: Also load dose grids when calling
            :meth:`get_images_and_mask`.
        flip_axes: 3-tuple of booleans – axes to flip after loading.
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
    images_dictionary: Dict[str, ImageBase]
    rt_dictionary: Dict[str, RTBase]
    rd_dictionary: Dict[str, RDBase]
    rp_dictionary: Dict[str, PlanBase]
    rois_in_index_dict: Dict[int, List[str]]
    dicom_handle: Optional[sitk.Image]
    dose_handle: Optional[sitk.Image]
    annotation_handle: Optional[sitk.Image]
    all_rois: List[str]
    roi_class_list: List[ROIClass]
    rois_in_loaded_index: List[str]
    indexes_with_contours: List[int]
    roi_groups: Dict[str, List[str]]
    all_RTs: Dict[str, List[str]]
    RTs_with_ROI_Names: Dict[str, List[str]]
    series_instances_dictionary: Dict[int, ImageBase]
    mask_dictionary: Dict[str, sitk.Image]
    mask: Optional[np.ndarray]

    def __init__(
        self,
        description: str = "",
        Contour_Names: Optional[List[str]] = None,
        associations: Optional[List[ROIAssociationClass]] = None,
        arg_max: bool = True,
        verbose: bool = True,
        create_new_RT: bool = True,
        template_dir: Optional[str] = None,
        delete_previous_rois: bool = True,
        require_all_contours: bool = True,
        iteration: int = 0,
        get_dose_output: bool = False,
        flip_axes: Tuple[bool, bool, bool] = (False, False, False),
        index: int = 0,
        series_instances_dictionary: Optional[Dict[int, ImageBase]] = None,
        plan_pydicom_string_keys: Optional[PyDicomKeys] = None,
        struct_pydicom_string_keys: Optional[PyDicomKeys] = None,
        image_sitk_string_keys: Optional[SitkDicomKeys] = None,
        dose_sitk_string_keys: Optional[SitkDicomKeys] = None,
        group_dose_by_frame_of_reference: bool = True,
    ) -> None:
        # Logging setup
        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        # Internal state
        self.roi_class_list: List[ROIClass] = []
        self.dose: Optional[np.ndarray] = None
        self.group_dose_by_frame_of_reference = group_dose_by_frame_of_reference
        self.verbose = verbose
        self.annotation_handle: Optional[sitk.Image] = None
        self.dicom_handle: Optional[sitk.Image] = None
        self.dose_handle: Optional[sitk.Image] = None
        self.rois_in_index_dict: Dict[int, List[str]] = {}
        self.rt_dictionary: Dict[str, RTBase] = {}
        self.mask_dictionary: Dict[str, sitk.Image] = {}
        self._dicom_handle_uid: Optional[str] = None
        self._dicom_info_uid: Optional[str] = None
        self._rs_struct_uid: Optional[str] = None
        self.mask: Optional[np.ndarray] = None
        self._rd_study_instance_uid: Optional[str] = None
        self.index = index
        self.all_RTs: Dict[str, List[str]] = {}
        self.RTs_with_ROI_Names: Dict[str, List[str]] = {}
        self.all_rois: List[str] = []
        self.roi_groups: Dict[str, List[str]] = {}
        self.indexes_with_contours: List[int] = []
        self.plan_pydicom_string_keys = plan_pydicom_string_keys
        self.struct_pydicom_string_keys = struct_pydicom_string_keys
        self.image_sitk_string_keys = image_sitk_string_keys
        self.dose_sitk_string_keys = dose_sitk_string_keys
        self.images_dictionary: Dict[str, ImageBase] = {}
        self.rd_dictionary: Dict[str, RDBase] = {}
        self.rp_dictionary: Dict[str, PlanBase] = {}
        self.series_instances_dictionary: Dict[int, ImageBase] = (
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
        self.Contour_Names: List[str] = [c.lower() for c in Contour_Names] if Contour_Names else []

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
    def RS_struct_uid(self) -> Optional[str]:  # type: ignore[override]
        return self._rs_struct_uid

    @RS_struct_uid.setter
    def RS_struct_uid(self, value: Optional[str]) -> None:
        self._rs_struct_uid = value

    @property
    def dicom_handle_uid(self) -> Optional[str]:  # type: ignore[override]
        return self._dicom_handle_uid

    @dicom_handle_uid.setter
    def dicom_handle_uid(self, value: Optional[str]) -> None:
        self._dicom_handle_uid = value

    @property
    def rd_study_instance_uid(self) -> Optional[str]:
        return self._rd_study_instance_uid

    @rd_study_instance_uid.setter
    def rd_study_instance_uid(self, value: Optional[str]) -> None:
        self._rd_study_instance_uid = value

    @property
    def dicom_info_uid(self) -> Optional[str]:
        return self._dicom_info_uid

    @dicom_info_uid.setter
    def dicom_info_uid(self, value: Optional[str]) -> None:
        self._dicom_info_uid = value

    # -- Index management ---------------------------------------------------

    def set_index(self, index: int) -> None:
        """Select the active series by integer index."""
        self.index = index
        self.rois_in_loaded_index = self.rois_in_index_dict.get(index, [])

    # -- Internal setters (kept with dunder names for compat) ---------------

    def __set_description__(self, description: str) -> None:
        self.description = description

    def __set_iteration__(self, iteration: int = 0) -> None:
        self.iteration = str(iteration)

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

    def _reset_mask(self) -> None:
        self._make_empty_mask()
        self.mask_dictionary = {}

    # Backward compat
    __mask_empty_mask__ = _make_empty_mask
    __reset_mask__ = _reset_mask

    def _reset(self) -> None:
        self._reset_rts()
        self._rd_study_instance_uid = None
        self._dicom_handle_uid = None
        self._dicom_info_uid = None
        self.series_instances_dictionary = {}
        self.rt_dictionary = {}
        self.images_dictionary = {}
        self.mask_dictionary = {}

    __reset__ = _reset

    def _reset_rts(self) -> None:
        self.all_rois = []
        self.roi_class_list = []
        self.roi_groups = {}
        self.indexes_with_contours = []
        self._rs_struct_uid = None
        self.RTs_with_ROI_Names = {}

    __reset_RTs__ = _reset_rts

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
            for img_key, img in self.series_instances_dictionary.items():
                for rt_key, rt_entry in img.RTs.items():
                    if rd.ReferencedStructureSetSOPInstanceUID == rt_entry.SOPInstanceUID:
                        rt_entry.Doses[rd_uid] = rd
                        img.RDs[rd_uid] = rd
                        rd.Grouped = True

        # --- 4. Plans → RT structures ---
        for rp_uid, rp in self.rp_dictionary.items():
            added = False
            ref = rp.ReferencedStructureSetSOPInstanceUID
            for img_key, img in self.series_instances_dictionary.items():
                for rt_key, rt_entry in img.RTs.items():
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
            for img_key, img in self.series_instances_dictionary.items():
                for rp_key, rp_entry in img.RPs.items():
                    if plan_ref == rp_entry.SOPInstanceUID:
                        # Find the associated RT via plan's struct ref
                        rt_sop = rp_entry.ReferencedStructureSetSOPInstanceUID
                        for rt_key, rt_entry in img.RTs.items():
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
                for img_key, img in self.series_instances_dictionary.items():
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

    __compile__ = _compile

    # -- Contour name management -------------------------------------------

    def set_contour_names_and_associations(
        self,
        contour_names: Optional[List[str]] = None,
        associations: Optional[List[ROIAssociationClass]] = None,
        check_contours: bool = True,
    ) -> None:
        """Set ROI names to extract and optional name associations.

        Args:
            contour_names: Canonical ROI names (case-insensitive).
            associations: Name-mapping rules.
            check_contours: Re-scan indexes for contour availability.
        """
        if contour_names is not None:
            self._reset_rts()
            self.Contour_Names = [n.lower() for n in contour_names]

        if associations is not None:
            self.associations = associations
            self.hierarchy: Dict = {}

        if check_contours:
            self._check_all_contours()

        if contour_names is not None or self.associations is not None:
            if self.verbose:
                logger.info("Contour names or associations changed, resetting mask")
            self._reset_mask()

    # Backward-compat alias
    set_contour_names_and_assocations = set_contour_names_and_associations

    def _check_contours_at_index(
        self, index: int, RTs: Optional[Dict[str, RTBase]] = None
    ) -> None:
        """Check which requested contours exist at a given index."""
        self.rois_in_loaded_index = []
        entry = self.series_instances_dictionary[index]
        if entry.path is None:
            return

        if RTs is None:
            RTs = entry.RTs

        true_rois: List[str] = []
        for rt_key, rt in RTs.items():
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

        if index not in self.indexes_with_contours:
            if not lacking:
                self.indexes_with_contours.append(index)
            elif some_exist and not self.require_all_contours:
                self.indexes_with_contours.append(index)

    __check_contours_at_index__ = _check_contours_at_index

    def _check_all_contours(self) -> None:
        self.indexes_with_contours = []
        for index in self.series_instances_dictionary:
            self._check_contours_at_index(index)
            self.rois_in_index_dict[index] = self.rois_in_loaded_index

    __check_if_all_contours_present__ = _check_all_contours

    # -- Public query methods -----------------------------------------------

    def return_rois(self, print_rois: bool = True) -> List[str]:
        """Return all ROI names found across loaded series."""
        if print_rois:
            logger.info("The following ROIs were found:")
            for roi in self.all_rois:
                print(roi)
        return self.all_rois

    def return_found_rois_with_same_code(self, print_rois: bool = True) -> Dict[str, List[str]]:
        """Return ROIs grouped by their DICOM structure code."""
        if print_rois:
            for code, names in self.roi_groups.items():
                print(f"Code {code}: {', '.join(names)}")
        return self.roi_groups

    def return_files_from_UID(self, UID: str) -> List[str]:
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

    def return_files_from_index(self, index: int) -> List[str]:
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

    def return_files_from_patientID(self, patientID: str) -> List[str]:
        """Return all file paths associated with a patient ID."""
        out: List[str] = []
        for index, entry in self.series_instances_dictionary.items():
            if entry.PatientID == patientID:
                out.extend(self.return_files_from_index(index))
        return out

    def where_are_RTs(self, ROIName: str) -> List[str]:
        """Deprecated – use :meth:`where_is_ROI`."""
        logger.warning("where_are_RTs() is deprecated; use where_is_ROI()")
        return self.where_is_ROI(ROIName=ROIName)

    def where_is_ROI(self, ROIName: str) -> List[str]:
        """Return the RT file paths that contain the given ROI name."""
        key = ROIName.lower()
        if key in self.RTs_with_ROI_Names:
            return list(self.RTs_with_ROI_Names[key])
        logger.warning("%s was not found; check spelling or list all ROIs", ROIName)
        return []

    def which_indexes_have_all_rois(self) -> List[int]:
        """Return indexes where all requested ROIs are present."""
        if not self.Contour_Names:
            logger.warning("No contour names set. Use set_contour_names_and_associations().")
            return []
        if self.verbose:
            for idx in self.indexes_with_contours:
                print(f"Index {idx} at {self.series_instances_dictionary[idx].path}")
        return list(self.indexes_with_contours)

    def which_indexes_lack_all_rois(self) -> List[int]:
        """Return indexes where one or more requested ROIs are missing."""
        if not self.Contour_Names:
            logger.warning("No contour names set. Use set_contour_names_and_associations().")
            return []
        lacking = [i for i in self.series_instances_dictionary if i not in self.indexes_with_contours]
        if self.verbose:
            for idx in lacking:
                print(f"Index {idx} at {self.series_instances_dictionary[idx].path}")
        return lacking

    # -- Folder walking -----------------------------------------------------

    def down_folder(self, input_path: PathLike) -> None:
        """Deprecated – use :meth:`walk_through_folders`."""
        logger.warning("down_folder() is deprecated; use walk_through_folders()")
        self.walk_through_folders(input_path)

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
        paths_with_dicom: List[str] = []
        for root, _dirs, files in os.walk(input_path):
            if any(f.lower().endswith(".dcm") for f in files):
                paths_with_dicom.append(root)

        if not paths_with_dicom:
            logger.info("No DICOM files found under %s", input_path)
            return

        loader = _DicomFolderLoader(
            self.plan_pydicom_string_keys,
            self.struct_pydicom_string_keys,
            self.image_sitk_string_keys,
            self.dose_sitk_string_keys,
        )

        pbar = tqdm(total=len(paths_with_dicom), desc="Loading DICOM files")

        # Use thread_count=1 for deterministic/test runs
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
                print(f"Index {key}, description {entry.Description} at {entry.path}")
            print(
                f"{len(self.series_instances_dictionary)} unique series IDs found. "
                "Default is index 0; change with set_index(index)."
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
        """Print every DICOM metadata key/value for the current index."""
        self.load_key_information_only()
        for key in self.image_reader.GetMetaDataKeys():
            print(f"{key} = {self.image_reader.GetMetaData(key)}")

    def return_key_info(self, key: str) -> Optional[str]:
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
        self._reset_mask()

        _add_sops(self.reader, self.series_instances_dictionary)

        if any(self.flip_axes):
            flip_filter = sitk.FlipImageFilter()
            flip_filter.SetFlipAxes(self.flip_axes)
            self.dicom_handle = flip_filter.Execute(self.dicom_handle)

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
        output: Optional[sitk.Image] = None
        filter_rds = len(entry.RDs) > 1

        for rd_uid, rd in entry.RDs.items():
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
            self.structure_references: Dict[int, int] = {}
            self.RS_struct = dcmread(rt.path)
            self._rs_struct_uid = rt.SeriesInstanceUID
            for i, contour_seq in enumerate(self.RS_struct.ROIContourSequence):
                self.structure_references[contour_seq.ReferencedROINumber] = i

    __characterize_RT__ = _characterize_rt

    def _return_mask_for_roi(self, rt: RTBase, roi_name: str) -> np.ndarray:
        self._characterize_rt(rt)
        struct_idx = self.structure_references[rt.ROIs_In_Structure[roi_name].ROINumber]
        return self._contours_to_mask(struct_idx, roi_name)

    __return_mask_for_roi__ = _return_mask_for_roi

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

        for rt_key, rt in entry.RTs.items():
            for roi_name in rt.ROIs_In_Structure:
                true_name = self._resolve_roi_name(roi_name)
                if true_name and true_name in self.Contour_Names:
                    mask = self._return_mask_for_roi(rt, roi_name)
                    ch = self.Contour_Names.index(true_name) + 1
                    self.mask[..., ch] += mask
                    self.mask[self.mask > 1] = 1

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

    def _resolve_roi_name(self, roi_name: str) -> Optional[str]:
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
        """Convert flat contour data to Nx3 matrix of voxel indices."""
        pts = np.asarray(contour_data).reshape(-1, 3)
        return np.array(
            [self.dicom_handle.TransformPhysicalPointToIndex(pts[i]) for i in range(len(pts))]
        )

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
        self, contour_points: np.ndarray, mask: Optional[np.ndarray] = None
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

        if Tag((0x3006, 0x0039)) not in self.RS_struct.keys():
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

    def write_images_annotations(self, out_path: PathLike) -> None:
        """Write the current image and mask to NIfTI files.

        Args:
            out_path: Directory for output files.
        """
        os.makedirs(out_path, exist_ok=True)
        img_path = os.path.join(out_path, f"Overall_Data_{self.description}_{self.iteration}.nii.gz")
        ann_path = os.path.join(out_path, f"Overall_mask_{self.description}_y{self.iteration}.nii.gz")

        handle = self.dicom_handle
        if handle.GetPixelIDTypeAsString().find("32-bit signed integer") != 0:
            handle = sitk.Cast(handle, sitk.sitkFloat32)
        sitk.WriteImage(handle, img_path)

        ann = self.annotation_handle
        ann.SetSpacing(self.dicom_handle.GetSpacing())
        ann.SetOrigin(self.dicom_handle.GetOrigin())
        ann.SetDirection(self.dicom_handle.GetDirection())
        if ann.GetPixelIDTypeAsString().find("int") == -1:
            ann = sitk.Cast(ann, sitk.sitkUInt8)
        sitk.WriteImage(ann, ann_path)

        if self.dose_handle:
            dose_path = os.path.join(out_path, f"Overall_dose_{self.description}_{self.iteration}.nii.gz")
            sitk.WriteImage(self.dose_handle, dose_path)

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
        excel_file: PathLike,
        thread_count: int = _DEFAULT_WORKERS,
    ) -> None:
        """Write all indexed series to NIfTI files in parallel.

        Args:
            out_path: Output directory.
            excel_file: Excel file tracking iterations and volumes.
            thread_count: Number of parallel worker threads.
        """
        os.makedirs(out_path, exist_ok=True)

        if not os.path.exists(excel_file):
            cols = {"PatientID": [], "Path": [], "Iteration": [], "Folder": [],
                    "SeriesInstanceUID": [], "Pixel_Spacing_X": [], "Pixel_Spacing_Y": [],
                    "Slice_Thickness": []}
            for roi in self.Contour_Names:
                cols[f"Volume_{roi} [cc]"] = []
            df = pd.DataFrame(cols)
            df.to_excel(excel_file, index=False)
        else:
            df = pd.read_excel(excel_file, engine="openpyxl")

        # Ensure volume columns exist
        for roi in self.Contour_Names:
            col = f"Volume_{roi} [cc]"
            if col not in df.columns:
                df[col] = np.nan
                df.to_excel(excel_file, index=False)

        # Register new series
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
            df.to_excel(excel_file, index=False)

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
                base.__set_iteration__(iteration)
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

        # Update volume data
        for item in items:
            idx = item[1]
            iteration = item[0]
            tags = self.series_instances_dictionary[idx].additional_tags
            if "Volumes" not in tags:
                continue
            for ri, roi in enumerate(self.Contour_Names):
                col = f"Volume_{roi} [cc]"
                df.loc[df.Iteration == iteration, col] = tags["Volumes"][ri]
        df.to_excel(excel_file, index=False)

    # -- Excel characterisation -------------------------------------------

    def characterize_data_to_excel(
        self,
        wanted_rois: Optional[List[str]] = None,
        excel_path: PathLike = "./Data.xlsx",
    ) -> None:
        """Scan all indexes and record volume data to an Excel file.

        Args:
            wanted_rois: ROI names to evaluate. Defaults to
                :attr:`Contour_Names` or all found ROIs.
            excel_path: Output Excel file path.
        """
        self.verbose = False
        loading_rois = self._resolve_wanted_rois(wanted_rois)
        loading_rois = list(set(loading_rois))

        final = {"PatientID": [], "PixelSpacingX": [], "PixelSpacingY": [],
                 "SliceThickness": [], "zzzRTPath": [], "zzzImagePath": []}
        image_data = {"PatientID": [], "ImagePath": [], "PixelSpacingX": [],
                      "PixelSpacingY": [], "SliceThickness": []}
        temp_assoc: Dict[str, str] = {}
        columns: List[str] = []

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

            for rt_idx, rt_base in entry.RTs.items():
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

        for key, target in temp_assoc.items():
            if target not in final:
                final[target] = [np.nan] * len(final["PatientID"])

        df = pd.DataFrame(final)
        for key, target in temp_assoc.items():
            df[target] = df[f"{key} cc"] + df.fillna(0)[target]
        df = df.reindex(sorted(df.columns), axis=1)
        df_img = pd.DataFrame(image_data)

        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name="ROIs", index=False)
            df_img.to_excel(writer, sheet_name="Images", index=False)

    def _resolve_wanted_rois(self, wanted_rois: Optional[List[str]]) -> List[str]:
        if wanted_rois is None:
            if self.Contour_Names:
                return list(self.Contour_Names)
            return list(self.all_rois)
        result: List[str] = []
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
        ROI_Names: List[str],
        ROI_Types: Optional[List[str]] = None,
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
            for uid_key, rt in entry.RTs.items():
                self.RS_struct = dcmread(rt.path)
                self._rs_struct_uid = entry.SeriesInstanceUID
                break

        prediction_array = np.squeeze(prediction_array)

        # Remove empty channels
        max_per_channel = np.max(prediction_array, axis=tuple(range(prediction_array.ndim - 1)))
        max_per_channel[0] = 1  # Keep background
        active = max_per_channel > 0
        prediction_array = prediction_array[..., active]
        active_names = [n for n, a in zip(ROI_Names, active[1:]) if a]
        dropped = [n for n, a in zip(ROI_Names, active[1:]) if not a]
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

    # Backward compat
    def with_annotations(self, annotations, output_dir, ROI_Names=None):
        logger.warning("with_annotations() is deprecated; use prediction_array_to_RT()")
        self.prediction_array_to_RT(annotations, output_dir, ROI_Names)

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

        for name, roi_type in zip(self.ROI_Names, self.ROI_Types):
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
            contour_dict: Dict[int, List[np.ndarray]] = {}
            maker = _PointOutputMaker(
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

        os.makedirs(self.output_dir, exist_ok=True)
        self.RS_struct.SeriesInstanceUID = pydicom.uid.generate_uid(prefix="1.2.826.0.1.3680043.8.498.")
        out_name = os.path.join(
            self.output_dir,
            f"RS_MRN{self.RS_struct.PatientID}_{self.RS_struct.SeriesInstanceUID}.dcm",
        )
        if os.path.exists(out_name):
            out_name = out_name.replace(".dcm", "1.dcm")

        logger.info("Writing RT structure to %s", self.output_dir)
        pydicom.dcmwrite(out_name, self.RS_struct)
        with open(os.path.join(self.output_dir, "Completed.txt"), "w"):
            pass
        logger.info("Finished writing RT structure")

    mask_to_contours = _mask_to_contours

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
        if ref_key in self.RS_struct.keys():
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

    def rewrite_RT(self, lstRSFile: Optional[PathLike] = None) -> None:
        """Rename ROIs in an existing RT structure file using associations.

        Args:
            lstRSFile: Path to the RT structure file to rewrite.
        """
        if lstRSFile is not None:
            self.RS_struct = dcmread(lstRSFile)

        roi_struct = (
            self.RS_struct.StructureSetROISequence
            if Tag((0x3006, 0x0020)) in self.RS_struct.keys()
            else []
        )
        obs_seq = (
            self.RS_struct.RTROIObservationsSequence
            if Tag((0x3006, 0x0080)) in self.RS_struct.keys()
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
