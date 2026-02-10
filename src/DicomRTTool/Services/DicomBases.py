"""Data models for DICOM RT structures, images, doses, and plans.

Uses dataclasses for clean, type-annotated data containers that replace
the previous class-based approach with explicit ``load_info`` methods.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pydicom
import SimpleITK as sitk
from pydicom.tag import BaseTag, Tag

logger = logging.getLogger(__name__)

# Public type aliases for user-facing key dictionaries
PyDicomKeys = Dict[str, BaseTag]
"""Maps user-chosen names to pydicom Tag objects, e.g. ``{"MyPlan": Tag((0x300a, 0x002))}``."""

SitkDicomKeys = Dict[str, str]
"""Maps user-chosen names to SITK metadata key strings, e.g. ``{"PatientName": "0010|0010"}``."""

PathLike = Union[str, bytes, os.PathLike]

# ---------------------------------------------------------------------------
# Compatibility shim: pydicom renamed ``read_file`` → ``dcmread``
# ---------------------------------------------------------------------------
if hasattr(pydicom, "read_file"):
    dcmread = pydicom.read_file
else:
    dcmread = pydicom.dcmread


# ---------------------------------------------------------------------------
# ROI metadata
# ---------------------------------------------------------------------------
@dataclass
class ROIClass:
    """Metadata for a single ROI found inside an RT Structure Set."""

    ROIName: str = ""
    ROIType: str = ""
    ROINumber: int = 0
    StructureCode: str = ""


# ---------------------------------------------------------------------------
# Base DICOM record
# ---------------------------------------------------------------------------
@dataclass
class DICOMBase:
    """Common fields shared across all DICOM-derived records."""

    PatientID: Optional[str] = None
    SeriesInstanceUID: Optional[str] = None
    SOPInstanceUID: Optional[str] = None
    StudyInstanceUID: Optional[str] = None
    path: Optional[PathLike] = None
    additional_tags: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Radiation Dose
# ---------------------------------------------------------------------------
@dataclass
class RDBase(DICOMBase):
    """Parsed metadata for an RT Dose file."""

    Description: Optional[str] = None
    ReferencedStructureSetSOPInstanceUID: Optional[str] = None
    ReferencedPlanSOPInstanceUID: Optional[str] = None
    ReferencedFrameOfReference: str = ""
    DoseSummationType: str = ""
    DoseType: str = ""
    DoseUnits: str = ""
    Grouped: bool = False
    Dose_Files: List[str] = field(default_factory=list)

    # -- loaders --------------------------------------------------------

    def load_info(
        self,
        sitk_dicom_reader: sitk.ImageFileReader,
        sitk_string_keys: Optional[SitkDicomKeys] = None,
    ) -> None:
        """Populate fields from a SimpleITK reader that has already executed."""
        file_name = sitk_dicom_reader.GetFileName()
        ds = dcmread(file_name)

        self.SeriesInstanceUID = ds.SeriesInstanceUID
        self.DoseType = ds.DoseType
        self.DoseUnits = ds.DoseUnits
        self.DoseSummationType = ds.DoseSummationType
        self.ReferencedFrameOfReference = sitk_dicom_reader.GetMetaData("0020|0052")

        if hasattr(ds, "ReferencedStructureSetSequence"):
            self.ReferencedStructureSetSOPInstanceUID = (
                ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
            )
        if hasattr(ds, "ReferencedRTPlanSequence"):
            self.ReferencedPlanSOPInstanceUID = (
                ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
            )

        self.StudyInstanceUID = sitk_dicom_reader.GetMetaData("0020|000d")
        meta_keys = sitk_dicom_reader.GetMetaDataKeys()
        if "0008|103e" in meta_keys:
            self.Description = sitk_dicom_reader.GetMetaData("0008|103e")

        self.path = file_name
        self.Dose_Files.append(self.path)
        self.SOPInstanceUID = sitk_dicom_reader.GetMetaData("0008|0018")
        self._read_sitk_extra_keys(sitk_dicom_reader, sitk_string_keys)

    def add_beam(self, sitk_dicom_reader: sitk.ImageFileReader) -> None:
        """Add an additional beam dose file if it belongs to this series."""
        file_name = sitk_dicom_reader.GetFileName()
        ds = dcmread(file_name)
        if self.SeriesInstanceUID == ds.SeriesInstanceUID and ds.DoseSummationType == "BEAM":
            self.Dose_Files.append(file_name)

    # -- helpers --------------------------------------------------------

    def _read_sitk_extra_keys(
        self,
        reader: sitk.ImageFileReader,
        keys: Optional[SitkDicomKeys],
    ) -> None:
        if keys is None:
            return
        for name, key in keys.items():
            if key in reader.GetMetaDataKeys():
                try:
                    self.additional_tags[name] = reader.GetMetaData(key)
                except RuntimeError:
                    logger.debug("Could not read SITK key %s", key)


# ---------------------------------------------------------------------------
# RT Plan
# ---------------------------------------------------------------------------
@dataclass
class PlanBase(DICOMBase):
    """Parsed metadata for an RT Plan file."""

    PlanLabel: Optional[str] = None
    PlanName: Optional[str] = None
    ReferencedStructureSetSOPInstanceUID: Optional[str] = None
    ReferencedDoseSOPUID: Optional[str] = None
    StudyDescription: Optional[str] = None
    SeriesDescription: Optional[str] = None

    def load_info(
        self,
        ds: pydicom.Dataset,
        path: PathLike,
        pydicom_string_keys: Optional[PyDicomKeys] = None,
    ) -> None:
        if hasattr(ds, "DoseReferenceSequence"):
            dose_ref = ds.DoseReferenceSequence[0]
            if hasattr(dose_ref, "DoseReferenceUID"):
                self.ReferencedDoseSOPUID = dose_ref.DoseReferenceUID

        if hasattr(ds, "ReferencedStructureSetSequence"):
            self.ReferencedStructureSetSOPInstanceUID = (
                ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
            )

        self.path = path
        self.SOPInstanceUID = ds.SOPInstanceUID
        self.PlanLabel = getattr(ds, "RTPlanLabel", None)
        self.PlanName = getattr(ds, "RTPlanName", None)

        if Tag((0x0008, 0x1030)) in ds.keys():
            self.StudyDescription = ds.StudyDescription
        if Tag((0x0008, 0x103E)) in ds.keys():
            self.SeriesDescription = ds.SeriesDescription

        self._read_pydicom_extra_keys(ds, pydicom_string_keys)

    def _read_pydicom_extra_keys(
        self,
        ds: pydicom.Dataset,
        keys: Optional[PyDicomKeys],
    ) -> None:
        if keys is None:
            return
        for name, key in keys.items():
            if key in ds.keys():
                try:
                    self.additional_tags[name] = ds[key].value
                except (KeyError, AttributeError):
                    logger.debug("Could not read pydicom key %s", key)


# ---------------------------------------------------------------------------
# RT Structure Set
# ---------------------------------------------------------------------------
@dataclass
class RTBase(DICOMBase):
    """Parsed metadata for an RT Structure Set."""

    ROI_Names: List[str] = field(default_factory=list)
    ROIs_In_Structure: Dict[str, ROIClass] = field(default_factory=dict)
    referenced_series_instance_uid: Optional[str] = None
    Plans: Dict[str, PlanBase] = field(default_factory=dict)
    Doses: Dict[str, RDBase] = field(default_factory=dict)
    CodeAssociations: Dict[str, List[str]] = field(default_factory=dict)

    def load_info(
        self,
        ds: pydicom.Dataset,
        path: PathLike,
        pydicom_string_keys: Optional[PyDicomKeys] = None,
    ) -> None:
        self.StudyInstanceUID = ds.StudyInstanceUID
        self.PatientID = getattr(ds, "PatientID", None)

        for ref_frame in ds.ReferencedFrameOfReferenceSequence:
            for study_seq in ref_frame.RTReferencedStudySequence:
                for series_seq in study_seq.RTReferencedSeriesSequence:
                    self._parse_series(ds, series_seq, path, pydicom_string_keys)

    def _parse_series(
        self,
        ds: pydicom.Dataset,
        series_seq: pydicom.Sequence,
        path: PathLike,
        pydicom_string_keys: Optional[PyDicomKeys],
    ) -> None:
        refed_uid = series_seq.SeriesInstanceUID

        roi_structures = ds.StructureSetROISequence if Tag((0x3006, 0x0020)) in ds.keys() else []
        roi_observations = ds.RTROIObservationsSequence if Tag((0x3006, 0x0080)) in ds.keys() else []

        # Build lookup tables from observations
        code_strings: Dict[int, str] = {}
        type_strings: Dict[int, str] = {}
        for obs in roi_observations:
            if Tag((0x3006, 0x0086)) in obs:
                code_strings[obs.ReferencedROINumber] = (
                    obs.RTROIIdentificationCodeSequence[0].CodeValue
                )
            if Tag(0x3006, 0x00A4) in obs:
                type_strings[obs.ReferencedROINumber] = obs.RTROIInterpretedType

        code_associations: Dict[str, List[str]] = {}
        rois: List[str] = []

        for structure in roi_structures:
            roi_name = structure.ROIName.lower()
            roi_number = structure.ROINumber
            rois.append(roi_name)

            new_roi = ROIClass(ROIName=roi_name, ROINumber=roi_number)

            if roi_number in code_strings:
                code = code_strings[roi_number]
                new_roi.StructureCode = code
                code_associations.setdefault(code, [])
                if roi_name not in code_associations[code]:
                    code_associations[code].append(roi_name)

            if roi_number in type_strings:
                new_roi.ROIType = type_strings[roi_number]

            if roi_name not in self.ROIs_In_Structure:
                self.ROIs_In_Structure[roi_name] = new_roi

        self.path = path
        self.ROI_Names = rois
        self.SeriesInstanceUID = refed_uid
        self.SOPInstanceUID = ds.SOPInstanceUID
        self.CodeAssociations = code_associations

        if pydicom_string_keys is not None:
            for name, key in pydicom_string_keys.items():
                if key in ds.keys():
                    try:
                        self.additional_tags[name] = ds[key].value
                    except (KeyError, AttributeError):
                        logger.debug("Could not read pydicom key %s", key)


# ---------------------------------------------------------------------------
# Image Series
# ---------------------------------------------------------------------------
@dataclass
class ImageBase(DICOMBase):
    """Parsed metadata for a DICOM image series."""

    Description: Optional[str] = None
    FrameOfReference: str = ""
    slice_thickness: Optional[float] = None
    pixel_spacing_x: Optional[float] = None
    pixel_spacing_y: Optional[float] = None
    SOPs: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    RTs: Dict[str, RTBase] = field(default_factory=dict)
    RDs: Dict[str, RDBase] = field(default_factory=dict)
    RPs: Dict[str, PlanBase] = field(default_factory=dict)

    def load_info(
        self,
        dicom_names: List[str],
        sitk_dicom_reader: sitk.ImageFileReader,
        path: PathLike,
        sitk_string_keys: Optional[SitkDicomKeys] = None,
    ) -> None:
        self.SeriesInstanceUID = sitk_dicom_reader.GetMetaData("0020|000e")
        self.FrameOfReference = sitk_dicom_reader.GetMetaData("0020|0052")

        patient_id = sitk_dicom_reader.GetMetaData("0010|0020").rstrip()
        self.PatientID = patient_id

        meta_keys = sitk_dicom_reader.GetMetaDataKeys()
        self.files = list(dicom_names)

        if "0008|103e" in meta_keys:
            self.Description = sitk_dicom_reader.GetMetaData("0008|103e")

        if "0028|0030" in meta_keys:
            parts = sitk_dicom_reader.GetMetaData("0028|0030").strip().split("\\")
            self.pixel_spacing_x, self.pixel_spacing_y = float(parts[0]), float(parts[1])

        if "0018|0050" in meta_keys:
            self.slice_thickness = float(sitk_dicom_reader.GetMetaData("0018|0050"))

        self.StudyInstanceUID = sitk_dicom_reader.GetMetaData("0020|000d")
        self.path = path

        if sitk_string_keys is not None:
            for name, key in sitk_string_keys.items():
                if key in meta_keys:
                    try:
                        self.additional_tags[name] = sitk_dicom_reader.GetMetaData(key)
                    except RuntimeError:
                        logger.debug("Could not read SITK key %s", key)
