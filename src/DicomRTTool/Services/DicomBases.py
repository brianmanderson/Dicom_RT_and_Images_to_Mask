import typing
import pydicom
from pydicom.tag import Tag, BaseTag
import SimpleITK as sitk
from typing import List, Dict
import os


PyDicomKeys = Dict[str, BaseTag]  # Example: {"MyNamedRTPlan": Tag((0x300a, 0x002))}
SitkDicomKeys = Dict[str, str]  # Example: {"MyPatientName": "0010|0010"}


class DICOMBase(object):
    PatientID: str = None
    SeriesInstanceUID: str = None
    SOPInstanceUID: str = None
    StudyInstanceUID: str = None
    path: typing.Union[str, bytes, os.PathLike] = None
    additional_tags: Dict

    def load_info(self, *args, **kwargs):
        pass


class RDBase(DICOMBase):
    SOPInstanceUID: str = None
    Description: str = None
    ReferencedStructureSetSOPInstanceUID: str = None
    ReferencedPlanSOPInstanceUID: str = None
    ReferencedFrameOfReference: str
    DoseSummationType: str
    DoseType: str  # GY or RELATIVE
    DoseUnits: str
    Dose_Files: List[str]  # If this is a beam dose, we will have multiple files

    def __init__(self):
        self.additional_tags = dict()
        self.Dose_Files = []

    def load_info(self, sitk_dicom_reader, sitk_string_keys: SitkDicomKeys = None):
        file_name = sitk_dicom_reader.GetFileName()
        ds = pydicom.read_file(file_name)
        self.SeriesInstanceUID = ds.SeriesInstanceUID
        self.DoseType = ds.DoseType
        self.DoseUnits = ds.DoseUnits
        self.DoseSummationType = ds.DoseSummationType
        self.ReferencedFrameOfReference = sitk_dicom_reader.GetMetaData("0020|0052")
        self.ReferencedStructureSetSOPInstanceUID = ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID \
            if "ReferencedStructureSetSequence" in ds.values() else None
        if Tag((0x300a, 0x002)) in ds.keys():
            self.ReferencedPlanSOPInstanceUID = ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
        self.StudyInstanceUID = sitk_dicom_reader.GetMetaData("0020|000d")
        if "0008|103e" in sitk_dicom_reader.GetMetaDataKeys():
            self.Description = sitk_dicom_reader.GetMetaData("0008|103e")
        self.path = sitk_dicom_reader.GetFileName()
        self.Dose_Files.append(self.path)
        self.SOPInstanceUID = sitk_dicom_reader.GetMetaData("0008|0018")
        if sitk_string_keys is not None:
            for string in sitk_string_keys:
                key = sitk_string_keys[string]
                if key in sitk_dicom_reader.GetMetaDataKeys():
                    try:
                        self.additional_tags[string] = sitk_dicom_reader.GetMetaData(key)
                    except:
                        continue

    def add_beam(self, sitk_dicom_reader):
        file_name = sitk_dicom_reader.GetFileName()
        ds = pydicom.read_file(file_name)
        if self.SeriesInstanceUID == ds.SeriesInstanceUID:
            """
            Means these are compatible beams
            """
            if ds.DoseSummationType == "BEAM":
                self.Dose_Files.append(file_name)


class PlanBase(DICOMBase):
    PlanLabel: str
    PlanName: str
    ReferencedStructureSetSOPInstanceUID: str
    ReferencedDoseSOPUID: str
    StudyDescription: str
    SeriesDescription: str

    def __init__(self):
        self.additional_tags = {}

    def load_info(self, ds: pydicom.Dataset, path: typing.Union[str, bytes, os.PathLike],
                  pydicom_string_keys: PyDicomKeys = None):
        refed_structure_uid = ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
        refed_dose_uid = ds.DoseReferenceSequence[0].DoseReferenceUID
        plan_label = None
        plan_name = None
        if Tag((0x300a, 0x002)) in ds.keys():
            plan_label = ds.RTPlanLabel
        if Tag((0x300a, 0x003)) in ds.keys():
            plan_name = ds.RTPlanName
        self.path = path
        self.SOPInstanceUID = ds.SOPInstanceUID
        self.PlanLabel = plan_label
        self.PlanName = plan_name
        self.ReferencedStructureSetSOPInstanceUID = refed_structure_uid
        self.ReferencedDoseSOPUID = refed_dose_uid
        if Tag((0x0008, 0x1030)) in ds.keys():
            self.StudyDescription = ds.StudyDescription
        if Tag((0x0008, 0x103e)) in ds.keys():
            self.SeriesDescription = ds.SeriesDescription
        if pydicom_string_keys is not None:
            for string in pydicom_string_keys:
                key = pydicom_string_keys[string]
                if key in ds.keys():
                    try:
                        self.additional_tags[string] = ds[key].value
                    except:
                        continue


class ROIClass(object):
    ROIName: str
    ROIType: str
    ROINumber: int
    StructureCode: str


class RTBase(DICOMBase):
    ROI_Names: List[str]
    ROIs_In_Structure: Dict[str, ROIClass]
    referenced_series_instance_uid: str
    Plans: Dict[str, PlanBase]
    Doses: Dict[str, RDBase]
    CodeAssociations: Dict[str, List[str]]

    def __init__(self):
        self.Plans = dict()
        self.Doses = dict()
        self.additional_tags = dict()
        self.ROI_Names = []
        self.ROIs_In_Structure = {}

    def load_info(self, ds: pydicom.Dataset, path: typing.Union[str, bytes, os.PathLike],
                  pydicom_string_keys: PyDicomKeys = None):
        self.StudyInstanceUID = ds.StudyInstanceUID
        for referenced_frame_of_reference in ds.ReferencedFrameOfReferenceSequence:
            for referred_study_sequence in referenced_frame_of_reference.RTReferencedStudySequence:
                for referred_series in referred_study_sequence.RTReferencedSeriesSequence:
                    refed_series_instance_uid = referred_series.SeriesInstanceUID
                    if Tag((0x3006, 0x020)) in ds.keys():
                        ROI_Structure = ds.StructureSetROISequence
                    else:
                        ROI_Structure = []
                    if Tag((0x3006, 0x080)) in ds.keys():
                        ROI_Observation = ds.RTROIObservationsSequence
                    else:
                        ROI_Observation = []
                    code_strings = {}
                    type_strings = {}
                    for Observation in ROI_Observation:
                        if Tag((0x3006, 0x086)) in Observation:
                            code_strings[Observation.ReferencedROINumber] = \
                                Observation.RTROIIdentificationCodeSequence[0].CodeValue
                        if (Tag(0x3006, 0x00a4)) in Observation:
                            type_strings[Observation.ReferencedROINumber] = Observation.RTROIInterpretedType
                    roi_structure_code_and_names = {}
                    rois = []
                    for Structures in ROI_Structure:
                        roi_name = Structures.ROIName.lower()
                        rois.append(roi_name)
                        roi_number = Structures.ROINumber
                        new_roi = ROIClass()
                        new_roi.ROIName = roi_name
                        new_roi.ROINumber = roi_number
                        if roi_number in code_strings:
                            structure_code = code_strings[roi_number]
                            new_roi.StructureCode = structure_code
                            if structure_code not in roi_structure_code_and_names:
                                roi_structure_code_and_names[structure_code] = []
                            if roi_name not in roi_structure_code_and_names[structure_code]:
                                roi_structure_code_and_names[structure_code].append(roi_name)
                        if roi_number in type_strings:
                            roi_type = type_strings[roi_number]
                            new_roi.ROIType = roi_type
                        if roi_name not in self.ROIs_In_Structure:
                            self.ROIs_In_Structure[roi_name] = new_roi
                    self.path = path
                    self.ROI_Names = rois
                    self.SeriesInstanceUID = refed_series_instance_uid
                    self.SOPInstanceUID = ds.SOPInstanceUID
                    self.CodeAssociations = roi_structure_code_and_names
                    if pydicom_string_keys is not None:
                        for string in pydicom_string_keys:
                            key = pydicom_string_keys[string]
                            if key in ds.keys():
                                try:
                                    self.additional_tags[string] = ds[key].value
                                except:
                                    continue


class ImageBase(DICOMBase):
    Description: str = None
    FrameOfReference: str
    slice_thickness: float = None
    pixel_spacing_x: float = None
    pixel_spacing_y: float = None
    SOPs: typing.List[str]
    files: typing.List[str]
    RTs: Dict[str, RTBase]
    RDs: Dict[str, RDBase]
    RPs: Dict[str, PlanBase]

    def __init__(self):
        self.RTs = dict()
        self.RPs = dict()
        self.RDs = dict()
        self.additional_tags = dict()

    def load_info(self, dicom_names: typing.List[str], sitk_dicom_reader: sitk.ImageFileReader,
                  path: typing.Union[str, bytes, os.PathLike],
                  sitk_string_keys: SitkDicomKeys = None):
        """
        Args:
            dicom_names:
            sitk_dicom_reader:
            path:
            sitk_string_keys:

        Returns:

        """
        self.SeriesInstanceUID = sitk_dicom_reader.GetMetaData("0020|000e")
        self.FrameOfReference = sitk_dicom_reader.GetMetaData("0020|0052")
        patientID = sitk_dicom_reader.GetMetaData("0010|0020")
        while len(patientID) > 0 and patientID[-1] == ' ':
            patientID = patientID[:-1]
        self.PatientID = patientID
        meta_keys = sitk_dicom_reader.GetMetaDataKeys()
        self.files = dicom_names
        if "0008|103e" in meta_keys:
            self.Description = sitk_dicom_reader.GetMetaData("0008|103e")
        if "0028|0030" in meta_keys:
            pixel_spacing_x, pixel_spacing_y = sitk_dicom_reader.GetMetaData("0028|0030").strip(' ').split('\\')
            self.pixel_spacing_x, self.pixel_spacing_y = float(pixel_spacing_x), float(pixel_spacing_y)
        if "0018|0050" in meta_keys:
            self.slice_thickness = float(sitk_dicom_reader.GetMetaData("0018|0050"))
        self.StudyInstanceUID = sitk_dicom_reader.GetMetaData("0020|000d")
        self.path = path
        if sitk_string_keys is not None:
            for string in sitk_string_keys:
                key = sitk_string_keys[string]
                if key in sitk_dicom_reader.GetMetaDataKeys():
                    try:
                        self.additional_tags[string] = sitk_dicom_reader.GetMetaData(key)
                    except:
                        continue
