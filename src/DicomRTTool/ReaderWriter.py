__author__ = 'Brian M Anderson'

# Created on 12/31/2020
import os
from .Services.DicomBases import ImageBase, RDBase, RTBase, PlanBase, PyDicomKeys, SitkDicomKeys, ROIClass
from .Services.StaticScripts import poly2mask, add_to_mask
from .Viewer import plot_scroll_Image
from NiftiResampler.ResampleTools import ImageResampler
from tqdm import tqdm
import typing
import pydicom
import numpy as np
from pydicom.tag import Tag
import SimpleITK as sitk
from skimage.measure import label, regionprops, find_contours
from threading import Thread
from multiprocessing import cpu_count
from queue import *
import pandas as pd
import copy
from typing import List, Dict


def contour_worker(A):
    q, kwargs = A
    point_maker = PointOutputMakerClass(**kwargs)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            point_maker.make_output(**item)
        q.task_done()


def worker_def(A):
    q, pbar = A
    while True:
        item = q.get()
        if item is None:
            break
        else:
            iteration, index, out_path, key_dict = item
            base_class = DicomReaderWriter(**key_dict)
            try:
                base_class.set_index(index)
                base_class.get_images_and_mask()
                base_class.__set_iteration__(iteration)
                base_class.write_images_annotations(out_path)
            except:
                print('failed on {}'.format(base_class.series_instances_dictionary[index].path))
                fid = open(os.path.join(base_class.series_instances_dictionary[index].path, 'failed.txt'),
                           'w+')
                fid.close()
            pbar.update()
            q.task_done()


def folder_worker(A):
    q, pbar = A
    while True:
        item = q.get()
        if item is None:
            break
        else:
            dicom_path, images_dictionary, rt_dictionary, rd_dictionary, rp_dictionary, verbose, strings = item
            plan_strings, structure_strings, image_strings, dose_strings = strings
            dicom_adder = AddDicomToDictionary(plan_strings, structure_strings, image_strings, dose_strings)
            try:
                if verbose:
                    print('Loading from {}'.format(dicom_path))
                dicom_adder.add_dicom_to_dictionary_from_path(dicom_path=dicom_path,
                                                              images_dictionary=images_dictionary,
                                                              rt_dictionary=rt_dictionary,
                                                              rd_dictionary=rd_dictionary,
                                                              rp_dictionary=rp_dictionary)
            except:
                print('failed on {}'.format(dicom_path))
            pbar.update()
            q.task_done()


class ROIAssociationClass(object):
    def __init__(self, roi_name: str, other_names: List[str]):
        self.roi_name = roi_name.lower()
        self.other_names = list(set([i.lower() for i in other_names]))

    def add_name(self, roi_name: str):
        if roi_name not in self.other_names:
            self.other_names.append(roi_name.lower())


class PointOutputMakerClass(object):
    def __init__(self, image_size_rows: int, image_size_cols: int, PixelSize, contour_dict, RS):
        self.image_size_rows, self.image_size_cols = image_size_rows, image_size_cols
        self.PixelSize = PixelSize
        self.contour_dict = contour_dict
        self.RS = RS

    def make_output(self, annotation, i, dicom_handle):
        self.contour_dict[i] = []
        regions = regionprops(label(annotation))
        for ii in range(len(regions)):
            temp_image = np.zeros([self.image_size_rows, self.image_size_cols])
            data = regions[ii].coords
            rows = []
            cols = []
            for iii in range(len(data)):
                rows.append(data[iii][0])
                cols.append(data[iii][1])
            temp_image[rows, cols] = 1
            contours = find_contours(temp_image, level=0.5, fully_connected='low', positive_orientation='high')
            for contour in contours:
                contour = np.squeeze(contour)
                with np.errstate(divide='ignore'):
                    slope = (contour[1:, 1] - contour[:-1, 1]) / (contour[1:, 0] - contour[:-1, 0])
                slope_index = None
                out_contour = []
                for index in range(len(slope)):
                    if slope[index] != slope_index:
                        out_contour.append(contour[index])
                    slope_index = slope[index]
                contour = [[float(c[1]), float(c[0]), float(i)] for c in out_contour]
                contour = np.asarray([dicom_handle.TransformContinuousIndexToPhysicalPoint(zz) for zz in contour])
                self.contour_dict[i].append(np.asarray(contour))


def add_images_to_dictionary(images_dictionary: Dict[str, ImageBase], dicom_names: typing.List[str],
                             sitk_dicom_reader: sitk.ImageFileReader, path: typing.Union[str, bytes, os.PathLike],
                             sitk_string_keys: SitkDicomKeys = None):
    """
    Args:
        images_dictionary:
        dicom_names:
        sitk_dicom_reader:
        path:
        sitk_string_keys:

    Returns:

    """
    series_instance_uid = sitk_dicom_reader.GetMetaData("0020|000e")
    if series_instance_uid not in images_dictionary:
        new_image = ImageBase()
        new_image.load_info(dicom_names, sitk_dicom_reader, path, sitk_string_keys)
        images_dictionary[series_instance_uid] = new_image


def add_rp_to_dictionary(ds: pydicom.Dataset, path: typing.Union[str, bytes, os.PathLike],
                         rp_dictionary: Dict[str, PlanBase], pydicom_string_keys: PyDicomKeys = None):
    try:
        series_instance_uid = ds.SeriesInstanceUID
        if series_instance_uid not in rp_dictionary:
            new_plan = PlanBase()
            new_plan.load_info(ds, path, pydicom_string_keys)
            rp_dictionary[series_instance_uid] = new_plan
    except:
        print("Had an error loading " + path)


def add_rt_to_dictionary(ds: pydicom.Dataset, path: typing.Union[str, bytes, os.PathLike], rt_dictionary: Dict[str, RTBase],
                         pydicom_string_keys: PyDicomKeys = None):
    """
    Args:
        ds:
        path:
        rt_dictionary:
        pydicom_string_keys:

    Returns:

    """
    try:
        series_instance_uid = ds.SeriesInstanceUID
        if series_instance_uid not in rt_dictionary:
            new_rt = RTBase()
            new_rt.load_info(ds, path, pydicom_string_keys)
            rt_dictionary[series_instance_uid] = new_rt
    except:
        print("Had an error loading " + path)


def add_rd_to_dictionary(sitk_dicom_reader, rd_dictionary: Dict[str, RDBase], sitk_string_keys: SitkDicomKeys = None):
    try:
        series_instance_uid = sitk_dicom_reader.GetMetaData("0020|000e")
        if series_instance_uid not in rd_dictionary:
            new_rd = RDBase()
            new_rd.load_info(sitk_dicom_reader, sitk_string_keys)
            rd_dictionary[series_instance_uid] = new_rd
        else:
            rd_base: RDBase
            rd_base = rd_dictionary[series_instance_uid]
            rd_base.add_beam(sitk_dicom_reader)
    except:
        print("Had an error loading " + sitk_dicom_reader.GetFileName())


def add_sops_to_dictionary(sitk_dicom_reader, series_instances_dictionary: Dict[str, ImageBase]):
    """
    :param sitk_dicom_reader: sitk.ImageSeriesReader()
    :param series_instances_dictionary: dictionary of series instance UIDs
    """
    series_instance_uid = sitk_dicom_reader.GetMetaData(0, "0020|000e")
    keys = []
    series_instance_uids = []
    for key, value in series_instances_dictionary.items():
        keys.append(key)
        series_instance_uids.append(value.SeriesInstanceUID)
    index = keys[series_instance_uids.index(series_instance_uid)]
    sopinstanceuids = [sitk_dicom_reader.GetMetaData(i, "0008|0018") for i in
                       range(len(sitk_dicom_reader.GetFileNames()))]
    series_instances_dictionary[index].SOPs = sopinstanceuids


def return_template_dictionary():
    template_dictionary = ImageBase()
    return template_dictionary


class AddDicomToDictionary(object):
    def __init__(self, plan_pydicom_string_keys: PyDicomKeys = Dict or None,
                 struct_pydicom_string_keys: PyDicomKeys = Dict or None,
                 image_sitk_string_keys: SitkDicomKeys = Dict or None,
                 dose_sitk_string_keys: SitkDicomKeys = Dict or None):
        self.image_reader = sitk.ImageFileReader()
        self.image_reader.LoadPrivateTagsOn()
        self.reader = sitk.ImageSeriesReader()
        self.reader.GlobalWarningDisplayOff()
        self.plan_pydicom_string_keys = plan_pydicom_string_keys
        self.struct_pydicom_string_keys = struct_pydicom_string_keys
        self.image_sitk_string_keys = image_sitk_string_keys
        self.dose_sitk_string_keys = dose_sitk_string_keys

    def add_dicom_to_dictionary_from_path(self, dicom_path, images_dictionary: Dict[str, ImageBase],
                                          rt_dictionary: Dict[str, RTBase],
                                          rd_dictionary: Dict[str, RDBase],
                                          rp_dictionary: Dict[str, PlanBase]):
        file_list = [os.path.join(dicom_path, i) for i in os.listdir(dicom_path) if i.lower().endswith('.dcm')]
        series_ids = self.reader.GetGDCMSeriesIDs(dicom_path)
        all_names = []
        for series_id in series_ids:
            dicom_names = self.reader.GetGDCMSeriesFileNames(dicom_path, series_id)
            all_names += dicom_names
            self.image_reader.SetFileName(dicom_names[0])
            self.image_reader.ReadImageInformation()
            modality = self.image_reader.GetMetaData("0008|0060")
            if modality.lower().find('rtdose') != -1:
                for dicom_name in dicom_names:
                    self.image_reader.SetFileName(dicom_name)
                    self.image_reader.Execute()
                    add_rd_to_dictionary(sitk_dicom_reader=self.image_reader,
                                         rd_dictionary=rd_dictionary, sitk_string_keys=self.dose_sitk_string_keys)
            else:
                self.image_reader.Execute()
                add_images_to_dictionary(images_dictionary=images_dictionary, dicom_names=dicom_names,
                                         sitk_dicom_reader=self.image_reader, path=dicom_path,
                                         sitk_string_keys=self.image_sitk_string_keys)
        rt_files = [file for file in file_list if file not in all_names]
        for lstRSFile in rt_files:
            rt = pydicom.read_file(lstRSFile)
            modality = rt.Modality
            if modality.lower().find('struct') != -1:
                add_rt_to_dictionary(ds=rt, path=lstRSFile, rt_dictionary=rt_dictionary)
            elif modality.lower().find('plan') != -1:
                add_rp_to_dictionary(ds=rt, path=lstRSFile, rp_dictionary=rp_dictionary,
                                     pydicom_string_keys=self.plan_pydicom_string_keys)
        xxx = 1


class DicomReaderWriter(object):
    images_dictionary: Dict[str, ImageBase]
    rt_dictionary: Dict[str, RTBase]
    rd_dictionary: Dict[str, RDBase]
    rp_dictionary: Dict[str, PlanBase]
    rois_in_index_dict: Dict[int, List[str]]  # List of rois at any index
    dicom_handle: sitk.Image or None
    dose_handle: sitk.Image or None
    annotation_handle: sitk.Image or None
    all_rois: List[str]
    roi_class_list: List[ROIClass]
    rois_in_loaded_index: List[str]
    indexes_with_contours: List[int]  # A list of all the indexes which contain the desired contours
    roi_groups: Dict[str, List[str]]  # A dictionary with ROI names grouped by code associations
    all_RTs: Dict[str, List[str]]  # A dictionary of RT being the key, and a list of ROIs in that RT
    RTs_with_ROI_Names: Dict[str, List[str]]  # A dictionary with key being an ROI name, and value being a list of RTs
    series_instances_dictionary = Dict[int, ImageBase]
    mask_dictionary: Dict[str, sitk.Image]
    mask: np.ndarray or None
    group_dose_by_frame_of_reference: bool

    def __init__(self, description='', Contour_Names: List[str] = None, associations: List[ROIAssociationClass] = None,
                 arg_max=True, verbose=True, create_new_RT=True, template_dir=None, delete_previous_rois=True,
                 require_all_contours=True, iteration=0, get_dose_output=False,
                 flip_axes=(False, False, False), index=0, series_instances_dictionary: Dict[int, ImageBase] = None,
                 plan_pydicom_string_keys: PyDicomKeys = None,
                 struct_pydicom_string_keys: PyDicomKeys = None,
                 image_sitk_string_keys: SitkDicomKeys = None,
                 dose_sitk_string_keys: SitkDicomKeys = None, group_dose_by_frame_of_reference=True):
        """
        :param description: string, description information to add to .nii files
        :param delete_previous_rois: delete the previous RTs within the structure when writing out a prediction
        :param Contour_Names: list of contour names
        :param template_dir: default to None, specifies path to template RT structure
        :param arg_max: perform argmax on the mask
        :param create_new_RT: boolean, if the Dicom-RT writer should create a new RT structure
        :param require_all_contours: Boolean, require all contours present when making nifti files?
        :param associations: dictionary of associations {'liver_bma_program_4': 'liver'}
        :param iteration: what iteration for writing .nii files
        :param get_dose_output: boolean, collect dose information
        :param flip_axes: tuple(3), axis that you want to flip, defaults to (False, False, False)
        :param index: index to reference series_instances_dictionary, default 0
        :param series_instances_dictionary: dictionary of series instance UIDs of images and RTs
        :param group_dose_by_frame_of_reference: a boolean, should dose files be associated with images based on the
        frame of reference. This is a last resort if the dose does not reference a structure or plan file.
        """
        self.roi_class_list = []
        self.dose = None
        self.group_dose_by_frame_of_reference = group_dose_by_frame_of_reference
        self.verbose = verbose
        self.annotation_handle = None
        self.dicom_handle = None
        self.dose_handle = None
        self.rois_in_index_dict = {}
        self.rt_dictionary = {}
        self.mask_dictionary = {}
        self.dicom_handle_uid = None
        self.dicom_info_uid = None
        self.RS_struct_uid = None
        self.mask = None
        self.rd_study_instance_uid = None
        self.index = index
        self.all_RTs = {}
        self.RTs_with_ROI_Names = {}
        self.all_rois = []
        self.roi_groups = {}
        self.indexes_with_contours = []
        self.plan_pydicom_string_keys = plan_pydicom_string_keys
        self.struct_pydicom_string_keys = struct_pydicom_string_keys
        self.image_sitk_string_keys = image_sitk_string_keys
        self.dose_sitk_string_keys = dose_sitk_string_keys
        self.images_dictionary = {}
        self.rd_dictionary = {}
        self.rp_dictionary = {}
        if series_instances_dictionary is None:
            series_instances_dictionary = {}
        self.series_instances_dictionary = series_instances_dictionary
        self.get_dose_output = get_dose_output
        self.require_all_contours = require_all_contours
        self.flip_axes = flip_axes
        self.create_new_RT = create_new_RT
        self.arg_max = arg_max
        if template_dir is None or not os.path.exists(template_dir):
            template_dir = os.path.join(os.path.split(__file__)[0], 'template_RS.dcm')
        self.template_dir = template_dir
        self.template = True
        self.delete_previous_rois = delete_previous_rois
        self.associations = associations
        if Contour_Names is None:
            self.Contour_Names = []
        else:
            self.Contour_Names = Contour_Names
        self.__initialize_reader__()
        self.set_contour_names_and_associations(contour_names=Contour_Names, associations=associations,
                                                check_contours=False)
        self.__set_description__(description)
        self.__set_iteration__(iteration)

    def __initialize_reader__(self):
        self.reader = sitk.ImageSeriesReader()
        self.image_reader = sitk.ImageFileReader()
        self.image_reader.LoadPrivateTagsOn()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.reader.SetOutputPixelType(sitk.sitkFloat32)

    def set_index(self, index: int):
        self.index = index
        if self.index in self.rois_in_index_dict:
            self.rois_in_loaded_index = self.rois_in_index_dict[self.index]
        else:
            self.rois_in_loaded_index = []

    def __mask_empty_mask__(self) -> None:
        if self.dicom_handle:
            self.image_size_cols, self.image_size_rows, self.image_size_z = self.dicom_handle.GetSize()
            self.mask = np.zeros(
                [self.dicom_handle.GetSize()[-1], self.image_size_rows, self.image_size_cols, len(self.Contour_Names) + 1],
                dtype=np.int8)
            self.annotation_handle = sitk.GetImageFromArray(self.mask)

    def __reset_mask__(self):
        self.__mask_empty_mask__()
        self.mask_dictionary = {}

    def __reset__(self):
        self.__reset_RTs__()
        self.rd_study_instance_uid = None
        self.dicom_handle_uid = None
        self.dicom_info_uid = None
        self.series_instances_dictionary = {}
        self.rt_dictionary = {}
        self.images_dictionary = {}
        self.mask_dictionary = {}

    def __reset_RTs__(self):
        self.all_rois = []
        self.roi_class_list = []
        self.roi_groups = {}
        self.indexes_with_contours = []
        self.RS_struct_uid = None
        self.RTs_with_ROI_Names = {}

    def __compile__(self):
        """
        The goal of this is to combine image, rt, and dose dictionaries based on the SeriesInstanceUIDs
        """
        if self.verbose:
            print('Compiling dictionaries together...')
        series_instance_uids = []
        for key, value in self.series_instances_dictionary.items():
            series_instance_uids.append(value.SeriesInstanceUID)
        index = 0
        image_keys = list(self.images_dictionary.keys())
        image_keys.sort()
        for series_instance_uid in image_keys:  # Will help keep things in order later
            if series_instance_uid not in series_instance_uids:
                while index in self.series_instances_dictionary:
                    index += 1
                self.series_instances_dictionary[index] = self.images_dictionary[series_instance_uid]
                series_instance_uids.append(series_instance_uid)
        for rt_series_instance_uid in self.rt_dictionary:
            series_instance_uid = self.rt_dictionary[rt_series_instance_uid].SeriesInstanceUID
            rt_dictionary = self.rt_dictionary[rt_series_instance_uid]
            path = rt_dictionary.path
            self.all_RTs[path] = rt_dictionary.ROI_Names
            for roi in rt_dictionary.ROI_Names:
                if roi not in self.RTs_with_ROI_Names:
                    self.RTs_with_ROI_Names[roi] = [path]
                else:
                    self.RTs_with_ROI_Names[roi].append(path)
            if series_instance_uid in series_instance_uids:
                index = series_instance_uids.index(series_instance_uid)
                self.series_instances_dictionary[index].RTs.update({rt_series_instance_uid: self.rt_dictionary[rt_series_instance_uid]})
            else:
                while index in self.series_instances_dictionary:
                    index += 1
                template = return_template_dictionary()
                template.RTs.update({rt_series_instance_uid: self.rt_dictionary[rt_series_instance_uid]})
                self.series_instances_dictionary[index] = template
        for rd_series_instance_uid in self.rd_dictionary:
            struct_ref = self.rd_dictionary[rd_series_instance_uid].ReferencedStructureSetSOPInstanceUID
            if struct_ref is None:
                continue
            for image_series_key in self.series_instances_dictionary:
                rts = self.series_instances_dictionary[image_series_key].RTs
                for rt_key in rts:
                    structure_sop_uid = rts[rt_key].SOPInstanceUID
                    if struct_ref == structure_sop_uid:
                        rts[rt_key].Doses[rd_series_instance_uid] = self.rd_dictionary[rd_series_instance_uid]
                        self.series_instances_dictionary[image_series_key].RDs.update({rd_series_instance_uid:
                                                                                           self.rd_dictionary[rd_series_instance_uid]})
        for rp_series_instance_uid in self.rp_dictionary:
            added = False
            struct_ref = self.rp_dictionary[rp_series_instance_uid].ReferencedStructureSetSOPInstanceUID
            for image_series_key in self.series_instances_dictionary:
                rts = self.series_instances_dictionary[image_series_key].RTs
                for rt_key in rts:
                    structure_sop_uid = rts[rt_key].SOPInstanceUID
                    if struct_ref == structure_sop_uid:
                        rts[rt_key].Plans[rp_series_instance_uid] = self.rp_dictionary[rp_series_instance_uid]
                        self.series_instances_dictionary[image_series_key].RPs.update({rp_series_instance_uid:
                                                                                           self.rp_dictionary[rp_series_instance_uid]})
                        added = True
            if not added:
                while index in self.series_instances_dictionary:
                    index += 1
                template = return_template_dictionary()
                template.RPs.update({rp_series_instance_uid: self.rp_dictionary[rp_series_instance_uid]})
                self.series_instances_dictionary[index] = template
        for rd_series_instance_uid in self.rd_dictionary:
            struct_ref = self.rd_dictionary[rd_series_instance_uid].ReferencedStructureSetSOPInstanceUID
            if struct_ref is not None:
                continue
            plan_ref = self.rd_dictionary[rd_series_instance_uid].ReferencedPlanSOPInstanceUID
            for image_series_key in self.series_instances_dictionary:
                rps = self.series_instances_dictionary[image_series_key].RPs
                rts = self.series_instances_dictionary[image_series_key].RTs
                for rp_key in rps:
                    plan_sop_uid = rps[rp_key].SOPInstanceUID
                    if plan_ref == plan_sop_uid:
                        rt_key_sopinstanceUID = rps[rp_key].ReferencedStructureSetSOPInstanceUID
                        for rt_key in rts:
                            if rts[rt_key].SOPInstanceUID == rt_key_sopinstanceUID:
                                rts[rt_key].Doses[rd_series_instance_uid] = self.rd_dictionary[rd_series_instance_uid]
                        self.series_instances_dictionary[image_series_key].RDs.update({rd_series_instance_uid:
                                                                                           self.rd_dictionary[rd_series_instance_uid]})
        for rd_series_instance_uid in self.rd_dictionary:
            added = False
            dose = self.rd_dictionary[rd_series_instance_uid]
            if self.group_dose_by_frame_of_reference:
                for image_series_key in self.series_instances_dictionary:
                    image = self.series_instances_dictionary[image_series_key]
                    if image.StudyInstanceUID != dose.StudyInstanceUID:
                        continue
                    if image.FrameOfReference == self.rd_dictionary[rd_series_instance_uid].ReferencedFrameOfReference:
                        self.series_instances_dictionary[image_series_key].RDs.update({rd_series_instance_uid: dose})
                        added = True
                        if self.verbose:
                            print(f"Could not associate the dose files {dose.Dose_Files} with a plan or structure.\n"
                                  f"Grouping with images {image.path} based on Frame of Reference UID")
            if not added:
                while index in self.series_instances_dictionary:
                    index += 1
                template = return_template_dictionary()
                template.RDs.update({rd_series_instance_uid: dose})
                self.series_instances_dictionary[index] = template

    def __manual_compile_based_on_folders__(self, reset_series_instances_dict=False):
        """
        The goal of this is to combine image, rt, and dose dictionaries based on folder location
        AKA, if the RT structure and images are in the same folder
        :return:
        """
        print("Don't use this unless you know why you're doing it...")
        if reset_series_instances_dict:
            self.series_instances_dictionary = {}
        if self.verbose:
            print('Compiling dictionaries together...')
        folders = []
        for key, value in self.series_instances_dictionary.items():
            folders.append(value.path)
        index = 0
        image_keys = list(self.images_dictionary.keys())
        image_keys.sort()
        for series_instance_uid in image_keys:  # Will help keep things in order later
            folder = self.images_dictionary[series_instance_uid].path
            if folder not in folders:
                while index in self.series_instances_dictionary:
                    index += 1
                self.series_instances_dictionary[index] = self.images_dictionary[series_instance_uid]
                folders.append(folder)
        for rt_series_instance_uid in self.rt_dictionary:
            rt_path = os.path.split(self.rt_dictionary[rt_series_instance_uid].path)[0]
            rt_dictionary = self.rt_dictionary[rt_series_instance_uid]
            path = rt_dictionary.path
            self.all_RTs[path] = rt_dictionary.ROI_Names
            for roi in rt_dictionary.ROI_Names:
                if roi not in self.RTs_with_ROI_Names:
                    self.RTs_with_ROI_Names[roi] = [path]
                else:
                    self.RTs_with_ROI_Names[roi].append(path)
            if rt_path in folders:
                index = folders.index(rt_path)
                self.series_instances_dictionary[index].RTs.update({rt_series_instance_uid:
                                                                        self.rt_dictionary[rt_series_instance_uid]})
            else:
                while index in self.series_instances_dictionary:
                    index += 1
                template = return_template_dictionary()
                template.RTs.update({rt_series_instance_uid: self.rt_dictionary[rt_series_instance_uid]})
                self.series_instances_dictionary[index] = template
        for rd_series_instance_uid in self.rd_dictionary:
            added = False
            struct_ref = self.rd_dictionary[rd_series_instance_uid].ReferencedStructureSetSOPInstanceUID
            for image_series_key in self.series_instances_dictionary:
                rts = self.series_instances_dictionary[image_series_key].RTs
                for rt_key in rts:
                    structure_sop_uid = rts[rt_key].SOPInstanceUID
                    if struct_ref == structure_sop_uid:
                        rts[rt_key].Doses[rd_series_instance_uid] = self.rd_dictionary[rd_series_instance_uid]
                        self.series_instances_dictionary[image_series_key].RDs.update({rd_series_instance_uid:
                                                                                              self.rd_dictionary[rd_series_instance_uid]})
                    added = True
            if not added:
                while index in self.series_instances_dictionary:
                    index += 1
                template = return_template_dictionary()
                template.RDs.update({rd_series_instance_uid: self.rd_dictionary[rd_series_instance_uid]})
                self.series_instances_dictionary[index] = template
        for rp_series_instance_uid in self.rp_dictionary:
            added = False
            struct_ref = self.rp_dictionary[rp_series_instance_uid].ReferencedStructureSetSOPInstanceUID
            for image_series_key in self.series_instances_dictionary:
                rts = self.series_instances_dictionary[image_series_key].RTs
                for rt_key in rts:
                    structure_sop_uid = rts[rt_key].SOPInstanceUID
                    if struct_ref == structure_sop_uid:
                        rts[rt_key].Plans[rp_series_instance_uid] = self.rp_dictionary[rp_series_instance_uid]
                        self.series_instances_dictionary[image_series_key].RPs.update({rp_series_instance_uid:
                                                                                              self.rp_dictionary[rp_series_instance_uid]})
                    added = True
            if not added:
                while index in self.series_instances_dictionary:
                    index += 1
                template = return_template_dictionary()
                template.RPs.update({rp_series_instance_uid: self.rp_dictionary[rp_series_instance_uid]})
                self.series_instances_dictionary[index] = template
        self.__check_if_all_contours_present__()

    def set_contour_names_and_associations(self, contour_names: List[str] = None,
                                           associations: List[ROIAssociationClass] = None, check_contours=True):
        if contour_names is not None:
            self.__set_contour_names__(contour_names=contour_names)
        if associations is not None:
            self.__set_associations__(associations=associations)
        if check_contours:  # I don't want to run this on the first build
            self.__check_if_all_contours_present__()
        if contour_names is not None or self.associations is not None:
            if self.verbose:
                print("Contour names or associations changed, resetting mask")
            self.__reset_mask__()

    def __set_associations__(self, associations: List[ROIAssociationClass] = None):
        if associations is not None:
            self.associations, self.hierarchy = associations, {}

    def __set_contour_names__(self, contour_names: List[str]):
        self.__reset_RTs__()
        contour_names = [i.lower() for i in contour_names]
        self.Contour_Names = contour_names

    def __set_description__(self, description: str):
        self.description = description

    def __set_iteration__(self, iteration=0):
        self.iteration = str(iteration)

    def __check_contours_at_index__(self, index: int, RTs: List[RTBase] = None) -> None:
        self.rois_in_loaded_index = []
        if self.series_instances_dictionary[index].path is None:
            return
        if RTs is None:
            RTs = self.series_instances_dictionary[index].RTs
        true_rois = []
        for RT_key in RTs:
            RT = RTs[RT_key]
            for code_key in RT.CodeAssociations:
                if code_key not in self.roi_groups:
                    self.roi_groups[code_key] = RT.CodeAssociations[code_key]
                else:
                    self.roi_groups[code_key] = list(set(self.roi_groups[code_key] + RT.CodeAssociations[code_key]))
            for roi in RT.ROIs_In_Structure.values():
                roi_name = roi.ROIName
                if roi_name not in self.RTs_with_ROI_Names:
                    self.RTs_with_ROI_Names[roi.ROIName] = [RT.path]
                elif RT.path not in self.RTs_with_ROI_Names[roi_name]:
                    self.RTs_with_ROI_Names[roi_name].append(RT.path)
                if roi_name not in self.rois_in_loaded_index:
                    self.rois_in_loaded_index.append(roi_name)
                if roi_name not in self.all_rois:
                    self.all_rois.append(roi_name)
                    self.roi_class_list.append(roi)
                if self.Contour_Names:
                    if roi_name in self.Contour_Names:
                        true_rois.append(roi_name)
                    elif self.associations:
                        for association in self.associations:
                            if roi_name in association.other_names:
                                true_rois.append(association.roi_name)
                            elif roi_name in self.Contour_Names:
                                true_rois.append(roi_name)
        all_contours_exist = True
        some_contours_exist = False
        lacking_rois = []
        for roi in self.Contour_Names:
            if roi not in true_rois:
                lacking_rois.append(roi)
            else:
                some_contours_exist = True
        if lacking_rois:
            all_contours_exist = False
            if self.verbose:
                print('Lacking {} in index {}, location {}. Found {}'.format(lacking_rois, index,
                                                                             self.series_instances_dictionary[index].path, self.rois_in_loaded_index))
        if index not in self.indexes_with_contours:
            if all_contours_exist:
                self.indexes_with_contours.append(index)
            elif some_contours_exist and not self.require_all_contours:
                self.indexes_with_contours.append(index)  # Add the index that have at least some of the contours

    def __check_if_all_contours_present__(self):
        self.indexes_with_contours = []
        for index in self.series_instances_dictionary:
            self.__check_contours_at_index__(index)
            self.rois_in_index_dict[index] = self.rois_in_loaded_index

    def return_rois(self, print_rois=True) -> List[str]:
        if print_rois:
            print('The following ROIs were found')
            for roi in self.all_rois:
                print(roi)
        return self.all_rois

    def return_found_rois_with_same_code(self, print_rois=True) -> Dict[str, List[str]]:
        if print_rois:
            print('The following ROIs were found to have the same structure code')
            for code in self.roi_groups:
                print(f"For code {code} we found:")
                for roi in self.roi_groups[code]:
                    print(roi)
        return self.roi_groups

    def return_files_from_UID(self, UID: str) -> List[str]:
        """
        Args:
            UID: A string UID found in images_dictionary.

        Returns:
            file_list: A list of file paths that are associated with that UID, being images, RTs, RDs, and RPs
        """
        out_file_paths = list()
        if UID not in self.images_dictionary:
            print(UID + " Not found in dictionary")
            return out_file_paths
        image_dictionary = self.images_dictionary[UID]
        dicom_path = image_dictionary.path
        image_reader = sitk.ImageFileReader()
        image_reader.LoadPrivateTagsOn()
        reader = sitk.ImageSeriesReader()
        reader.GlobalWarningDisplayOff()
        out_file_paths += reader.GetGDCMSeriesFileNames(dicom_path, UID)
        for structure_key in image_dictionary.RTs:
            out_file_paths += [image_dictionary.RTs[structure_key].path]
        for structure_key in image_dictionary.RDs:
            out_file_paths += [image_dictionary.RDs[structure_key].path]
        return out_file_paths

    def return_files_from_index(self, index: int) -> List[str]:
        """
        Args:
            index: An integer index found in images_dictionary.

        Returns:
            file_list: A list of file paths that are associated with that index, being images, RTs, RDs, and RPs
        """
        out_file_paths = list()
        image_dictionary = self.series_instances_dictionary[index]
        UID = image_dictionary.SeriesInstanceUID
        dicom_path = image_dictionary.path
        image_reader = sitk.ImageFileReader()
        image_reader.LoadPrivateTagsOn()
        reader = sitk.ImageSeriesReader()
        reader.GlobalWarningDisplayOff()
        out_file_paths += reader.GetGDCMSeriesFileNames(dicom_path, UID)
        for structure_key in image_dictionary.RTs:
            out_file_paths += [image_dictionary.RTs[structure_key].path]
        for structure_key in image_dictionary.RPs:
            out_file_paths += [image_dictionary.RPs[structure_key].path]
        for structure_key in image_dictionary.RDs:
            out_file_paths += [image_dictionary.RDs[structure_key].path]
        return out_file_paths

    def return_files_from_patientID(self, patientID: str) -> List[str]:
        """
        Args:
            patientID:

        Returns:

        """
        out_file_paths = list()
        for index in self.series_instances_dictionary:
            if self.series_instances_dictionary[index].PatientID == patientID:
                out_file_paths += self.return_files_from_index(index)
        return out_file_paths

    def where_are_RTs(self, ROIName: str) -> List[str]:
        print('Please move over to using .where_is_ROI(), as this better represents the definition')
        return self.where_is_ROI(ROIName=ROIName)

    def where_is_ROI(self, ROIName: str) -> List[str]:
        out_folders = list()
        if ROIName.lower() in self.RTs_with_ROI_Names:
            print('Contours of {} are located:'.format(ROIName.lower()))
            for path in self.RTs_with_ROI_Names[ROIName.lower()]:
                out_folders.append(path)
                print(path)
        else:
            print('{} was not found within the set, check spelling or list all rois'.format(ROIName))
        return out_folders

    def which_indexes_have_all_rois(self):
        if self.Contour_Names:
            print('The following indexes have all ROIs present')
            for index in self.indexes_with_contours:
                print('Index {}, located at {}'.format(index, self.series_instances_dictionary[index].path))
            print('Finished listing present indexes')
            return self.indexes_with_contours
        else:
            print('You need to first define what ROIs you want, please use'
                  ' .set_contour_names_and_associations()')

    def characterize_data_to_excel(self, wanted_rois: List[str] = None,
                                   excel_path: typing.Union[str, bytes, os.PathLike] = "./Data.xlsx"):
        print("This is going to load every index and record volume data to the excel_path"
              " indicated above. Be aware that this can take some time...")
        self.verbose = False
        print("To prevent annoying messages, verbosity has been turned off...")
        loading_rois = []
        if wanted_rois is None:
            if self.Contour_Names:
                loading_rois = self.Contour_Names
                print("Since no rois were explicitly defined, this will evaluate previously defined Contour Names")
            else:
                print("Since no rois were explicitly defined, this will evaluate all rois")
                loading_rois = self.all_rois
        else:
            for roi in wanted_rois:
                if roi in self.all_rois:
                    loading_rois.append(roi)
                else:
                    if self.associations:
                        for association in self.associations:
                            if association.roi_name == roi:
                                loading_rois += association.other_names
        loading_rois = list(set(loading_rois))
        final_out_dict = {'PatientID': [], 'PixelSpacingX': [], 'PixelSpacingY': [],
                          'SliceThickness': [], 'zzzRTPath': [], 'zzzImagePath': []}
        image_out_dict = {'PatientID': [], 'ImagePath': [], 'PixelSpacingX': [], 'PixelSpacingY': [],
                          'SliceThickness': []}
        temp_associations = {}
        column_names = []
        for roi in loading_rois:
            if self.associations:
                for association in self.associations:
                    if roi in association.other_names:
                        true_name = association.roi_name
                        temp_associations[roi] = true_name
            if roi not in final_out_dict:
                final_out_dict[f"{roi} cc"] = []
                column_names.append(roi)
        """
        Now we load the images/mask, and get volume data
        """
        pbar = tqdm(total=len(self.series_instances_dictionary), desc='Building data...')
        for index in self.series_instances_dictionary:
            pbar.update()
            if self.series_instances_dictionary[index].SeriesInstanceUID is None:  # No image? Move along
                continue
            self.set_index(index)
            has_wanted_roi = False
            for roi in column_names:
                if roi in self.rois_in_loaded_index:
                    has_wanted_roi = True
                    break
            if not has_wanted_roi:
                continue
            image_base = self.series_instances_dictionary[index]
            image_out_dict['PatientID'].append(image_base.PatientID)
            image_out_dict['ImagePath'].append(image_base.path)
            image_out_dict['PixelSpacingX'].append(image_base.pixel_spacing_x)
            image_out_dict['PixelSpacingY'].append(image_base.pixel_spacing_y)
            image_out_dict['SliceThickness'].append(image_base.slice_thickness)
            self.get_images()
            """
            If there is no image set, move along
            """
            dimension = np.prod(self.dicom_handle.GetSpacing())  # Voxel dimensions, in mm
            for rt_index in image_base.RTs:
                rt_base = image_base.RTs[rt_index]
                self.__check_contours_at_index__(index)
                final_out_dict['PatientID'].append(rt_base.PatientID)
                final_out_dict['zzzRTPath'].append(rt_base.path)
                final_out_dict['zzzImagePath'].append(image_base.path)
                final_out_dict['PixelSpacingX'].append(image_base.pixel_spacing_x)
                final_out_dict['PixelSpacingY'].append(image_base.pixel_spacing_y)
                final_out_dict['SliceThickness'].append(image_base.slice_thickness)
                """
                Default values to be nothing, then replace them as they come
                """
                for roi in column_names:
                    final_out_dict[f"{roi} cc"].append(np.nan)
                for roi in column_names:
                    if roi in rt_base.ROI_Names:
                        mask = self.__return_mask_for_roi__(rt_base, roi)
                        volume = np.around(np.sum(mask) * dimension / 1000, 3)  # Volume in cm^3, not mm^3. 3 sig figs
                        final_out_dict[f"{roi} cc"][-1] = volume
        for key in temp_associations.keys():
            if temp_associations[key] not in final_out_dict:
                final_out_dict[temp_associations[key]] = [np.nan for _ in range(len(final_out_dict['PatientID']))]
        df = pd.DataFrame(final_out_dict)
        for key in temp_associations:
            df[temp_associations[key]] = df[f"{key} cc"] + df.fillna(0)[temp_associations[key]]
        df = df.reindex(sorted(df.columns), axis=1)
        df_image = pd.DataFrame(image_out_dict)
        with pd.ExcelWriter(excel_path) as writer:
            # use to_excel function and specify the sheet_name and index
            # to store the dataframe in specified sheet
            df.to_excel(writer, sheet_name="ROIs", index=False)
            df_image.to_excel(writer, sheet_name="Images", index=False)

    def which_indexes_lack_all_rois(self):
        if self.Contour_Names:
            print('The following indexes are lacking all ROIs')
            indexes_lacking_rois = []
            for index in self.series_instances_dictionary:
                if index not in self.indexes_with_contours:
                    indexes_lacking_rois.append(index)
                    print('Index {}, located at '
                          '{}'.format(index, self.series_instances_dictionary[index].path))
            print('Finished listing lacking indexes')
            return indexes_lacking_rois
        else:
            print('You need to first define what ROIs you want, please use'
                  ' .set_contour_names_and_associations(roi_list)')

    def down_folder(self, input_path: typing.Union[str, bytes, os.PathLike]):
        print('Please move from down_folder() to walk_through_folders()')
        self.walk_through_folders(input_path=input_path)

    def walk_through_folders(self, input_path: typing.Union[str, bytes, os.PathLike],
                             thread_count=int(cpu_count() * 0.9 - 1)):
        """
        Iteratively work down paths to find DICOM files, if they are present, add to the series instance UID dictionary
        :param input_path: path to walk
        """
        paths_with_dicom = []
        for root, dirs, files in os.walk(input_path):
            dicom_files = [i for i in files if i.lower().endswith('.dcm')]
            if dicom_files:
                paths_with_dicom.append(root)
                # dicom_adder.add_dicom_to_dictionary_from_path(dicom_path=root, images_dictionary=self.images_dictionary,
                #                                               rt_dictionary=self.rt_dictionary)
        if paths_with_dicom:
            q = Queue(maxsize=thread_count)
            pbar = tqdm(total=len(paths_with_dicom), desc='Loading through DICOM files')
            a = (q, pbar)
            threads = []
            for worker in range(thread_count):
                t = Thread(target=folder_worker, args=(a,))
                t.start()
                threads.append(t)
            for index, path in enumerate(paths_with_dicom):
                item = [path, self.images_dictionary, self.rt_dictionary, self.rd_dictionary, self.rp_dictionary,
                        self.verbose, (self.plan_pydicom_string_keys, self.struct_pydicom_string_keys,
                                       self.image_sitk_string_keys, self.dose_sitk_string_keys)]
                q.put(item)
            for i in range(thread_count):
                q.put(None)
            for t in threads:
                t.join()
            self.__compile__()
        if self.verbose or len(self.series_instances_dictionary) > 1:
            for key in self.series_instances_dictionary:
                print('Index {}, description {} at {}'.format(key,
                                                              self.series_instances_dictionary[key].Description,
                                                              self.series_instances_dictionary[key].path))
            print('{} unique series IDs were found. Default is index 0, to change use '
                  'set_index(index)'.format(len(self.series_instances_dictionary)))
            self.set_index(0)
        self.__check_if_all_contours_present__()
        return None

    def write_parallel(self, out_path: typing.Union[str, bytes, os.PathLike],
                       excel_file: typing.Union[str, bytes, os.PathLike],
                       thread_count=int(cpu_count() * 0.9 - 1)):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(excel_file):
            final_out_dict = {'PatientID': [], 'Path': [], 'Iteration': [], 'Folder': [], 'SeriesInstanceUID': [],
                              'Pixel_Spacing_X': [], 'Pixel_Spacing_Y': [], 'Slice_Thickness': []}
            for roi in self.Contour_Names:
                column_name = 'Volume_{} [cc]'.format(roi)
                final_out_dict[column_name] = []
            df = pd.DataFrame(final_out_dict)
            df.to_excel(excel_file, index=False)
        else:
            df = pd.read_excel(excel_file, engine='openpyxl')
        add_columns = False
        for roi in self.Contour_Names:
            column_name = 'Volume_{} [cc]'.format(roi)
            if column_name not in df.columns:
                df[column_name] = np.nan
                add_columns = True
        if add_columns:
            df.to_excel(excel_file, index=False)
        key_dict = {'series_instances_dictionary': self.series_instances_dictionary, 'associations': self.associations,
                    'arg_max': self.arg_max, 'require_all_contours': self.require_all_contours,
                    'Contour_Names': self.Contour_Names,
                    'description': self.description, 'get_dose_output': self.get_dose_output}
        rewrite_excel = False
        '''
        First, build the excel file that we will use to reference iterations, Series UIDs, and paths
        '''
        for index in self.indexes_with_contours:
            series_instance_uid = self.series_instances_dictionary[index].SeriesInstanceUID
            previous_run = df.loc[df['SeriesInstanceUID'] == series_instance_uid]
            if previous_run.shape[0] == 0:
                rewrite_excel = True
                iteration = 0
                while iteration in df['Iteration'].values:
                    iteration += 1
                temp_dict = {'PatientID': [self.series_instances_dictionary[index].PatientID],
                             'Path': [self.series_instances_dictionary[index].path],
                             'Iteration': [int(iteration)], 'Folder': [None],
                             'SeriesInstanceUID': [series_instance_uid],
                             'Pixel_Spacing_X': [self.series_instances_dictionary[index].pixel_spacing_x],
                             'Pixel_Spacing_Y': [self.series_instances_dictionary[index].pixel_spacing_y],
                             'Slice_Thickness': [self.series_instances_dictionary[index].slice_thickness]}
                temp_df = pd.DataFrame(temp_dict)
                df = df.append(temp_df)
        if rewrite_excel:
            df.to_excel(excel_file, index=False)
        '''
        Next, read through the excel sheet and see if the out paths already exist
        '''
        items = []
        for index in self.indexes_with_contours:
            series_instance_uid = self.series_instances_dictionary[index].SeriesInstanceUID
            previous_run = df.loc[df['SeriesInstanceUID'] == series_instance_uid]
            if previous_run.shape[0] == 0:
                continue
            iteration = int(previous_run['Iteration'].values[0])
            folder = previous_run['Folder'].values[0]
            if pd.isnull(folder):
                folder = None
            write_path = out_path
            if folder is not None:
                write_path = os.path.join(out_path, folder)
            write_image = os.path.join(write_path, 'Overall_Data_{}_{}.nii.gz'.format(self.description, iteration))
            rerun = True
            if os.path.exists(write_image):
                print('Already wrote out index {} at {}'.format(index, write_path))
                rerun = False
                for roi in self.Contour_Names:
                    column_name = 'Volume_{} [cc]'.format(roi)
                    if pd.isnull(previous_run[column_name].values[0]):
                        rerun = True
                        print('Volume for {} was not defined at index {}.. so rerunning'.format(roi, index))
                        break
            if not rerun:
                continue
            item = [iteration, index, write_path, key_dict]
            items.append(item)
        if items:
            q = Queue(maxsize=thread_count)
            pbar = tqdm(total=len(items), desc='Writing nifti files...')
            a = (q, pbar)
            threads = []
            for worker in range(thread_count):
                t = Thread(target=worker_def, args=(a,))
                t.start()
                threads.append(t)
            for item in items:
                q.put(item)
            for i in range(thread_count):
                q.put(None)
            for t in threads:
                t.join()
            """
            Now, take the volumes that have been calculated during this process and add them to the excel sheet
            """
            for item in items:
                index = item[1]
                iteration = item[0]
                if 'Volumes' not in self.series_instances_dictionary[index].additional_tags.keys():
                    continue
                for roi_index, roi in enumerate(self.Contour_Names):
                    column_name = 'Volume_{} [cc]'.format(roi)
                    df.loc[df.Iteration == iteration, column_name] = \
                        self.series_instances_dictionary[index].additional_tags['Volumes'][roi_index]
            df.to_excel(excel_file, index=False)

    def get_images_and_mask(self) -> None:
        if self.index not in self.series_instances_dictionary:
            print("Index is not preset in the dictionary! Set it using set_index(index)")
            return None
        self.get_images()
        self.get_mask()
        if self.get_dose_output:
            self.get_dose()

    def get_all_info(self) -> None:
        """
        Print all the keys and their respective values
        :return:
        """
        self.load_key_information_only()
        for key in self.image_reader.GetMetaDataKeys():
            print("{} is {}".format(key, self.image_reader.GetMetaData(key)))

    def return_key_info(self, key):
        """
        Return the dicom information for a particular key
        Example: "0008|0022" will return the date acquired in YYYYMMDD format
        :param key: dicom key "0008|0022"
        :return: value associated with the key
        """
        self.load_key_information_only()
        if not self.image_reader.HasMetaDataKey(key):
            print("{} is not present in the reader".format(key))
            return None
        return self.image_reader.GetMetaData(key)

    def load_key_information_only(self) -> None:
        if self.index not in self.series_instances_dictionary:
            print('Index is not present in the dictionary! Set it using set_index(index)')
            return None
        index = self.index
        series_instance_uid = self.series_instances_dictionary[index].SeriesInstanceUID
        if self.dicom_info_uid != series_instance_uid:  # Only load if needed
            dicom_names = self.series_instances_dictionary[index].files
            self.image_reader.SetFileName(dicom_names[0])
            self.image_reader.ReadImageInformation()
            self.dicom_info_uid = series_instance_uid

    def get_images(self) -> None:
        if self.index not in self.series_instances_dictionary:
            print('Index is not present in the dictionary! Set it using set_index(index)')
            return None
        index = self.index
        series_instance_uid = self.series_instances_dictionary[index].SeriesInstanceUID
        if series_instance_uid is None:
            print("This index does not have an associated image within the loaded folders")
            return None
        if self.dicom_handle_uid != series_instance_uid:  # Only load if needed
            if self.verbose:
                print('Loading images for {} at \n {}\n'.format(self.series_instances_dictionary[index].Description,
                                                                self.series_instances_dictionary[index].path))
            dicom_names = self.series_instances_dictionary[index].files
            self.ds = pydicom.read_file(dicom_names[0])
            self.reader.SetFileNames(dicom_names)
            self.dicom_handle = self.reader.Execute()
            if self.verbose:
                print("Erasing any previous mask as we load a new new image set")
            self.__reset_mask__()
            add_sops_to_dictionary(sitk_dicom_reader=self.reader,
                                   series_instances_dictionary=self.series_instances_dictionary)
            if max(self.flip_axes):
                flipimagefilter = sitk.FlipImageFilter()
                flipimagefilter.SetFlipAxes(self.flip_axes)
                self.dicom_handle = flipimagefilter.Execute(self.dicom_handle)
            self.ArrayDicom = sitk.GetArrayFromImage(self.dicom_handle)
            self.image_size_cols, self.image_size_rows, self.image_size_z = self.dicom_handle.GetSize()
            self.dicom_handle_uid = series_instance_uid

    def get_dose(self, dose_type="PLAN") -> None:
        """
        :param dose_type: Type of dose to pull, https://dicom.innolitics.com/ciods/rt-dose/rt-dose/3004000a
        Can be "PLAN", "BEAM", etc.
        :return:
        """
        if self.index not in self.series_instances_dictionary:
            print('Index is not present in the dictionary! Set it using set_index(index)')
            return None
        index = self.index
        if self.dicom_handle_uid != self.series_instances_dictionary[index].SeriesInstanceUID:
            print('Loading images for index {}, since mask was requested but image loading was '
                  'previously different\n'.format(index))
            self.get_images()
        if self.rd_study_instance_uid is not None:
            if self.rd_study_instance_uid == self.series_instances_dictionary[index].StudyInstanceUID:  # Already loaded
                return None
        self.rd_study_instance_uid = self.series_instances_dictionary[index].StudyInstanceUID
        radiation_doses = self.series_instances_dictionary[index].RDs
        reader = sitk.ImageFileReader()
        output, spacing, direction, origin = None, None, None, None
        self.dose = None
        resampler = ImageResampler()
        resampled_dose_handle: sitk.Image
        filter_rds = False
        if len(radiation_doses) > 1:
            filter_rds = True
        for rd_series_instance_uid in radiation_doses:
            rd = radiation_doses[rd_series_instance_uid]
            if filter_rds:
                if rd.DoseSummationType != dose_type:
                    if self.verbose:
                        print(f"Found multiple dose types, loading {dose_type}, this can be changed via"
                              f" .get_dose(dose_type='PLAN'), etc.")
                    continue
            for dose_file in rd.Dose_Files:
                reader.SetFileName(dose_file)
                reader.ReadImageInformation()
                dose_handle = reader.Execute()
                resampled_dose_handle = resampler.resample_image(input_image_handle=dose_handle,
                                                                 ref_resampling_handle=self.dicom_handle,
                                                                 interpolator='Linear', empty_value=0)
                resampled_dose_handle = sitk.Cast(resampled_dose_handle, sitk.sitkFloat32)
                scaling_factor = float(reader.GetMetaData("3004|000e"))
                resampled_dose_handle = resampled_dose_handle * scaling_factor
                if output is None:
                    output = resampled_dose_handle
                else:
                    output += resampled_dose_handle
        if output is not None:
            self.dose = sitk.GetArrayFromImage(output)
            self.dose_handle = output

    def __characterize_RT__(self, RT: RTBase):
        if self.RS_struct_uid != RT.SeriesInstanceUID:
            self.structure_references = {}
            self.RS_struct = pydicom.read_file(RT.path)
            self.RS_struct_uid = RT.SeriesInstanceUID
            for contour_number in range(len(self.RS_struct.ROIContourSequence)):
                self.structure_references[
                    self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number

    def __return_mask_for_roi__(self, RT: RTBase, roi_name: str):
        self.__characterize_RT__(RT)
        structure_index = self.structure_references[RT.ROIs_In_Structure[roi_name].ROINumber]
        mask = self.contours_to_mask(structure_index, roi_name)
        return mask

    def get_mask(self) -> None:
        if self.index not in self.series_instances_dictionary:
            print('Index is not present in the dictionary! Set it using set_index(index)')
            return None
        if not self.Contour_Names:
            print('If you want a mask, you need to set the contour names you are looking for, use '
                  'set_contour_names_and_associations(list_of_roi_names).\nIf you just '
                  'want to look at images  use get_images() not get_images_and_mask() or get_mask()')
            return None
        index = self.index
        if self.dicom_handle_uid != self.series_instances_dictionary[index].SeriesInstanceUID:
            print('Loading images for index {}, since mask was requested but image loading was '
                  'previously different\n'.format(index))
            self.get_images()
        RTs = self.series_instances_dictionary[index].RTs
        for RT_key in RTs:
            RT = RTs[RT_key]
            for ROI_Name in RT.ROIs_In_Structure.keys():
                true_name = None
                if ROI_Name.lower() in self.Contour_Names:
                    true_name = ROI_Name.lower()
                else:
                    if self.associations:
                        for association in self.associations:
                            if ROI_Name.lower() in association.other_names:
                                true_name = association.roi_name
                                break  # Found the name we wanted
                if true_name and true_name in self.Contour_Names:
                    mask = self.__return_mask_for_roi__(RT, ROI_Name)
                    self.mask[..., self.Contour_Names.index(true_name) + 1] += mask
                    self.mask[self.mask > 1] = 1
        for true_name in self.Contour_Names:
            mask_img = sitk.GetImageFromArray(self.mask[..., self.Contour_Names.index(true_name) + 1].astype(np.uint8))
            mask_img.SetSpacing(self.dicom_handle.GetSpacing())
            mask_img.SetDirection(self.dicom_handle.GetDirection())
            mask_img.SetOrigin(self.dicom_handle.GetOrigin())
            self.mask_dictionary[true_name] = mask_img
        if self.flip_axes[0]:
            self.mask = self.mask[:, :, ::-1, ...]
        if self.flip_axes[1]:
            self.mask = self.mask[:, ::-1, ...]
        if self.flip_axes[2]:
            self.mask = self.mask[::-1, ...]
        voxel_size = np.prod(self.dicom_handle.GetSpacing())/1000  # volume in cc per voxel
        volumes = np.sum(self.mask[..., 1:], axis=(0, 1, 2)) * voxel_size  # Volume in cc
        self.series_instances_dictionary[index].additional_tags['Volumes'] = volumes
        if self.arg_max:
            self.mask = np.argmax(self.mask, axis=-1)
        self.annotation_handle = sitk.GetImageFromArray(self.mask.astype(np.int8))
        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())
        return None

    def reshape_contour_data(self, as_array: np.array):
        as_array = np.asarray(as_array)
        if as_array.shape[-1] != 3:
            as_array = np.reshape(as_array, [as_array.shape[0] // 3, 3])
        matrix_points = np.asarray([self.dicom_handle.TransformPhysicalPointToIndex(as_array[i])
                                    for i in range(as_array.shape[0])])
        return matrix_points

    def return_mask(self, mask: np.array, matrix_points: np.array, geometric_type: str):
        col_val = matrix_points[:, 0]
        row_val = matrix_points[:, 1]
        z_vals = matrix_points[:, 2]
        if geometric_type != "OPEN_NONPLANAR":
            temp_mask = poly2mask(row_val, col_val, (self.image_size_rows, self.image_size_cols))
            # temp_mask[self.row_val, self.col_val] = 0
            mask[z_vals[0], temp_mask] += 1
        else:
            for point_index in range(len(z_vals) - 1, 0, -1):
                z_start = z_vals[point_index]
                z_stop = z_vals[point_index - 1]
                z_dif = z_stop - z_start
                r_start = row_val[point_index]
                r_stop = row_val[point_index - 1]
                r_dif = r_stop - r_start
                c_start = col_val[point_index]
                c_stop = col_val[point_index - 1]
                c_dif = c_stop - c_start

                step = 1
                if z_dif != 0:
                    r_slope = r_dif / z_dif
                    c_slope = c_dif / z_dif
                    if z_dif < 0:
                        step = -1
                    for z_value in range(z_start, z_stop + step, step):
                        r_value = r_start + r_slope * (z_value - z_start)
                        c_value = c_start + c_slope * (z_value - z_start)
                        add_to_mask(mask=mask, z_value=z_value, r_value=r_value, c_value=c_value)
                if r_dif != 0:
                    c_slope = c_dif / r_dif
                    z_slope = z_dif / r_dif
                    if r_dif < 0:
                        step = -1
                    for r_value in range(r_start, r_stop + step, step):
                        c_value = c_start + c_slope * (r_value - r_start)
                        z_value = z_start + z_slope * (r_value - r_start)
                        add_to_mask(mask=mask, z_value=z_value, r_value=r_value, c_value=c_value)
                if c_dif != 0:
                    r_slope = r_dif / c_dif
                    z_slope = z_dif / c_dif
                    if c_dif < 0:
                        step = -1
                    for c_value in range(c_start, c_stop + step, step):
                        r_value = r_start + r_slope * (c_value - c_start)
                        z_value = z_start + z_slope * (c_value - c_start)
                        add_to_mask(mask=mask, z_value=z_value, r_value=r_value, c_value=c_value)
        return mask

    def contour_points_to_mask(self, contour_points, mask=None):
        if mask is None:
            mask = np.zeros([self.dicom_handle.GetSize()[-1], self.image_size_rows, self.image_size_cols], dtype=np.int8)
        matrix_points = self.reshape_contour_data(contour_points)
        mask = self.return_mask(mask, matrix_points, geometric_type="CLOSED_PLANAR")
        return mask

    def contours_to_mask(self, index: int, true_name: str):
        mask = np.zeros([self.dicom_handle.GetSize()[-1], self.image_size_rows, self.image_size_cols], dtype=np.int8)
        if Tag((0x3006, 0x0039)) in self.RS_struct.keys():
            contour_sequence = self.RS_struct.ROIContourSequence[index]
            if Tag((0x3006, 0x0040)) in contour_sequence:
                contour_data = contour_sequence.ContourSequence
                for i in range(len(contour_data)):
                    matrix_points = self.reshape_contour_data(contour_data[i].ContourData[:])
                    mask = self.return_mask(mask, matrix_points, geometric_type=contour_data[i].ContourGeometricType)
                mask = mask % 2
            else:
                print(f"This structure set had no data present for {true_name}! Returning a blank mask")
        else:
            print("This structure set had no data present! Returning a blank mask")
        return mask

    def use_template(self) -> None:
        self.template = True
        if not self.template_dir:
            self.template_dir = os.path.join('\\\\mymdafiles', 'ro-admin', 'SHARED', 'Radiation physics', 'BMAnderson',
                                             'Auto_Contour_Sites', 'template_RS.dcm')
            if not os.path.exists(self.template_dir):
                self.template_dir = os.path.join('..', '..', 'Shared_Drive', 'Auto_Contour_Sites', 'template_RS.dcm')
        self.key_list = self.template_dir.replace('template_RS.dcm', 'key_list.txt')
        self.RS_struct = pydicom.read_file(self.template_dir)
        print('Running off a template')
        self.change_template()

    def write_images_annotations(self, out_path: typing.Union[str, bytes, os.PathLike]) -> None:
        image_path = os.path.join(out_path, 'Overall_Data_{}_{}.nii.gz'.format(self.description, self.iteration))
        annotation_path = os.path.join(out_path, 'Overall_mask_{}_y{}.nii.gz'.format(self.description, self.iteration))
        pixel_id = self.dicom_handle.GetPixelIDTypeAsString()
        if pixel_id.find('32-bit signed integer') != 0:
            self.dicom_handle = sitk.Cast(self.dicom_handle, sitk.sitkFloat32)
        sitk.WriteImage(self.dicom_handle, image_path)

        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())
        pixel_id = self.annotation_handle.GetPixelIDTypeAsString()
        if pixel_id.find('int') == -1:
            self.annotation_handle = sitk.Cast(self.annotation_handle, sitk.sitkUInt8)
        sitk.WriteImage(self.annotation_handle, annotation_path)
        if self.dose_handle:
            dose_path = os.path.join(out_path, 'Overall_dose_{}_{}.nii.gz'.format(self.description, self.iteration))
            sitk.WriteImage(self.dose_handle, dose_path)
        fid = open(os.path.join(self.series_instances_dictionary[self.index].path,
                                '{}_Iteration_{}.txt'.format(self.description, self.iteration)), 'w+')
        fid.close()

    def prediction_array_to_RT(self, prediction_array: np.array, output_dir: typing.Union[str, bytes, os.PathLike],
                               ROI_Names: List[str], ROI_Types: List[str] = None) -> None:
        """
        :param prediction_array: numpy array of prediction, expected shape is [#Images, Rows, Cols, #Classes + 1]
        :param output_dir: directory to pass RT structure to
        :param ROI_Names: list of ROI names equal to the number of classes
        :return:
        """
        if ROI_Names is None:
            print("You need to provide ROI_Names")
            return None
        if prediction_array.shape[-1] != (len(ROI_Names) + 1):
            print("Your last dimension of prediction array should be equal  to the number or ROI_names minus 1,"
                  "channel. 0 is for background")
            return None
        if self.index not in self.series_instances_dictionary:
            print("Index is not present in the dictionary! Set it using set_index(index)")
            return None
        index = self.index
        if self.dicom_handle_uid != self.series_instances_dictionary[index].SeriesInstanceUID:
            self.get_images()
        self.SOPInstanceUIDs = self.series_instances_dictionary[index].SOPs
        if self.create_new_RT or len(self.series_instances_dictionary[index].RTs) == 0:
            self.use_template()
        elif self.RS_struct_uid != self.series_instances_dictionary[index].SeriesInstanceUID:
            rt_structures = self.series_instances_dictionary[index].RTs
            for uid_key in rt_structures:
                self.RS_struct = pydicom.read_file(rt_structures[uid_key].path)
                self.RS_struct_uid = self.series_instances_dictionary[index].SeriesInstanceUID
                break

        prediction_array = np.squeeze(prediction_array)
        contour_values = np.max(prediction_array, axis=0)  # See what the maximum value is across the prediction array
        while len(contour_values.shape) > 1:
            contour_values = np.max(contour_values, axis=0)
        contour_values[0] = 1  # Keep background
        prediction_array = prediction_array[..., contour_values == 1]
        contour_values = contour_values[1:]
        not_contained = list(np.asarray(ROI_Names)[contour_values == 0])
        ROI_Names = list(np.asarray(ROI_Names)[contour_values == 1])
        if not_contained:
            print('RT Structure not made for ROIs {}, given prediction_array had no mask'.format(not_contained))
        self.image_size_z, self.image_size_rows, self.image_size_cols = prediction_array.shape[:3]
        self.ROI_Names = ROI_Names
        if ROI_Types is None:
            self.ROI_Types = ["ORGAN" for _ in ROI_Names]
        else:
            self.ROI_Types = ROI_Types
        self.output_dir = output_dir
        if len(prediction_array.shape) == 3:
            prediction_array = np.expand_dims(prediction_array, axis=-1)
        if self.flip_axes[0]:
            prediction_array = prediction_array[:, :, ::-1, ...]
        if self.flip_axes[1]:
            prediction_array = prediction_array[:, ::-1, ...]
        if self.flip_axes[2]:
            prediction_array = prediction_array[::-1, ...]
        self.annotations = prediction_array
        self.mask_to_contours()

    def with_annotations(self, annotations: np.array, output_dir: typing.Union[str, bytes, os.PathLike],
                         ROI_Names=None) -> None:
        print('Please move over to using prediction_array_to_RT')
        self.prediction_array_to_RT(prediction_array=annotations, output_dir=output_dir, ROI_Names=ROI_Names)

    def mask_to_contours(self) -> None:
        self.PixelSize = self.dicom_handle.GetSpacing()
        current_names = []
        for names in self.RS_struct.StructureSetROISequence:
            current_names.append(names.ROIName)
        Contour_Key = {}
        xxx = 1
        for name in self.ROI_Names:
            Contour_Key[name] = xxx
            xxx += 1
        base_annotations = copy.deepcopy(self.annotations)
        temp_color_list = []
        color_list = [[128, 0, 0], [170, 110, 40], [0, 128, 128], [0, 0, 128], [230, 25, 75], [225, 225, 25],
                      [0, 130, 200], [145, 30, 180],
                      [255, 255, 255]]
        self.struct_index = 0
        new_roi_number = 1000
        for Name, ROI_Type in zip(self.ROI_Names, self.ROI_Types):
            new_roi_number -= 1
            if not temp_color_list:
                temp_color_list = copy.deepcopy(color_list)
            color_int = np.random.randint(len(temp_color_list))
            print('Writing data for ' + Name)
            annotations = copy.deepcopy(base_annotations[:, :, :, int(self.ROI_Names.index(Name) + 1)])
            annotations = annotations.astype('int')

            make_new = 1
            allow_slip_in = True
            if (Name not in current_names and allow_slip_in) or self.delete_previous_rois:
                self.RS_struct.StructureSetROISequence.insert(0,
                                                              copy.deepcopy(self.RS_struct.StructureSetROISequence[0]))
            else:
                print('Prediction ROI {} is already within RT structure'.format(Name))
                continue
            self.RS_struct.StructureSetROISequence[self.struct_index].ROINumber = new_roi_number
            self.RS_struct.StructureSetROISequence[self.struct_index].ReferencedFrameOfReferenceUID = \
                self.ds.FrameOfReferenceUID
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIName = Name
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIVolume = 0
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIGenerationAlgorithm = 'SEMIAUTOMATIC'
            if make_new == 1:
                self.RS_struct.RTROIObservationsSequence.insert(0,
                                                                copy.deepcopy(
                                                                    self.RS_struct.RTROIObservationsSequence[0]))
                if 'MaterialID' in self.RS_struct.RTROIObservationsSequence[self.struct_index]:
                    del self.RS_struct.RTROIObservationsSequence[self.struct_index].MaterialID
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ObservationNumber = new_roi_number
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ReferencedROINumber = new_roi_number
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ROIObservationLabel = Name

            self.RS_struct.RTROIObservationsSequence[self.struct_index].RTROIInterpretedType = ROI_Type

            if make_new == 1:
                self.RS_struct.ROIContourSequence.insert(0, copy.deepcopy(self.RS_struct.ROIContourSequence[0]))
            self.RS_struct.ROIContourSequence[self.struct_index].ReferencedROINumber = new_roi_number
            del self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[1:]
            self.RS_struct.ROIContourSequence[self.struct_index].ROIDisplayColor = temp_color_list[color_int]
            del temp_color_list[color_int]
            thread_count = int(cpu_count() * 0.9 - 1)
            contour_dict = {}
            q = Queue(maxsize=thread_count)
            threads = []
            kwargs = {'image_size_rows': self.image_size_rows, 'image_size_cols': self.image_size_cols,
                      'PixelSize': self.PixelSize, 'contour_dict': contour_dict, 'RS': self.RS_struct}

            a = [q, kwargs]
            # pointer_class = PointOutputMakerClass(**kwargs)
            for worker in range(thread_count):
                t = Thread(target=contour_worker, args=(a,))
                t.start()
                threads.append(t)
            contour_num = 0
            if np.max(annotations) > 0:  # If we have an annotation, write it
                image_locations = np.max(annotations, axis=(1, 2))
                indexes = np.where(image_locations > 0)[0]
                for index in indexes:
                    item = {'annotation': annotations[index], 'i': index, 'dicom_handle': self.dicom_handle}
                    # pointer_class.make_output(**item)
                    q.put(item)
                for i in range(thread_count):
                    q.put(None)
                for t in threads:
                    t.join()
                for i in contour_dict.keys():
                    for points in contour_dict[i]:
                        output = np.asarray(points).flatten('C')
                        if contour_num > 0:
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence.append(
                                copy.deepcopy(
                                    self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[0]))
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].ContourNumber = str(contour_num)
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].ContourGeometricType = 'CLOSED_PLANAR'
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].ContourImageSequence[0].ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].ContourData = list(output)
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].NumberOfContourPoints = len(output) // 3
                        contour_num += 1
        self.RS_struct.SOPInstanceUID += '.' + str(np.random.randint(999))
        if self.template or self.delete_previous_rois:
            for i in range(len(self.RS_struct.StructureSetROISequence), len(self.ROI_Names), -1):
                del self.RS_struct.StructureSetROISequence[-1]
            for i in range(len(self.RS_struct.RTROIObservationsSequence), len(self.ROI_Names), -1):
                del self.RS_struct.RTROIObservationsSequence[-1]
            for i in range(len(self.RS_struct.ROIContourSequence), len(self.ROI_Names), -1):
                del self.RS_struct.ROIContourSequence[-1]
            for i in range(len(self.RS_struct.StructureSetROISequence)):
                self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
                self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
                self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.RS_struct.SeriesInstanceUID = pydicom.uid.generate_uid(prefix='1.2.826.0.1.3680043.8.498.')
        out_name = os.path.join(self.output_dir,
                                'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '.dcm')
        if os.path.exists(out_name):
            out_name = os.path.join(self.output_dir,
                                    'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '1.dcm')
        print('Writing out data...{}'.format(self.output_dir))
        pydicom.write_file(out_name, self.RS_struct)
        fid = open(os.path.join(self.output_dir, 'Completed.txt'), 'w+')
        fid.close()
        print('Finished!')
        return None

    def change_template(self):
        keys = self.RS_struct.keys()
        ref_key = Tag((0x3006), (0x0010))
        if ref_key in keys:
            self.RS_struct[ref_key]._value[0].FrameOfReferenceUID = self.ds.FrameOfReferenceUID
            self.RS_struct[ref_key]._value[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = self.ds.StudyInstanceUID
            self.RS_struct[ref_key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].SeriesInstanceUID = self.ds.SeriesInstanceUID
            for i in range(len(self.RS_struct[ref_key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                                   0].ContourImageSequence) - 1):
                del self.RS_struct[ref_key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                    0].ContourImageSequence[-1]
            fill_segment = copy.deepcopy(
                self.RS_struct[ref_key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                    0].ContourImageSequence[0])
            for i in range(len(self.SOPInstanceUIDs)):
                temp_segment = copy.deepcopy(fill_segment)
                temp_segment.ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
                self.RS_struct[ref_key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                    0].ContourImageSequence.append(temp_segment)
            del self.RS_struct[ref_key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[0]

        new_keys = open(self.key_list)
        keys = {}
        i = 0
        for line in new_keys:
            keys[i] = line.strip('\n').split(',')
            i += 1
        new_keys.close()
        for index in keys.keys():
            new_key = keys[index]
            try:
                self.RS_struct[new_key[0], new_key[1]] = self.ds[[new_key[0], new_key[1]]]
            except:
                continue
        return None

    def rewrite_RT(self, lstRSFile: typing.Union[str, bytes, os.PathLike] = None):
        if lstRSFile is not None:
            self.RS_struct = pydicom.read_file(lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        if Tag((0x3006, 0x080)) in self.RS_struct.keys():
            self.Observation_Sequence = self.RS_struct.RTROIObservationsSequence
        else:
            self.Observation_Sequence = []
        self.rois_in_loaded_index = []
        for i, Structures in enumerate(self.ROI_Structure):
            if Structures.ROIName in self.associations:
                new_name = self.associations[Structures.ROIName]
                self.RS_struct.StructureSetROISequence[i].ROIName = new_name
            self.rois_in_loaded_index.append(self.RS_struct.StructureSetROISequence[i].ROIName)
        for i, ObsSequence in enumerate(self.Observation_Sequence):
            if ObsSequence.ROIObservationLabel in self.associations:
                new_name = self.associations[ObsSequence.ROIObservationLabel]
                self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel = new_name
        self.RS_struct.save_as(lstRSFile)


if __name__ == '__main__':
    pass
