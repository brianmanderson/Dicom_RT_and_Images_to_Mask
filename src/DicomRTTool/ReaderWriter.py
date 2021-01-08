__author__ = 'Brian M Anderson'

# Created on 12/31/2020
import os
import pydicom
import numpy as np
from pydicom.tag import Tag
import SimpleITK as sitk
from skimage import draw
from skimage.measure import label, regionprops, find_contours
from threading import Thread
from multiprocessing import cpu_count
from queue import *
import pandas as pd
import copy
from .Viewer import plot_scroll_Image


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
    q = A[0]
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
                print('failed on {}'.format(base_class.series_instances_dictionary[index]['Image_Path']))
            q.task_done()


def folder_worker(A):
    q = A[0]
    while True:
        item = q.get()
        if item is None:
            break
        else:
            dicom_adder = AddDicomToDictionary()
            dicom_path, images_dictionary, rt_dictionary = item
            try:
                print(dicom_path)
                dicom_adder.add_dicom_to_dictionary_from_path(dicom_path=dicom_path,
                                                              images_dictionary=images_dictionary,
                                                              rt_dictionary=rt_dictionary)
            except:
                print('failed on {}'.format(dicom_path))
            q.task_done()


class PointOutputMakerClass(object):
    def __init__(self, image_size_rows, image_size_cols, PixelSize, contour_dict, RS):
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


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def add_images_to_dictionary(images_dictionary, sitk_dicom_reader, path):
    """
    :param images_dictionary: dictionary of series instance UIDs for images
    :param sitk_dicom_reader: sitk.ImageFileReader()
    :param path: path to the images or structure in question
    """
    series_instance_uid = sitk_dicom_reader.GetMetaData("0020|000e")
    if series_instance_uid not in images_dictionary:
        patientID = sitk_dicom_reader.GetMetaData("0010|0020")[:-1]
        description = sitk_dicom_reader.GetMetaData("0008|103e")
        temp_dict = {'PatientID': patientID, 'SeriesInstanceUID': series_instance_uid,
                     'Image_Path': path, 'Description': description, 'RTs': {}, 'RDs': {}}
        images_dictionary[series_instance_uid] = temp_dict


def add_rt_to_dictionary(ds, path, rt_dictionary):
    """
    :param ds: pydicom data structure
    :param path: path to the images or structure in question
    """
    series_instance_uid = ds.SeriesInstanceUID
    if series_instance_uid not in rt_dictionary:
        for referenced_frame_of_reference in ds.ReferencedFrameOfReferenceSequence:
            for referred_study_sequence in referenced_frame_of_reference.RTReferencedStudySequence:
                for referred_series in referred_study_sequence.RTReferencedSeriesSequence:
                    refed_series_instance_uid = referred_series.SeriesInstanceUID
                    if Tag((0x3006, 0x020)) in ds.keys():
                        ROI_Structure = ds.StructureSetROISequence
                    else:
                        ROI_Structure = []
                    rois_in_structure = {}
                    rois = []
                    for Structures in ROI_Structure:
                        rois.append(Structures.ROIName.lower())
                        if Structures.ROIName not in rois_in_structure:
                            rois_in_structure[Structures.ROIName] = Structures.ROINumber
                    temp_dict = {'Path': path, 'ROI_Names': rois, 'ROIs_in_structure': rois_in_structure,
                                 'SeriesInstanceUID': refed_series_instance_uid}
                    rt_dictionary[series_instance_uid] = temp_dict


def add_sops_to_dictionary(sitk_dicom_reader, series_instances_dictionary):
    """
    :param sitk_dicom_reader: sitk.ImageSeriesReader()
    :param series_instances_dictionary: dictionary of series instance UIDs
    """
    series_instance_uid = sitk_dicom_reader.GetMetaData(0, "0020|000e")
    keys = []
    series_instance_uids = []
    for key, value in series_instances_dictionary.items():
        keys.append(key)
        series_instance_uids.append(value['SeriesInstanceUID'])
    index = keys[series_instance_uids.index(series_instance_uid)]
    sopinstanceuids = [sitk_dicom_reader.GetMetaData(i, "0008|0018") for i in
                       range(len(sitk_dicom_reader.GetFileNames()))]
    temp_dict = {'SOP_Instance_UIDs': sopinstanceuids}
    series_instances_dictionary[index].update(temp_dict)


def return_template_dictionary():
    template_dictionary = {'Image_Path': None, 'PatientID': None, 'RTs': {}, 'Description': None,
                           'SOP_Instance_UIDs': None, 'SeriesInstanceUID': None}
    return template_dictionary


class AddDicomToDictionary(object):
    def __init__(self):
        self.image_reader = sitk.ImageFileReader()
        self.image_reader.LoadPrivateTagsOn()
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()

    def add_dicom_to_dictionary_from_path(self, dicom_path, images_dictionary, rt_dictionary):
        fileList = []
        for dirName, dirs, fileList in os.walk(dicom_path):
            break
        fileList = [i for i in fileList if i.find('.dcm') != -1]
        series_ids = self.reader.GetGDCMSeriesIDs(dicom_path)
        all_names = []
        for series_id in series_ids:
            dicom_names = self.reader.GetGDCMSeriesFileNames(dicom_path, series_id)
            all_names += dicom_names
            self.image_reader.SetFileName(dicom_names[0])
            self.image_reader.Execute()
            add_images_to_dictionary(images_dictionary=images_dictionary,
                                     sitk_dicom_reader=self.image_reader, path=dicom_path)
        RT_Files = [os.path.join(dicom_path, file) for file in fileList if file not in all_names]
        for lstRSFile in RT_Files:
            rt = pydicom.read_file(lstRSFile)
            modality = rt.Modality
            if modality.lower().find('struct') != -1:
                add_rt_to_dictionary(ds=rt, path=lstRSFile, rt_dictionary=rt_dictionary)


class DicomReaderWriter:
    def __init__(self, description='', rewrite_RT_file=False, delete_previous_rois=True, Contour_Names=[],
                 verbose=True, template_dir=None, arg_max=True, create_new_RT=True, require_all_contours=True,
                 associations={}, iteration=0, get_dose_output=False, flip_axes=(False, False, False), index=0,
                 series_instances_dictionary={}):
        """
        :param description: string, description information to add to .nii files
        :param rewrite_RT_file: Boolean, should we re-write the RT structure
        :param delete_previous_rois: delete the previous RTs within the structure
        :param Contour_Names: list of contour names
        :param template_dir: default to None, specifies path to template RT structure
        :param arg_max: perform argmax on the mask
        :param require_all_contours: Boolean, require all contours present when making nifti files?
        :param associations: dictionary of associations {'liver_bma_program_4': 'liver'}
        :param iteration: what iteration for writing .nii files
        :param get_dose_output: boolean, collect dose information
        :param flip_axes: tuple(3), axis that you want to flip, defaults to (False, False, False)
        :param series_instances_dictionary: dictionary of series instance UIDs of images and RTs
        """
        self.rt_dictionary = {}
        self.images_dictionary = {}
        self.series_instances_dictionary = series_instances_dictionary
        self.get_dose_output = get_dose_output
        self.require_all_contours = require_all_contours
        self.flip_axes = flip_axes
        self.create_new_RT = create_new_RT
        self.associations = associations
        self.Contour_Names = Contour_Names
        self.set_contour_names_and_associations(Contour_Names=Contour_Names, associations=associations)
        self.__set_description__(description)
        self.__set_iteration__(iteration)
        self.arg_max = arg_max
        self.rewrite_RT_file = rewrite_RT_file
        self.dose_handles = []
        if template_dir is None or not os.path.exists(template_dir):
            template_dir = os.path.join(os.path.split(__file__)[0], 'template_RS.dcm')
        self.template_dir = template_dir
        self.template = True
        self.delete_previous_rois = delete_previous_rois
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.verbose = verbose
        self.dicom_handle_uid = None
        self.RS_struct_uid = None
        self.index = index
        self.all_RTs = {}
        self.RTs_with_ROI_Names = {}
        self.all_rois = []
        self.indexes_with_contours = []

    def set_index(self, index):
        self.index = index

    def __compile__(self):
        """
        The goal of this is to combine image, rt, and dose dictionaries based on the SeriesInstanceUIDs
        """
        series_instance_uids = []
        for key, value in self.series_instances_dictionary.items():
            series_instance_uids.append(value['SeriesInstanceUID'])
        index = 0
        for series_instance_uid in self.images_dictionary:
            if series_instance_uid not in series_instance_uids:
                while index in self.series_instances_dictionary:
                    index += 1
                self.series_instances_dictionary[index] = self.images_dictionary[series_instance_uid]
                series_instance_uids.append(series_instance_uid)
        for rt_series_instance_uid in self.rt_dictionary:
            series_instance_uid = self.rt_dictionary[rt_series_instance_uid]['SeriesInstanceUID']
            rt_dictionary = self.rt_dictionary[rt_series_instance_uid]
            path = rt_dictionary['Path']
            self.all_RTs[path] = rt_dictionary['ROI_Names']
            for roi in rt_dictionary['ROI_Names']:
                if roi not in self.RTs_with_ROI_Names:
                    self.RTs_with_ROI_Names[roi] = [path]
                else:
                    self.RTs_with_ROI_Names[roi].append(path)
            if series_instance_uid in series_instance_uids:
                index = series_instance_uids.index(series_instance_uid)
                self.series_instances_dictionary[index]['RTs'].update({rt_series_instance_uid: self.rt_dictionary[rt_series_instance_uid]})
            else:
                while index in self.series_instances_dictionary:
                    index += 1
                template = return_template_dictionary()
                template['RTs'].update({rt_series_instance_uid: self.rt_dictionary[rt_series_instance_uid]})
                self.series_instances_dictionary[index] = template

    def __reset__(self):
        self.__reset_RTs__()
        self.dicom_handle_uid = None
        self.series_instances_dictionary = {}

    def __reset_RTs__(self):
        self.all_rois = []
        self.indexes_with_contours = []
        self.RS_struct_uid = None
        self.RTs_with_ROI_Names = {}

    def set_contour_names_and_associations(self, Contour_Names=None, associations=None):
        if Contour_Names is not None:
            self.__set_contour_names__(Contour_Names=Contour_Names)
        if associations is not None:
            self.__set_associations__(associations=associations)
        self.__check_if_all_contours_present__()

    def __set_associations__(self, associations={}):
        keys = list(associations.keys())
        for key in keys:
            associations[key.lower()] = associations[key].lower()
        if self.Contour_Names is not None:
            for name in self.Contour_Names:
                if name not in associations:
                    associations[name] = name
        self.associations, self.hierarchy = associations, {}

    def __set_contour_names__(self, Contour_Names):
        self.__reset_RTs__()
        Contour_Names = [i.lower() for i in Contour_Names]
        for name in Contour_Names:
            if name not in self.associations:
                self.associations[name] = name
        self.Contour_Names = Contour_Names

    def __set_description__(self, description):
        self.desciption = description

    def __set_iteration__(self, iteration=0):
        self.iteration = str(iteration)

    def __check_if_all_contours_present__(self):
        self.indexes_with_contours = []
        for index in self.series_instances_dictionary:
            if self.series_instances_dictionary[index]['Image_Path'] is None:
                continue
            RTs = self.series_instances_dictionary[index]['RTs']
            true_rois = []
            self.rois_in_case = []
            for RT_key in RTs:
                RT = RTs[RT_key]
                ROI_Names = RT['ROI_Names']
                for roi in ROI_Names:
                    if roi.lower() not in self.RTs_with_ROI_Names:
                        self.RTs_with_ROI_Names[roi.lower()] = [RT['Path']]
                    elif RT['Path'] not in self.RTs_with_ROI_Names[roi.lower()]:
                        self.RTs_with_ROI_Names[roi.lower()].append(RT['Path'])
                    if roi.lower() not in self.rois_in_case:
                        self.rois_in_case.append(roi.lower())
                    if roi.lower() not in self.all_rois:
                        self.all_rois.append(roi.lower())
                    if self.Contour_Names:
                        if roi.lower() in self.associations:
                            true_rois.append(self.associations[roi.lower()])
                        elif roi.lower() in self.Contour_Names:
                            true_rois.append(roi.lower())
            self.all_contours_exist = True
            lacking_rois = []
            for roi in self.Contour_Names:
                if roi not in true_rois:
                    lacking_rois.append(roi)
            if lacking_rois:
                self.all_contours_exist = False
                print('Lacking {} in index {}, location {}. Found {}'.format(lacking_rois, index,
                                                                             self.series_instances_dictionary[index]
                                                                             ['Image_Path'], self.rois_in_case))
            if index not in self.indexes_with_contours:
                if self.all_contours_exist or not self.require_all_contours:
                    self.indexes_with_contours.append(index)  # Add the index that has the contours

    def return_rois(self, print_rois=True):
        if print_rois:
            print('The following ROIs were found')
            for roi in self.all_rois:
                print(roi)
        return self.all_rois

    def where_are_RTs(self, ROIName):
        print('Please move over to using .where_is_ROI(), as this better represents the definition')
        self.where_is_ROI(ROIName=ROIName)

    def where_is_ROI(self, ROIName):
        if ROIName.lower() in self.RTs_with_ROI_Names:
            print('Contours of {} are located:'.format(ROIName.lower()))
            for path in self.RTs_with_ROI_Names[ROIName.lower()]:
                print(path)
        else:
            print('{} was not found within the set, check spelling or list all rois'.format(ROIName))

    def which_indexes_have_all_rois(self):
        if self.Contour_Names:
            print('The following indexes have all ROIs present')
            for index in self.indexes_with_contours:
                print('Index {}, located at {}'.format(index, self.series_instances_dictionary[index]['Image_Path']))
            return self.indexes_with_contours
        else:
            print('You need to first define what ROIs you want, please use .set_contour_names(roi_list)')

    def down_folder(self, input_path):
        print('Please move from down_folder() to walk_through_folders()')
        self.walk_through_folders(input_path=input_path)

    def walk_through_folders(self, input_path, thread_count = int(cpu_count() * 0.9 - 1)):
        """
        Iteratively work down paths to find DICOM files, if they are present, add to the series instance UID dictionary
        :param input_path: path to walk
        """
        paths_with_dicom = []
        for root, dirs, files in os.walk(input_path):
            dicom_files = [i for i in files if i.endswith('.dcm')]
            if dicom_files:
                paths_with_dicom.append(root)
                # dicom_adder.add_dicom_to_dictionary_from_path(dicom_path=root, images_dictionary=self.images_dictionary,
                #                                               rt_dictionary=self.rt_dictionary)
        if paths_with_dicom:
            q = Queue(maxsize=thread_count)
            A = (q,)
            threads = []
            for worker in range(thread_count):
                t = Thread(target=folder_worker, args=(A,))
                t.start()
                threads.append(t)
            for index, path in enumerate(paths_with_dicom):
                item = [path, self.images_dictionary, self.rt_dictionary]
                q.put(item)
            for i in range(thread_count):
                q.put(None)
            for t in threads:
                t.join()
            self.__compile__()
        if self.verbose or len(self.series_instances_dictionary) > 1:
            for key in self.series_instances_dictionary:
                print('Index {}, description {} at {}'.format(key,
                                                              self.series_instances_dictionary[key]['Description'],
                                                              self.series_instances_dictionary[key]['Image_Path']))
            print('{} unique series IDs were found. Default is index 0, to change use '
                  'set_index(index)'.format(len(self.series_instances_dictionary)))
        self.__check_if_all_contours_present__()
        return None

    def add_rt_to_dictionary(self, ds, path):
        """
        :param ds: pydicom data structure
        :param path: path to the images or structure in question
        """
        series_instance_uid = ds.SeriesInstanceUID
        keys = []
        series_instance_uids = []
        for key, value in self.series_instances_dictionary.items():
            keys.append(key)
            series_instance_uids.append(value['SeriesInstanceUID'])
        for referenced_frame_of_reference in ds.ReferencedFrameOfReferenceSequence:
            for referred_study_sequence in referenced_frame_of_reference.RTReferencedStudySequence:
                for referred_series in referred_study_sequence.RTReferencedSeriesSequence:
                    refed_series_instance_uid = referred_series.SeriesInstanceUID
                    if refed_series_instance_uid not in series_instance_uids:
                        index = 0
                        while index in self.series_instances_dictionary:
                            index += 1
                        temp_dict = return_template_dictionary()
                        temp_dict['SeriesInstanceUID'] = refed_series_instance_uid
                        temp_dict['PatientID'] = ds.PatientID
                        self.series_instances_dictionary[index] = temp_dict
                    else:
                        index = keys[series_instance_uids.index(refed_series_instance_uid)]
                    if series_instance_uid not in self.series_instances_dictionary[index]['RTs']:
                        if Tag((0x3006, 0x020)) in ds.keys():
                            ROI_Structure = ds.StructureSetROISequence
                        else:
                            ROI_Structure = []
                        rois_in_structure = {}
                        rois = []
                        for Structures in ROI_Structure:
                            rois.append(Structures.ROIName.lower())
                            if Structures.ROIName not in rois_in_structure:
                                rois_in_structure[Structures.ROIName] = Structures.ROINumber
                            if Structures.ROIName.lower() not in self.RTs_with_ROI_Names:
                                self.RTs_with_ROI_Names[Structures.ROIName.lower()] = [path]
                            elif path not in self.RTs_with_ROI_Names[Structures.ROIName.lower()]:
                                self.RTs_with_ROI_Names[Structures.ROIName.lower()].append(path)
                        temp_dict = {series_instance_uid: {'Path': path, 'ROI_Names': rois,
                                                           'ROIs_in_structure': rois_in_structure}}
                        self.all_RTs[path] = rois_in_structure
                        self.RTs_in_case[path] = rois_in_structure
                        self.series_instances_dictionary[index]['RTs'].update(temp_dict)

    def write_parallel(self, out_path, excel_file, thread_count=int(cpu_count() * 0.9 - 1)):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        final_out_dict = {'PatientID': [], 'Path': [], 'Iteration': [], 'Folder': [], 'SeriesInstanceUID': []}
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
            data = df.to_dict()
            for key in final_out_dict.keys():
                for index in data[key]:
                    final_out_dict[key].append(data[key][index])
        else:
            df = pd.DataFrame(final_out_dict)
            df.to_excel(excel_file, index=0)
        key_dict = {'series_instances_dictionary': self.series_instances_dictionary, 'associations': self.associations,
                    'arg_max': self.arg_max, 'require_all_contours': self.require_all_contours,
                    'Contour_Names': self.Contour_Names,
                    'description': self.desciption, 'get_dose_output': True}
        rewrite_excel = False
        '''
        First, build the excel file that we will use to reference iterations, Series UIDs, and paths
        '''
        for index in self.indexes_with_contours:
            series_instance_uid = self.series_instances_dictionary[index]['SeriesInstanceUID']
            previous_run = df.loc[df['SeriesInstanceUID'] == series_instance_uid]
            if previous_run.shape[0] == 0:
                rewrite_excel = True
                iteration = 0
                while iteration in df['Iteration'].values:
                    iteration += 1
                temp_dict = {'PatientID': [self.series_instances_dictionary[index]['PatientID']],
                             'Path': [self.series_instances_dictionary[index]['Image_Path']],
                             'Iteration': [int(iteration)], 'Folder': [None], 'SeriesInstanceUID': [series_instance_uid]}
                temp_df = pd.DataFrame(temp_dict)
                df = df.append(temp_df)
        if rewrite_excel:
            df.to_excel(excel_file, index=0)
        '''
        Next, read through the excel sheet and see if the out paths already exist
        '''
        q = Queue(maxsize=thread_count)
        A = (q,)
        threads = []
        for worker in range(thread_count):
            t = Thread(target=worker_def, args=(A,))
            t.start()
            threads.append(t)
        for index in self.indexes_with_contours:
            series_instance_uid = self.series_instances_dictionary[index]['SeriesInstanceUID']
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
            write_image = os.path.join(write_path, 'Overall_Data_{}_{}.nii.gz'.format(self.desciption, iteration))
            if os.path.exists(write_image):
                print('Already wrote out at {}'.format(write_path))
                continue
            item = [iteration, index, write_path, key_dict]
            q.put(item)
        for i in range(thread_count):
            q.put(None)
        for t in threads:
            t.join()

    def get_images_and_mask(self):
        assert self.index in self.series_instances_dictionary, \
            'Index is not present in the dictionary! Set it using set_index(index)'
        self.get_images()
        self.get_mask()

    def get_images(self):
        assert self.index in self.series_instances_dictionary, \
            'Index is not present in the dictionary! Set it using set_index(index)'
        index = self.index
        series_instance_uid = self.series_instances_dictionary[index]['SeriesInstanceUID']
        if self.dicom_handle_uid != series_instance_uid:  # Only load if needed
            if self.verbose:
                print('Loading images for {} at \n {}\n'.format(self.series_instances_dictionary[index]['Description'],
                                                                self.series_instances_dictionary[index]['Image_Path']))

            dicom_path = self.series_instances_dictionary[index]['Image_Path']
            dicom_names = self.reader.GetGDCMSeriesFileNames(dicom_path, series_instance_uid)
            self.ds = pydicom.read_file(dicom_names[0])
            self.reader.SetFileNames(dicom_names)
            self.dicom_handle = self.reader.Execute()
            add_sops_to_dictionary(sitk_dicom_reader=self.reader,
                                   series_instances_dictionary=self.series_instances_dictionary)
            if max(self.flip_axes):
                flipimagefilter = sitk.FlipImageFilter()
                flipimagefilter.SetFlipAxes(self.flip_axes)
                self.dicom_handle = flipimagefilter.Execute(self.dicom_handle)
            self.ArrayDicom = sitk.GetArrayFromImage(self.dicom_handle)
            self.image_size_cols, self.image_size_rows, self.image_size_z = self.dicom_handle.GetSize()
            self.dicom_handle_uid = series_instance_uid

    def get_mask(self):
        assert self.index in self.series_instances_dictionary, \
            'Index is not present in the dictionary! Set it using set_index(index)'
        assert self.Contour_Names, 'If you want a mask, you need to set the contour names you are looking ' \
                                   'for, use set_contour_names(list_of_roi_names).\nIf you just want to look at images ' \
                                   'use get_images() not get_images_and_mask() or get_mask()'
        index = self.index
        if self.dicom_handle_uid != self.series_instances_dictionary[index]['SeriesInstanceUID']:
            print('Loading images for index {}, since mask was requested but image loading was '
                  'previously different\n'.format(index))
            self.get_images()
        if self.RS_struct_uid == self.series_instances_dictionary[index]['SeriesInstanceUID']:  # Already loaded
            return None
        self.mask = np.zeros(
            [self.dicom_handle.GetSize()[-1], self.image_size_rows, self.image_size_cols, len(self.Contour_Names) + 1],
            dtype='int8')
        RTs = self.series_instances_dictionary[index]['RTs']
        for RT_key in RTs:
            RT = RTs[RT_key]
            ROIName_Number = RT['ROIs_in_structure']
            RS_struct = None
            self.structure_references = {}
            for ROI_Name in ROIName_Number.keys():
                true_name = None
                if ROI_Name in self.associations:
                    true_name = self.associations[ROI_Name].lower()
                elif ROI_Name.lower() in self.associations:
                    true_name = self.associations[ROI_Name.lower()]
                if true_name and true_name in self.Contour_Names:
                    if RS_struct is None:
                        self.RS_struct = RS_struct = pydicom.read_file(RT['Path'])
                        self.RS_struct_uid = self.series_instances_dictionary[index]['SeriesInstanceUID']
                    for contour_number in range(len(self.RS_struct.ROIContourSequence)):
                        self.structure_references[
                            self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number
                    structure_index = self.structure_references[ROIName_Number[ROI_Name]]
                    mask = self.contours_to_mask(structure_index)
                    self.mask[..., self.Contour_Names.index(true_name) + 1] += mask
                    self.mask[self.mask > 1] = 1
        if self.flip_axes[0]:
            self.mask = self.mask[:, :, ::-1, ...]
        if self.flip_axes[1]:
            self.mask = self.mask[:, ::-1, ...]
        if self.flip_axes[2]:
            self.mask = self.mask[::-1, ...]
        if self.arg_max:
            self.mask = np.argmax(self.mask, axis=-1)
        self.annotation_handle = sitk.GetImageFromArray(self.mask.astype('int8'))
        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())
        return None

    def contours_to_mask(self, index):
        mask = np.zeros([self.dicom_handle.GetSize()[-1], self.image_size_rows, self.image_size_cols], dtype='int8')
        Contour_data = self.RS_struct.ROIContourSequence[index].ContourSequence
        for i in range(len(Contour_data)):
            as_array = np.asarray(Contour_data[i].ContourData[:])
            reshaped = np.reshape(as_array, [as_array.shape[0] // 3, 3])
            matrix_points = np.asarray([self.dicom_handle.TransformPhysicalPointToIndex(reshaped[i])
                                        for i in range(reshaped.shape[0])])
            self.col_val = matrix_points[:, 0]
            self.row_val = matrix_points[:, 1]
            z_vals = matrix_points[:, 2]
            temp_mask = poly2mask(self.row_val, self.col_val, [self.image_size_rows, self.image_size_cols])
            # temp_mask[self.row_val, self.col_val] = 0
            mask[z_vals[0], temp_mask] += 1
        mask = mask % 2
        return mask

    def use_template(self):
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

    def write_images_annotations(self, out_path):
        image_path = os.path.join(out_path, 'Overall_Data_{}_{}.nii.gz'.format(self.desciption, self.iteration))
        annotation_path = os.path.join(out_path, 'Overall_mask_{}_y{}.nii.gz'.format(self.desciption, self.iteration))
        if os.path.exists(image_path):
            return None
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
        if len(self.dose_handles) > 0:
            for dose_index, dose_handle in enumerate(self.dose_handles):
                if len(self.dose_handles) > 1:
                    dose_path = os.path.join(out_path,
                                             'Overall_dose_{}_{}_{}.nii.gz'.format(self.desciption, self.iteration,
                                                                                   dose_index))
                else:
                    dose_path = os.path.join(out_path,
                                             'Overall_dose_{}_{}.nii.gz'.format(self.desciption, self.iteration))
                sitk.WriteImage(dose_handle, dose_path)
        fid = open(os.path.join(self.series_instances_dictionary[self.index]['Image_Path'],
                                '{}_Iteration_{}.txt'.format(self.desciption, self.iteration)), 'w+')
        fid.close()

    def prediction_array_to_RT(self, prediction_array, output_dir, ROI_Names):
        """
        :param prediction_array: numpy array of prediction, expected shape is [#Images, Rows, Cols, #Classes + 1]
        :param output_dir: directory to pass RT structure to
        :param ROI_Names: list of ROI names equal to the number of classes
        :param index: index of the series instance UID to match with
        :return:
        """
        assert ROI_Names is not None, 'You need to provide ROI_Names'
        assert prediction_array.shape[-1] == len(ROI_Names) + 1, 'Your last dimension of prediction array should be' \
                                                                 ' equal  to the number or ROI_names minus 1, channel' \
                                                                 ' 0 is background'
        assert self.index in self.series_instances_dictionary, \
            'Index is not present in the dictionary! Set it using set_index(index)'
        index = self.index
        if self.dicom_handle_uid != self.series_instances_dictionary[index]['SeriesInstanceUID']:
            self.get_images()
        self.SOPInstanceUIDs = self.series_instances_dictionary[index]['SOP_Instance_UIDs']

        if self.create_new_RT or len(self.series_instances_dictionary[index]['RTs']) == 0:
            self.use_template()
        elif self.RS_struct_uid != self.series_instances_dictionary[index]['SeriesInstanceUID']:
            RTs = self.series_instances_dictionary[index]['RTs']
            for uid_key in RTs:
                self.RS_struct = pydicom.read_file(RTs[uid_key]['Path'])
                self.RS_struct_uid = self.series_instances_dictionary[index]['SeriesInstanceUID']
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

    def with_annotations(self, annotations, output_dir, ROI_Names=None):
        print('Please move over to using prediction_array_to_RT')
        self.prediction_array_to_RT(prediction_array=annotations, output_dir=output_dir, ROI_Names=ROI_Names)

    def mask_to_contours(self):
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
        new_ROINumber = 1000
        for Name in self.ROI_Names:
            new_ROINumber -= 1
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
            self.RS_struct.StructureSetROISequence[self.struct_index].ROINumber = new_ROINumber
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
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ObservationNumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ROIObservationLabel = Name
            self.RS_struct.RTROIObservationsSequence[self.struct_index].RTROIInterpretedType = 'ORGAN'

            if make_new == 1:
                self.RS_struct.ROIContourSequence.insert(0, copy.deepcopy(self.RS_struct.ROIContourSequence[0]))
            self.RS_struct.ROIContourSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            del self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[1:]
            self.RS_struct.ROIContourSequence[self.struct_index].ROIDisplayColor = temp_color_list[color_int]
            del temp_color_list[color_int]
            thread_count = int(cpu_count() * 0.9 - 1)
            contour_dict = {}
            q = Queue(maxsize=thread_count)
            threads = []
            kwargs = {'image_size_rows': self.image_size_rows, 'image_size_cols': self.image_size_cols,
                      'PixelSize': self.PixelSize, 'contour_dict': contour_dict, 'RS': self.RS_struct}

            A = [q, kwargs]
            # pointer_class = PointOutputMakerClass(**kwargs)
            for worker in range(thread_count):
                t = Thread(target=contour_worker, args=(A,))
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
        print('Writing out data...')
        pydicom.write_file(out_name, self.RS_struct)
        fid = open(os.path.join(self.output_dir, 'Completed.txt'), 'w+')
        fid.close()
        print('Finished!')
        return None

    def change_template(self):
        keys = self.RS_struct.keys()
        for key in keys:
            # print(self.RS_struct[key].name)
            if self.RS_struct[key].name == 'Referenced Frame of Reference Sequence':
                break
        self.RS_struct[key]._value[0].FrameOfReferenceUID = self.ds.FrameOfReferenceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = self.ds.StudyInstanceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
            0].SeriesInstanceUID = self.ds.SeriesInstanceUID
        for i in range(len(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                               0].ContourImageSequence) - 1):
            del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[-1]
        fill_segment = copy.deepcopy(
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[0])
        for i in range(len(self.SOPInstanceUIDs)):
            temp_segment = copy.deepcopy(fill_segment)
            temp_segment.ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence.append(temp_segment)
        del \
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
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

    def get_dose(self):
        reader = sitk.ImageFileReader()
        output, spacing, direction, origin = None, None, None, None
        for dose_file in self.RDs_in_case:
            if os.path.split(dose_file)[-1].startswith('RTDOSE - PLAN'):
                reader.SetFileName(dose_file)
                reader.ReadImageInformation()
                dose = reader.Execute()
                spacing = dose.GetSpacing()
                origin = dose.GetOrigin()
                direction = dose.GetDirection()
                scaling_factor = float(reader.GetMetaData("3004|000e"))
                dose = sitk.GetArrayFromImage(dose) * scaling_factor
                if output is None:
                    output = dose
                else:
                    output += dose
        if output is not None:
            output = sitk.GetImageFromArray(output)
            output.SetSpacing(spacing)
            output.SetDirection(direction)
            output.SetOrigin(origin)
            self.dose_handles.append(output)

    def Make_Contour_From_directory(self, PathDicom):
        print('Please move over to using add_dicom_to_dictionary_from_path')
        self.add_dicom_to_dictionary_from_path(PathDicom=PathDicom)

    def make_contour_from_directory(self, dicom_path):
        print('Please move over to using add_dicom_to_dictionary_from_path')
        self.add_dicom_to_dictionary_from_path(PathDicom=dicom_path)
        return None

    def rewrite_RT(self, lstRSFile=None):
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
        self.rois_in_case = []
        for i, Structures in enumerate(self.ROI_Structure):
            if Structures.ROIName in self.associations:
                new_name = self.associations[Structures.ROIName]
                self.RS_struct.StructureSetROISequence[i].ROIName = new_name
            self.rois_in_case.append(self.RS_struct.StructureSetROISequence[i].ROIName)
        for i, ObsSequence in enumerate(self.Observation_Sequence):
            if ObsSequence.ROIObservationLabel in self.associations:
                new_name = self.associations[ObsSequence.ROIObservationLabel]
                self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel = new_name
        self.RS_struct.save_as(self.lstRSFile)


class Dicom_to_Imagestack(DicomReaderWriter):
    def __init__(self, **kwargs):
        print('Please move from using Dicom_to_Imagestack to DicomReaderWriter, same arguments are passed')
        super().__init__(**kwargs)


if __name__ == '__main__':
    pass
