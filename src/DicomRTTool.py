import os, copy, pydicom
import numpy as np
from pydicom.tag import Tag
import SimpleITK as sitk
from skimage import draw
from skimage.measure import label,regionprops,find_contours
from threading import Thread
from multiprocessing import cpu_count
from queue import *
import pandas as pd
import copy
import cv2 as cv
import matplotlib.pyplot as plt


def plot_scroll_Image(x):
    '''
    :param x: input to view of form [rows, columns, # images]
    :return:
    '''
    if x.dtype not in ['float32','float64']:
        x = copy.deepcopy(x).astype('float32')
    if len(x.shape) > 3:
        x = np.squeeze(x)
    if len(x.shape) == 3:
        if x.shape[0] != x.shape[1]:
            x = np.transpose(x,[1,2,0])
        elif x.shape[0] == x.shape[2]:
            x = np.transpose(x, [1, 2, 0])
    fig, ax = plt.subplots(1, 1)
    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=-1)
    tracker = IndexTracker(ax, x)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker
    #Image is input in the form of [#images,512,512,#channels]


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = np.where((np.min(self.X,axis=(0,1))!= np.max(self.X,axis=(0,1))))[-1]
        if len(self.ind) > 0:
            self.ind = self.ind[len(self.ind)//2]
        else:
            self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def contour_worker(A):
    q, kwargs = A
    point_maker = Point_Output_Maker_Class(**kwargs)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            point_maker.make_output(*item)
        q.task_done()


def worker_def(A):
    q, Contour_Names, associations, desc, final_out_dict = A
    base_class = DicomReaderWriter(get_images_mask=True, associations=associations,
                                   Contour_Names=Contour_Names, desc=desc, get_dose_output=True)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            path, iteration, out_path = item
            print(path)
            try:
                base_class.Make_Contour_From_directory(PathDicom=path)
                base_class.set_iteration(iteration)
                base_class.write_images_annotations(out_path)
                final_out_dict['MRN'].append(base_class.ds.PatientID)
                final_out_dict['Iteration'].append(iteration)
                final_out_dict['Path'].append(path)
                final_out_dict['Folder'].append('')
            except:
                print('failed on {}'.format(path))
            q.task_done()


class Point_Output_Maker_Class(object):
    def __init__(self, image_size_rows, image_size_cols, PixelSize, contour_dict, RS):
        self.image_size_rows, self.image_size_cols = image_size_rows, image_size_cols
        self.PixelSize = PixelSize
        self.contour_dict = contour_dict
        self.RS = RS

    def make_output(self, annotation, i):
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
            contours = find_contours(temp_image, 0)
            for contour in contours:
                self.contour_dict[i].append(contour)


class DicomReaderWriter:
    def __init__(self, rewrite_RT_file=False, delete_previous_rois=True,Contour_Names=None,
                 template_dir=None, get_images_mask=True, arg_max=True,
                 associations={},desc='',iteration=0, get_dose_output=False, flip_axes=(False, False, False), **kwargs):
        '''
        :param rewrite_RT_file: Boolean, should we re-write the RT structure
        :param delete_previous_rois: delete the previous RTs within the structure
        :param Contour_Names: list of contour nmes
        :param template_dir: default to None, specifies path to template RT structure
        :param get_images_mask: boolean, load the images and mask
        :param arg_max: perform argmax on the mask
        :param associations: dictionary of associations {'liver_bma_program_4': 'liver'}
        :param desc: description information to add to .nii files
        :param iteration: what iteration for writing .nii files
        :param get_dose_output: boolean, collect dose information
        :param flip_axes: tuple(3), axis that you want to flip, defaults to (False, False, False)
        :param kwargs:
        '''
        self.get_dose_output = get_dose_output
        self.flip_axes = flip_axes
        self.associations = associations
        self.set_contour_names(Contour_Names)
        self.set_associations(associations)
        self.set_get_images_and_mask(get_images_mask)
        self.set_description(desc)
        self.set_iteration(iteration)
        self.arg_max = arg_max
        self.rewrite_RT_file = rewrite_RT_file
        self.dose_handles = []
        if template_dir is None or not os.path.exists(template_dir):
            template_dir = os.path.join(os.path.split(__file__)[0], 'template_RS.dcm')
        self.template_dir = template_dir
        self.template = True
        self.delete_previous_rois = delete_previous_rois
        self.get_images_mask = get_images_mask
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.__reset__()

    def __reset__(self):
        self.all_RTs = {}
        self.all_rois = []
        self.all_paths = []
        self.paths_with_contours = []

    def set_associations(self, associations={}):
        keys = list(associations.keys())
        for key in keys:
            associations[key.lower()] = associations[key].lower()
        if self.Contour_Names is not None:
            for name in self.Contour_Names:
                if name not in associations:
                    associations[name] = name
        self.associations, self.hierarchy = associations, {}

    def set_get_images_and_mask(self, get_images_mask=True):
        self.get_images_mask = get_images_mask

    def set_contour_names(self, Contour_Names=None):
        self.__reset__()
        if Contour_Names is None:
            Contour_Names = []
        else:
            Contour_Names = [i.lower() for i in Contour_Names]
        self.Contour_Names = Contour_Names
        self.set_associations(self.associations)

    def set_description(self, description):
        self.desciption = description

    def set_iteration(self, iteration=0):
        self.iteration = str(iteration)

    def down_folder(self, input_path, reset=True):
        files = []
        dirs = []
        file = []
        for root, dirs, files in os.walk(input_path):
            break
        for val in files:
            if val.find('.dcm') != -1:
                file = val
                break
        if file and input_path:
            self.all_paths.append(input_path)
            self.Make_Contour_From_directory(input_path)
        for dir in dirs:
            new_directory = os.path.join(input_path, dir)
            self.down_folder(new_directory)
        return None

    def make_array(self, PathDicom):
        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.lstRSFile = None
        self.Dicom_info = []
        fileList = []
        self.RTs_in_case = {}
        self.RDs_in_case = {}
        for dirName, dirs, fileList in os.walk(PathDicom):
            break
        fileList = [i for i in fileList if i.find('.dcm') != -1]
        if not self.get_images_mask:
            RT_fileList = [i for i in fileList if i.find('RT') == 0 or i.find('RS') == 0]
            if RT_fileList:
                fileList = RT_fileList
            for filename in fileList:
                try:
                    ds = pydicom.read_file(os.path.join(dirName, filename))
                    self.ds = ds
                    if ds.Modality == 'CT' or ds.Modality == 'MR' or ds.Modality == 'PT':
                        self.lstFilesDCM.append(os.path.join(dirName, filename))
                        self.Dicom_info.append(ds)
                        self.ds = ds
                    elif ds.Modality == 'RTSTRUCT':
                        self.lstRSFile = os.path.join(dirName, filename)
                        self.RTs_in_case[self.lstRSFile] = []
                except:
                    continue
            if self.lstFilesDCM:
                self.RefDs = pydicom.read_file(self.lstFilesDCM[0])
        else:
            self.dicom_names = self.reader.GetGDCMSeriesFileNames(self.PathDicom)
            if self.dicom_names:
                self.reader.SetFileNames(self.dicom_names)
                self.RefDs = pydicom.read_file(self.dicom_names[0])
                self.ds = pydicom.read_file(self.dicom_names[0])
            self.get_images()
            image_files = [i.split(PathDicom)[1][1:] for i in self.dicom_names]
            RT_Files = [os.path.join(PathDicom, file) for file in fileList if file not in image_files]
            reader = sitk.ImageFileReader()
            for lstRSFile in RT_Files:
                reader.SetFileName(lstRSFile)
                try:
                    reader.ReadImageInformation()
                    modality = reader.GetMetaData("0008|0060")
                except:
                    modality = pydicom.read_file(lstRSFile).Modality
                if modality.lower().find('dose') != -1:
                    self.RDs_in_case[lstRSFile] = []
                elif modality.lower().find('struct') != -1:
                    self.RTs_in_case[lstRSFile] = []

        self.all_contours_exist = False
        self.rois_in_case = []
        self.all_RTs.update(self.RTs_in_case)
        if len(self.RTs_in_case.keys()) > 0:
            self.template = False
            for self.lstRSFile in self.RTs_in_case:
                self.get_rois_from_RT()
        elif self.get_images_mask:
            self.use_template()

    def write_parallel(self, out_path, excel_file, thread_count=int(cpu_count()*0.9-1)):
        out_path = os.path.join(out_path,self.desciption)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        q = Queue(maxsize=thread_count)
        final_out_dict = {'MRN': [], 'Path': [], 'Iteration': [], 'Folder': []}
        if os.path.exists(excel_file):
            data = pd.read_excel(excel_file)
            data = data.to_dict()
            for key in final_out_dict.keys():
                for index in data[key]:
                    final_out_dict[key].append(data[key][index])
        A = [q, self.Contour_Names, self.associations, self.desciption, final_out_dict]
        threads = []
        for worker in range(thread_count):
            t = Thread(target=worker_def, args=(A,))
            t.start()
            threads.append(t)
        out_dict = {'Path':[], 'Iteration':[]}
        iterations = copy.deepcopy(final_out_dict['Iteration'])
        for path in self.paths_with_contours:
            iteration_files = [i for i in os.listdir(path) if i.find('{}_Iteration'.format(self.desciption)) != -1]
            iteration = 0
            if iteration_files:
                file = iteration_files[0]
                iteration = int(file.split('_')[-1].split('.')[0])
                iterations.append(iteration)
            elif path in final_out_dict['Path']:
                iteration = final_out_dict['Iteration'][final_out_dict['Path'].index(path)]
            else:
                while iteration in iterations:
                    iteration += 1
                iterations.append(iteration)
            out_dict['Path'].append(path)
            out_dict['Iteration'].append(iteration)
        for index in range(len(out_dict['Path'])):
            path = out_dict['Path'][index]
            iteration = out_dict['Iteration'][index]
            item = [path, iteration, out_path]
            if os.path.exists(os.path.join(out_path, 'Overall_Data_{}_{}.nii.gz'.format(self.desciption, iteration))):
                continue
            if iteration in final_out_dict['Iteration']:
                if final_out_dict['Folder'][final_out_dict['Iteration'].index(iteration)] in ['Train','Test','Validation']:
                    continue
            q.put(item)
        for i in range(thread_count):
            q.put(None)
        for t in threads:
            t.join()
        df = pd.DataFrame(final_out_dict)
        df.to_excel(excel_file,index=0)

    def get_rois_from_RT(self):
        rois_in_structure = {}
        self.RS_struct = pydicom.read_file(self.lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            ROI_Structure = []
        for Structures in ROI_Structure:
            if Structures.ROIName not in self.rois_in_case:
                self.rois_in_case.append(Structures.ROIName)
                rois_in_structure[Structures.ROIName] = Structures.ROINumber
        self.all_RTs[self.lstRSFile] = rois_in_structure
        self.RTs_in_case[self.lstRSFile] = rois_in_structure

    def get_mask(self):
        self.mask = np.zeros(
            [len(self.dicom_names), self.image_size_rows, self.image_size_cols, len(self.Contour_Names) + 1],
            dtype='int8')
        for RT_key in self.RTs_in_case:
            found_rois = {}
            ROIName_Number = self.RTs_in_case[RT_key]
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
                        self.RS_struct = RS_struct = pydicom.read_file(RT_key)
                        for contour_number in range(len(self.RS_struct.ROIContourSequence)):
                            self.structure_references[
                                self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number
                    found_rois[true_name] = {'Hierarchy': 999, 'Name': ROI_Name, 'Roi_Number': self.RTs_in_case[RT_key][ROI_Name]}
            for ROI_Name in found_rois.keys():
                if found_rois[ROI_Name]['Roi_Number'] in self.structure_references:
                    index = self.structure_references[found_rois[ROI_Name]['Roi_Number']]
                    mask = self.Contours_to_mask(index)
                    self.mask[..., self.Contour_Names.index(ROI_Name) + 1][mask == 1] = 1
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

    def Contours_to_mask(self, index):
        mask = np.zeros([len(self.dicom_names), self.image_size_rows, self.image_size_cols], dtype='int8')
        Contour_data = self.RS_struct.ROIContourSequence[index].ContourSequence
        for i in range(len(Contour_data)):
            as_array = np.asarray(Contour_data[i].ContourData[:])
            reshaped = np.reshape(as_array, [as_array.shape[0]//3, 3])
            matrix_points = np.asarray([self.dicom_handle.TransformPhysicalPointToIndex(reshaped[i])
                                        for i in range(reshaped.shape[0])])
            self.col_val = matrix_points[:, 0]
            self.row_val = matrix_points[:, 1]
            z_vals = matrix_points[:, 2]
            temp_mask = self.poly2mask(self.row_val, self.col_val, [self.image_size_rows, self.image_size_cols])
            temp_mask[self.row_val, self.col_val] = 0
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
        self.changetemplate()

    def get_images(self):
        self.dicom_handle = self.reader.Execute()
        sop_instance_UID_key = "0008|0018"
        self.SOPInstanceUIDs = [self.reader.GetMetaData(i, sop_instance_UID_key) for i in
                                range(self.dicom_handle.GetDepth())]
        if max(self.flip_axes):
            flipimagefilter = sitk.FlipImageFilter()
            flipimagefilter.SetFlipAxes(self.flip_axes)
            self.dicom_handle = flipimagefilter.Execute(self.dicom_handle)
        self.ArrayDicom = sitk.GetArrayFromImage(self.dicom_handle)
        self.image_size_cols, self.image_size_rows, self.image_size_z = self.dicom_handle.GetSize()


    def write_images_annotations(self, out_path):
        image_path = os.path.join(out_path, 'Overall_Data_{}_{}.nii.gz'.format(self.desciption, self.iteration))
        annotation_path = os.path.join(out_path, 'Overall_mask_{}_y{}.nii.gz'.format(self.desciption,self.iteration))
        if os.path.exists(image_path):
            return None
        pixel_id = self.dicom_handle.GetPixelIDTypeAsString()
        if pixel_id.find('32-bit signed integer') != 0:
            self.dicom_handle = sitk.Cast(self.dicom_handle, sitk.sitkFloat32)
        sitk.WriteImage(self.dicom_handle,image_path)

        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())
        pixel_id = self.annotation_handle.GetPixelIDTypeAsString()
        if pixel_id.find('int') == -1:
            self.annotation_handle = sitk.Cast(self.annotation_handle, sitk.sitkUInt8)
        sitk.WriteImage(self.annotation_handle,annotation_path)
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
        fid = open(os.path.join(self.PathDicom, self.desciption + '_Iteration_' + self.iteration + '.txt'), 'w+')
        fid.close()

    def poly2mask(self, vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask

    def with_annotations(self, annotations, output_dir, ROI_Names=None):
        assert ROI_Names is not None, 'You need to provide ROI_Names'
        annotations = np.squeeze(annotations)
        self.image_size_z, self.image_size_rows, self.image_size_cols = annotations.shape[:3]
        self.ROI_Names = ROI_Names
        self.output_dir = output_dir
        if len(annotations.shape) == 3:
            annotations = np.expand_dims(annotations, axis=-1)
        if self.flip_axes[0]:
            annotations = annotations[:, :, ::-1, ...]
        if self.flip_axes[1]:
            annotations = annotations[:, ::-1, ...]
        if self.flip_axes[2]:
            annotations = annotations[::-1, ...]
        self.annotations = annotations
        self.Mask_to_Contours()

    def Mask_to_Contours(self):
        self.RefDs = self.ds
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
                self.RS_struct.StructureSetROISequence.insert(0,copy.deepcopy(self.RS_struct.StructureSetROISequence[0]))
                # self.RS_struct.ROIContourSequence[0].ContourSequence[0].ContourData = []
                # self.RS_struct.ROIContourSequence[0].ContourSequence[0].NumberofContourPoints = 0
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
                    copy.deepcopy(self.RS_struct.RTROIObservationsSequence[0]))
                if 'MaterialID' in self.RS_struct.RTROIObservationsSequence[self.struct_index]:
                    del self.RS_struct.RTROIObservationsSequence[self.struct_index].MaterialID
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ObservationNumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ROIObservationLabel = Name
            self.RS_struct.RTROIObservationsSequence[self.struct_index].RTROIInterpretedType = 'ORGAN'

            if make_new == 1:
                self.RS_struct.ROIContourSequence.insert(0,copy.deepcopy(self.RS_struct.ROIContourSequence[0]))
            self.RS_struct.ROIContourSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            del self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[1:]
            self.RS_struct.ROIContourSequence[self.struct_index].ROIDisplayColor = temp_color_list[color_int]
            del temp_color_list[color_int]
            thread_count = int(cpu_count()*0.9-1)
            contour_dict = {}
            q = Queue(maxsize=thread_count)
            threads = []
            kwargs = {'image_size_rows': self.image_size_rows, 'image_size_cols': self.image_size_cols,
                      'PixelSize': self.PixelSize, 'contour_dict': contour_dict, 'RS': self.RS_struct}

            A = [q,kwargs]
            # pointer_class = Point_Output_Maker_Class(**kwargs)
            for worker in range(thread_count):
                t = Thread(target=contour_worker, args=(A,))
                t.start()
                threads.append(t)
            contour_num = 0
            if np.max(annotations) > 0:  # If we have an annotation, write it
                image_locations = np.max(annotations, axis=(1, 2))
                indexes = np.where(image_locations > 0)[0]
                for index in indexes:
                    item = [annotations[index, ...], index]
                    # pointer_class.make_output(*item)
                    q.put(item)
                for i in range(thread_count):
                    q.put(None)
                for t in threads:
                    t.join()
                for i in contour_dict.keys():
                    for all_contours in contour_dict[i]:
                        contours = [[int(contour[1]), int(contour[0]), int(i)] for contour in np.squeeze(all_contours)]
                        points = [self.dicom_handle.TransformIndexToPhysicalPoint(ii) for ii in contours]
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
                            contour_num].NumberofContourPoints = len(output) // 3
                        contour_num += 1
        self.RS_struct.SOPInstanceUID += '.' + str(np.random.randint(999))
        if self.template or self.delete_previous_rois:
            for i in range(len(self.RS_struct.StructureSetROISequence),len(self.ROI_Names),-1):
                del self.RS_struct.StructureSetROISequence[-1]
            for i in range(len(self.RS_struct.RTROIObservationsSequence),len(self.ROI_Names),-1):
                del self.RS_struct.RTROIObservationsSequence[-1]
            for i in range(len(self.RS_struct.ROIContourSequence),len(self.ROI_Names),-1):
                del self.RS_struct.ROIContourSequence[-1]
            for i in range(len(self.RS_struct.StructureSetROISequence)):
                self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
                self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
                self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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

    def changetemplate(self):
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
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0]

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
                dose = sitk.GetArrayFromImage(dose)*scaling_factor
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
        self.make_array(PathDicom)
        if self.rewrite_RT_file:
            self.rewrite_RT()
        if self.get_images_mask and self.Contour_Names is not None:
            self.mask = np.zeros([len(self.dicom_names), self.image_size_rows, self.image_size_cols, len(self.Contour_Names) + 1],
                                 dtype='int8')
            if not self.template:
                self.get_mask()
        if self.get_dose_output:
            self.get_dose()
        true_rois = []
        for roi in self.rois_in_case:
            if roi not in self.all_rois:
                self.all_rois.append(roi)
            if self.Contour_Names:
                if roi.lower() in self.associations:
                    true_rois.append(self.associations[roi.lower()])
                elif roi.lower() in self.Contour_Names:
                    true_rois.append(roi.lower())
        self.all_contours_exist = True
        for roi in self.Contour_Names:
            if roi not in true_rois:
                print('Lacking {} in {}'.format(roi, PathDicom))
                print('Found {}'.format(self.rois_in_case))
                self.all_contours_exist = False
                break
        if PathDicom not in self.paths_with_contours and self.all_contours_exist:
            self.paths_with_contours.append(PathDicom) # Add the path that has the contours
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
    xxx = 1
