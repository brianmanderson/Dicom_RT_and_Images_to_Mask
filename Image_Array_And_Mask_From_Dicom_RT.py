import os, copy, pydicom
import numpy as np
from pydicom.tag import Tag
import SimpleITK as sitk
from skimage import draw
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
        x = np.expand_dims(x,axis=0)
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
        self.ind = np.where(self.X != 0)[-1]
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


class DicomImagestoData:
    def __init__(self,path='',rewrite_RT_file=False,get_images_mask=True, associations={}, wanted_rois=[]):
        self.wanted_rois = wanted_rois
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.associations = associations
        self.hierarchy = {}
        self.all_rois = []
        self.all_RTs = []
        self.rewrite_RT_file = rewrite_RT_file
        self.get_images_mask = get_images_mask
        self.down_folder(path)

    def down_folder(self,input_path):
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
            self.Make_Contour_From_directory(input_path)
        for dir in dirs:
            new_directory = os.path.join(input_path,dir)
            self.down_folder(new_directory)
        return None

    def get_mask(self, Contour_Names):
        if type(Contour_Names) is not list:
            Contour_Names = [Contour_Names]
        for roi in Contour_Names:
            if roi not in self.associations:
                self.associations[roi] = roi
        self.Contour_Names = Contour_Names

        # And this is making a mask file
        self.mask = np.zeros([len(self.dicom_names), self.image_size_1, self.image_size_2,len(self.Contour_Names)+1], dtype='int8')

        self.structure_references = {}
        for contour_number in range(len(self.RS_struct.ROIContourSequence)):
            self.structure_references[
                self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number

        found_rois = {}
        for roi in self.Contour_Names:
            found_rois[roi] = {'Hierarchy': 999, 'Name': [], 'Roi_Number': 0}
        for Structures in self.ROI_Structure:
            ROI_Name = Structures.ROIName
            if Structures.ROINumber not in self.structure_references.keys():
                continue
            true_name = None
            if ROI_Name in self.associations:
                true_name = self.associations[ROI_Name]
            elif ROI_Name in self.associations:
                true_name = self.associations[ROI_Name]
            if true_name and true_name in self.Contour_Names:
                if true_name in self.hierarchy.keys():
                    for roi in self.hierarchy[true_name]:
                        if roi == ROI_Name:
                            index_val = self.hierarchy[true_name].index(roi)
                            if index_val < found_rois[true_name]['Hierarchy']:
                                found_rois[true_name]['Hierarchy'] = index_val
                                found_rois[true_name]['Name'] = ROI_Name
                                found_rois[true_name]['Roi_Number'] = Structures.ROINumber
                else:
                    found_rois[true_name] = {'Hierarchy': 999, 'Name': ROI_Name, 'Roi_Number': Structures.ROINumber}
        i = 1 # For background
        for ROI_Name in found_rois.keys():
            if found_rois[ROI_Name]['Roi_Number'] in self.structure_references:
                index = self.structure_references[found_rois[ROI_Name]['Roi_Number']]
                mask = self.get_mask_for_contour(index)
                self.mask[...,i][mask == 1] = 1
                i += 1
        self.mask = np.argmax(self.mask,axis=-1).astype('int8')
        self.mask_handle = sitk.GetImageFromArray(self.mask)
        self.mask_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.mask_handle.SetDirection(self.dicom_handle.GetDirection())
        self.mask_handle.SetOrigin(self.dicom_handle.GetOrigin())
        return None

    def Make_Contour_From_directory(self,PathDicom):
        self.prep_data(PathDicom)
        if self.rewrite_RT_file:
            self.rewrite_RT()
        if self.get_images_mask:
            self.get_images_and_mask()
        true_rois = []
        for roi in self.rois_in_case:
            if roi not in self.all_rois:
                self.all_rois.append(roi)
            if self.wanted_rois:
                    if roi in self.associations:
                        true_rois.append(self.associations[roi])
                    elif roi in self.wanted_rois:
                        true_rois.append(roi)
            for roi in self.wanted_rois:
                if roi not in true_rois:
                    print('Lacking {} in {}'.format(roi, PathDicom))
        return None

    def prep_data(self,PathDicom):
        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.lstRSFile = []
        self.Dicom_info = []
        fileList = []
        for dirName, dirs, fileList in os.walk(PathDicom):
            break
        fileList = [i for i in fileList if i.find('.dcm') != -1]
        if not self.get_images_mask:
            RT_fileList = [i for i in fileList if i.find('RT') == 0 or i.find('RS') == 0]
            print(RT_fileList)
            if RT_fileList:
                fileList = RT_fileList
            for filename in fileList:
                try:
                    ds = pydicom.read_file(os.path.join(dirName,filename))
                    if ds.Modality == 'CT' or ds.Modality == 'MR' or ds.Modality == 'PT':  # check whether the file's DICOM
                        self.lstFilesDCM.append(os.path.join(dirName, filename))
                        self.Dicom_info.append(ds)
                    elif ds.Modality == 'RTSTRUCT':
                        self.lstRSFile = os.path.join(dirName, filename)
                        self.all_RTs.append(self.lstRSFile)
                except:
                    # if filename.find('Iteration_') == 0:
                    #     os.remove(PathDicom+filename)
                    continue
            if self.lstFilesDCM:
                self.RefDs = pydicom.read_file(self.lstFilesDCM[0])
        else:
            self.dicom_names = self.reader.GetGDCMSeriesFileNames(self.PathDicom)
            self.reader.SetFileNames(self.dicom_names)
            image_files = [i.split(PathDicom)[1][1:] for i in self.dicom_names]
            lstRSFiles = [os.path.join(PathDicom, file) for file in fileList if file not in image_files]
            if lstRSFiles:
                self.lstRSFile = lstRSFiles[0]
            self.RefDs = pydicom.read_file(self.dicom_names[0])
            self.ds = pydicom.read_file(self.dicom_names[0])
        self.mask_exist = False
        self.rois_in_case = []
        if self.lstRSFile:
            self.get_rois_from_RT()

    def get_rois_from_RT(self):
        self.RS_struct = pydicom.read_file(self.lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        for Structures in self.ROI_Structure:
            if Structures.ROIName not in self.rois_in_case:
                self.rois_in_case.append(Structures.ROIName)

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

    def get_images_and_mask(self):
        # Working on the RS structure now
        # The array is sized based on 'ConstPixelDims'
        # ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        self.dicom_handle = self.reader.Execute()
        # slice_location_key = "0020|1041"
        sop_instance_UID_key = "0008|0018"
        self.SOPInstanceUIDs = [self.reader.GetMetaData(i,sop_instance_UID_key) for i in range(self.dicom_handle.GetDepth())]
        self.ArrayDicom = sitk.GetArrayFromImage(self.dicom_handle)
        self.image_size_1, self.image_size_2, _ = self.dicom_handle.GetSize()


    def get_mask_for_contour(self,i):
        self.Liver_Locations = self.RS_struct.ROIContourSequence[i].ContourSequence
        return self.Contours_to_mask()

    def Contours_to_mask(self):
        mask = np.zeros([len(self.dicom_names), self.image_size_1, self.image_size_2], dtype='int8')
        Contour_data = self.Liver_Locations
        ShiftCols, ShiftRows, _ = [float(i) for i in self.reader.GetMetaData(0,"0020|0032").split('\\')]
        # ShiftCols = self.ds.ImagePositionPatient[0]
        # ShiftRows = self.ds.ImagePositionPatient[1]
        PixelSize = self.dicom_handle.GetSpacing()[0]
        Mag = 1 / PixelSize
        mult1 = mult2 = 1
        if ShiftCols > 0:
            mult1 = -1
        if ShiftRows > 0:
            print('take a look at this one...')
        #    mult2 = -1

        for i in range(len(Contour_data)):
            referenced_sop_instance_uid = Contour_data[i].ContourImageSequence[0].ReferencedSOPInstanceUID
            if referenced_sop_instance_uid not in self.SOPInstanceUIDs:
                print('Error here with instance UID')
                return None
            else:
                slice_index = self.SOPInstanceUIDs.index(referenced_sop_instance_uid)
            cols = Contour_data[i].ContourData[1::3]
            rows = Contour_data[i].ContourData[0::3]
            col_val = [Mag * abs(x - mult1 * ShiftRows) for x in cols]
            row_val = [Mag * abs(x - mult2 * ShiftCols) for x in rows]
            temp_mask = self.poly2mask(col_val, row_val, [self.image_size_1, self.image_size_2])
            mask[slice_index,...][temp_mask > 0] = 1
            #scm.imsave('C:\\Users\\bmanderson\\desktop\\images\\mask_'+str(i)+'.png',mask_slice)

        return mask

    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask

if __name__ == '__main__':
    xxx = 1
