import dicom, os, copy
import numpy as np
from dicom.tag import Tag
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
    def __init__(self,path='',rewrite_RT_file=False,get_images_mask=True, associations={}):
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
        for roi in Contour_Names:
            if roi not in self.associations:
                self.associations[roi] = roi
        self.Contour_Names = Contour_Names

        # And this is making a mask file
        self.mask = np.zeros([self.image_size_1, self.image_size_2, len(self.lstFilesDCM), len(self.Contour_Names)],
                             dtype='float32')

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
        i = 0
        for ROI_Name in found_rois.keys():
            if found_rois[ROI_Name]['Roi_Number'] in self.structure_references:
                index = self.structure_references[found_rois[ROI_Name]['Roi_Number']]
                mask = self.get_mask_for_contour(index)
                self.mask[..., i][mask == 1] = 1
                i += 1
        self.mask = np.transpose(self.mask, axes=(2, 0, 1, 3))
        return None

    def Make_Contour_From_directory(self,PathDicom):
        self.prep_data(PathDicom)
        if self.rewrite_RT_file:
            self.rewrite_RT()
        if self.get_images_mask:
            self.get_images_and_mask()
        for roi in self.rois_in_case:
            if roi not in self.all_rois:
                self.all_rois.append(roi)
        return None

    def prep_data(self,PathDicom):
        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.lstRSFile = []
        self.Dicom_info = []

        fileList = []
        for dirName, dirs, fileList in os.walk(PathDicom):
            break
        # fileList = [i for i in fileList if i.find('RT') == 0 or i.find('RS') == 0] #
        for filename in fileList:
            try:
                ds = dicom.read_file(os.path.join(dirName,filename))
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
            self.RefDs = dicom.read_file(self.lstFilesDCM[0])
        self.mask_exist = False
        self.rois_in_case = []
        if self.lstRSFile:
            self.get_rois_from_RT()

    def get_rois_from_RT(self):
        self.RS_struct = dicom.read_file(self.lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        for Structures in self.ROI_Structure:
            if Structures.ROIName not in self.rois_in_case:
                self.rois_in_case.append(Structures.ROIName)

    def rewrite_RT(self, lstRSFile=None):
        if lstRSFile is not None:
            self.RS_struct = dicom.read_file(lstRSFile)
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
        if self.lstRSFile:
            checking_mult = dicom.read_file(self.lstRSFile)
            checking_mult = round(checking_mult.ROIContourSequence[0].ContourSequence[0].ContourData[2],2)
        self.image_size_1 = self.Dicom_info[0].pixel_array.shape[0]
        self.image_size_2 = self.Dicom_info[0].pixel_array.shape[1]
        self.ArrayDicom = np.zeros([self.image_size_1, self.image_size_2, len(self.lstFilesDCM)], dtype='float32')

        # loop through all the DICOM files
        self.slice_locations = []
        self.slice_info = np.zeros([len(self.lstFilesDCM)])
        self.SOPClassUID_temp = {}
        self.mult = 1
        # This makes the dicom array of 'real' images
        for filenameDCM in self.lstFilesDCM:
            # read the file
            self.ds = self.Dicom_info[self.lstFilesDCM.index(filenameDCM)]
            # store the raw image data
            if self.ds.pixel_array.shape[0] != self.image_size_1:
                print('Size issue')
            else:
                im = self.ds.pixel_array
            # im[im<200] = 200 #Don't know what the hell these units are, but the min (air) is 0
            self.ArrayDicom[:, :, self.lstFilesDCM.index(filenameDCM)] = im
            # Get slice locations
            slice_location = round(self.ds.ImagePositionPatient[2],2)
            self.slice_locations.append(slice_location)
            self.slice_info[self.lstFilesDCM.index(filenameDCM)] = round(self.ds.ImagePositionPatient[2],3)
            self.SOPClassUID_temp[self.lstFilesDCM.index(filenameDCM)] = self.ds.SOPInstanceUID
        try:
            RescaleIntercept = self.ds.RescaleIntercept
            RescaleSlope = self.ds.RescaleSlope
        except:
            RescaleIntercept = 1
            RescaleSlope = 1
        if self.lstRSFile:
            if min([abs(i - checking_mult) for i in self.slice_locations]) < 0.01:
                self.mult = 1
            elif min([abs(i - checking_mult) for i in self.slice_locations]) < 0.01:
                self.mult = -1
            else:
                print('Slice values are off..')
                self.skip_val = True
                return None
        self.ArrayDicom = (self.ArrayDicom+RescaleIntercept)/RescaleSlope
        indexes = [i[0] for i in sorted(enumerate(self.slice_locations), key=lambda x: x[1])]
        self.lstFilesDCM = list(np.asarray(self.lstFilesDCM)[indexes])
        self.ArrayDicom = self.ArrayDicom[:, :, indexes]
        self.ArrayDicom = np.transpose(self.ArrayDicom,[-1,0,1])
        self.slice_info = self.slice_info[indexes]
        self.SeriesInstanceUID = self.ds.SeriesInstanceUID
        self.slice_locations.sort()
        self.SOPClassUID = {}
        i = 0
        for index in indexes:
            self.SOPClassUID[i] = self.SOPClassUID_temp[index]
            i += 1

    def get_mask_for_contour(self,i):
        self.Liver_Locations = self.RS_struct.ROIContourSequence[i].ContourSequence
        self.Liver_Slices = []
        for contours in self.Liver_Locations:
            data_point = contours.ContourData[2]
            if data_point not in self.Liver_Slices:
                self.Liver_Slices.append(contours.ContourData[2])
        return self.Contours_to_mask()

    def Contours_to_mask(self):
        mask = np.zeros([self.image_size_1, self.image_size_2, len(self.lstFilesDCM)], dtype='float32')
        Contour_data = self.Liver_Locations
        ShiftCols = self.RefDs.ImagePositionPatient[0]
        ShiftRows = self.RefDs.ImagePositionPatient[1]
        PixelSize = self.RefDs.PixelSpacing[0]
        Mag = 1 / PixelSize
        mult1 = mult2 = 1
        if ShiftCols > 0:
            mult1 = -1
        if ShiftRows > 0:
            print('take a look at this one...')
        #    mult2 = -1

        for i in range(len(Contour_data)):
            slice_val = round(Contour_data[i].ContourData[2],2)
            dif = [abs(i * self.mult - slice_val) for i in self.slice_locations]
            try:
                slice_index = dif.index(min(dif))  # Now we know which slice to alter in the mask file
            except:
                print('might have had an issue here..')
                continue
            cols = Contour_data[i].ContourData[1::3]
            rows = Contour_data[i].ContourData[0::3]
            self.col_val = [Mag * abs(x - mult1 * ShiftRows) for x in cols]
            self.row_val = [Mag * abs(x - mult2 * ShiftCols) for x in rows]
            temp_mask = self.poly2mask(self.col_val, self.row_val, [self.image_size_1, self.image_size_2])
            mask[:,:,slice_index][temp_mask > 0] = 1
            #scm.imsave('C:\\Users\\bmanderson\\desktop\\images\\mask_'+str(i)+'.png',mask_slice)

        return mask

    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask

if __name__ == '__main__':
    xxx = 1
