import pydicom as dicom
import os

def get_roi_names(contour_data): 
    """
    This function will return the names of different contour data, 
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the 
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names

def get_roi_contour_ds(rt_sequence, index):
    """
    Extract desired ROI contour datasets
    from RT Sequence.

    E.g. rt_sequence can have contours for different parts of the brain
    such as ventricles, tumor, etc...

    You can use get_roi_names to find which index to use

    Inputs:
        rt_sequence (dicom.dataset.FileDataset): Contour file dataset, what you get
                                                 after reading contour DICOM file
        index (int): Index for ROI Sequence
    Return:
        contours (list): list of ROI contour dicom.dataset.Dataset s
    """
    # index 0 means that we are getting RTV information
    ROI = rt_sequence.ROIContourSequence[index]
    # get contour datasets in a list
    contours = [contour for contour in ROI.ContourSequence]
    return contours

def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence 
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html
    
    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get .dcm contour file
    fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
    n = 0
    contour_file = None
    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            contour_file = fpath.split('/')[-1]
            n += 1
    if n > 1: print("There are multiple contour files, returning the last one!")
    if contour_file is None: print("No contour file found in directory")
    return contour_file

def get_corresponding_img_ids(contour_datasets, path):

    # the list of ReferencedSOPInstanceUID for the new RT struct of the flair image
    new_ReferencedSOPInstanceUID = []
    new_coords = []
    for contour_dataset in contour_datasets:

        contour_coord = contour_dataset.ContourData
        # x, y, z coordinates of the contour in mm
        coord = []
        for i in range(0, len(contour_coord), 3):
            coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

        # extract the image id corresponding to given countour
        # read that dicom file
        img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID

        # search through the flair dicom directory and accumulate the necessary data to find out the right image id for the ReferencedSOPInstanceUID of the contour slice
        dcms = os.listdir(path)
        slices_ = []
        for dcm in dcms:
            if '.DCM' not in dcm:
                continue
            dcm_slice = dicom.read_file(os.path.join(path, dcm)) 
            slices_.append((dcm_slice, dcm_slice.SliceLocation, contour_dataset.ContourData[-1], dcm_slice.SOPInstanceUID, abs(-dcm_slice.SliceLocation - contour_dataset.ContourData[-1])))
            if dcm_slice.SOPInstanceUID == img_ID:
                print(dcm_slice.SliceLocation, contour_dataset.ContourData[-1])
                print('NOT SUPPOSED TO BE HERE')
                # NOTE: dcm_slice.SeriesInstanceUID = 
                filename = dcm
                break

        slices_ = sorted(slices_, key=lambda x: x[-1])
        # print(slices_[0][1], slices_[0][2]) # TO EXAMINE THE LOCATION
        new_img_ID = slices_[0][3]
        # dcm_slice = slices_[0][0]
        
        # if float(c[-1]) < 0
        # new_coords.append([ (c[0], c[1], ) for c in coord])
        

        # map the new reference instance id in the flair image folder
        new_ReferencedSOPInstanceUID.append(new_img_ID)

    return new_ReferencedSOPInstanceUID, dcm_slice.SeriesInstanceUID


def construct_corresponding_flair_dir(patient_day, patient_dir):

    patient_day_t2f = patient_day
    if 'CT' in patient_day:
        patient_day_t2f = patient_day[:-len('CT')] + 'FLAIR'
    if 'T1c' in patient_day:
        patient_day_t2f = patient_day[:-len('T1c')] + 'FLAIR'

    corresponding_t2f_dir = os.path.join(patient_dir, patient_day_t2f)    
    return corresponding_t2f_dir


def construct_RTStruct_for_t2f_dicoms(params):
    contour_names = params['contour_names']
    rt_sequence = params['rt_sequence']
    patient_day_dicoms_Flair = params['patient_day_dicoms_Flair']

    for roi_index, cname in enumerate(contour_names): # TODO: note that contour that is in different time point other than day 0 seem to require coordinate registration to work, need to wait for james to give me the struct for process.
        print('     processing contour {}'.format(cname))
        # get contour datasets with index idx
        contour_datasets = get_roi_contour_ds(rt_sequence=rt_sequence, index=roi_index)

        # construct mask dictionary
        new_ReferencedSOPInstanceUIDs, new_seriesInstanceUID = get_corresponding_img_ids(contour_datasets, patient_day_dicoms_Flair)

        # write the new reference id to the rt struct
        # TODO: overwrite the rt_sequence.StudyInstanceUID if something went wrong, should be returned from get_corresponding_img_ids
        rt_sequence.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID = new_seriesInstanceUID
        for i, cd in enumerate(rt_sequence.ROIContourSequence[roi_index].ContourSequence):
            before = cd.ContourImageSequence[0].ReferencedSOPInstanceUID
            cd.ContourImageSequence[0].ReferencedSOPInstanceUID = new_ReferencedSOPInstanceUIDs[i]
            # print('before: {} after: {}'.format(before, cd.ContourImageSequence[0].ReferencedSOPInstanceUID))

    # return a new rt_sequence that compatible with the t2f dicoms
    return rt_sequence

glio_data = '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/'

patients = os.listdir(glio_data)
patients = [p for p in patients if 'GBM' in p]

for patient_name in patients:

    patient_contours = {} # collect the rt_struct file path for each day of the same patient

    patient_dir = os.path.join(glio_data, patient_name)
    patient_days = os.listdir(patient_dir)
    patient_days = [d for d in patient_days if 'Day' in d and 'T1c' in d and 'Recur' not in d]
    for patient_day in patient_days:
        # root directory of the dicoms files within a patient day
        patient_day_dicoms_nonT2F = os.path.join(patient_dir, patient_day) # '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/GBM108/Day0T1c'

        contours_file = get_contour_file(patient_day_dicoms_nonT2F)
        contour_file_path = os.path.join(patient_day_dicoms_nonT2F, contours_file)
        
        rt_sequence = dicom.read_file(contour_file_path)   
        contour_names = get_roi_names(rt_sequence)
        contour_names = [name for name in contour_names if 'Day' not in name]

        # contruct the corresponding FLAIR directory
        t2f_dir = construct_corresponding_flair_dir(patient_day, patient_dir)
        print('start processing for {}'.format(t2f_dir))
        params = {
            'contour_names': contour_names,
            'rt_sequence': rt_sequence,
            'patient_day_dicoms_Flair': t2f_dir
        }
        new_rt_sequence = construct_RTStruct_for_t2f_dicoms(params)
        new_rt_sequence.save_as(os.path.join(t2f_dir, contours_file[:-4]+'_flair.dcm'))
        print('Finish processing for {}'.format(os.path.join(t2f_dir, contours_file[:-4]+'_flair.dcm')))


# PATIENT_NAME = 'GBM149'
# DAY_FOLDER_FLAIR_NAME = 'Day0FLAIR'
# DAY_FOLDER_T1C_NAME = 'Day0T1c'

# patient_day_dicoms_T1c = '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/{}/{}'.format(PATIENT_NAME, DAY_FOLDER_T1C_NAME)
# patient_day_dicoms_Flair = '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/{}/{}'.format(PATIENT_NAME, DAY_FOLDER_FLAIR_NAME)
# contours_file = get_contour_file(patient_day_dicoms_T1c)
# contour_file_path = os.path.join(patient_day_dicoms_T1c, contours_file)

# rt_sequence = dicom.read_file(contour_file_path)   
# contour_names = get_roi_names(rt_sequence)
# contour_names = [name for name in contour_names if 'Day' not in name]

# for roi_index, contour_name in enumerate(contour_names): # TODO: note that contour that is in different time point other than day 0 seem to require coordinate registration to work, need to wait for james to give me the struct for process.

#     # get contour datasets with index idx
#     contour_datasets = get_roi_contour_ds(rt_sequence=rt_sequence, index=roi_index)

#     # construct mask dictionary
#     new_ReferencedSOPInstanceUIDs, new_seriesInstanceUID = get_corresponding_img_ids(contour_datasets, patient_day_dicoms_Flair)

#     # write the new reference id to the rt struct
#     # TODO: overwrite the rt_sequence.StudyInstanceUID if something went wrong, should be returned from get_corresponding_img_ids
#     rt_sequence.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID = new_seriesInstanceUID
#     for i, cd in enumerate(rt_sequence.ROIContourSequence[roi_index].ContourSequence):
#         before = cd.ContourImageSequence[0].ReferencedSOPInstanceUID
#         cd.ContourImageSequence[0].ReferencedSOPInstanceUID = new_ReferencedSOPInstanceUIDs[i]
#         # print('before: {} after: {}'.format(before, cd.ContourImageSequence[0].ReferencedSOPInstanceUID))

# rt_sequence.save_as(os.path.join(patient_day_dicoms_Flair, contours_file[:-4]+'_flair.dcm'))
# print('done')
# save the rt struct with a new name with a new location
