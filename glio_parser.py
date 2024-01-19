"""
This script assume there is a RT structure file for each dicoms directory
    - if there are not RT_structure file for the dicom directory, use the build_hair_dicom_struct.py to use the RT_Structure from 
    the corresponding T1c dicom directory and map the RT_Structure to the FLAIR images
    
"""
from src.DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
# file mangagment 
import os 
# medical image manipulation 
import SimpleITK as sitk

glio_data = '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/'

patients = os.listdir(glio_data)
patients = [p for p in patients if 'GBM' in p]

for patient_name in patients:

    patient_contours = {} # collect the rt_struct file path for each day of the same patient

    patient_dir = os.path.join(glio_data, patient_name)
    patient_days = os.listdir(patient_dir)
    patient_days = [d for d in patient_days if 'Recur' not in d and 'Day' in d] #NOTE: all patient day directory expects to have a struct for now
    for patient_day in patient_days:
        print('Start processing {} for {}'.format(patient_name, patient_day))
        try:
            # root directory of the dicoms files within a patient day
            patient_day_dicoms = os.path.join(patient_dir, patient_day) # '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/GBM108/Day0T1c'

            Dicom_reader = DicomReaderWriter(description=patient_name, arg_max=True)
            Dicom_reader.walk_through_folders(patient_day_dicoms) # need to define in order to use all_roi method
            all_rois = Dicom_reader.return_rois(print_rois=True)  # Return a list of all rois present, and print them

            # filter out the contours that do not belong to this
            all_rois = [roi for roi in all_rois if ('day' not in roi) or 'day0' in roi]

            # designated directory to store the converted nii file for the dicom
            image_dir = os.path.join('/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin-nii/', patient_name, patient_day) # '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin-nii/GBM108/Day0T1c'
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            # designated directory to store the converted nii mask file for the dicom (might have two for each image GTV and the CTV)
            mask_dir = os.path.join('/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin-nii-mask/', patient_name, patient_day) # '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin-nii-mask/GBM108/Day0T1c'
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)

            # convert the format of the dicom image to nii
            Dicom_reader.get_images()
            sitk.WriteImage(Dicom_reader.dicom_handle, os.path.join(image_dir, '{}_{}.nii.gz'.format(patient_name, patient_day)))
            for roi in all_rois:
                print('     processing contour {} for patient {} on day {}'.format(roi, patient_name, patient_day))
                ROI_NAME = roi
                # Print the locations of all RTs with a certain ROI name, automatically lower cased
                Dicom_reader.where_is_ROI(ROIName=ROI_NAME)
                Dicom_reader.set_contour_names_and_associations([ROI_NAME])
                Dicom_reader.get_mask()
                sitk.WriteImage(Dicom_reader.annotation_handle, os.path.join(mask_dir, '{}_{}_{}.nii'.format(patient_name, patient_day, ROI_NAME)))
        
        except:
            print('Error in processing {} for {}'.format(patient_name, patient_day))


print('done')

# PATIENT_NAME = 'GBM108'
# DAY_FOLDER_NAME = 'Day0FLAIR'
# # DAY_FOLDER_NAME = 'Day10T1c'
# # ROI_NAME = 'brainstem'

# patient_day_dicoms = '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/{}/{}'.format(PATIENT_NAME, DAY_FOLDER_NAME)

# Dicom_reader = DicomReaderWriter(description=PATIENT_NAME, arg_max=True)
# Dicom_reader.walk_through_folders(patient_day_dicoms) # need to define in order to use all_roi method
# all_rois = Dicom_reader.return_rois(print_rois=True)  # Return a list of all rois present, and print them

# # filter out the contours that do not belong to this
# all_rois = [roi for roi in all_rois if ('day' not in roi) or 'day0' in roi]

# sitk.WriteImage(Dicom_reader.dicom_handle, '/Users/maxxyouu/Desktop/IAMLAB/Contour_outputs_test/{}_{}.nii.gz'.format(PATIENT_NAME, DAY_FOLDER_NAME))
# for roi in all_rois:
#     print('processing contour {}'.format(roi))
#     ROI_NAME = roi
#     # Print the locations of all RTs with a certain ROI name, automatically lower cased
#     Dicom_reader.where_is_ROI(ROIName=ROI_NAME)
#     Dicom_reader.set_contour_names_and_associations([ROI_NAME])
#     # Dicom_reader.get_images()
#     # Dicom_reader.get_mask()
#     Dicom_reader.get_images_and_mask()
#     sitk.WriteImage(Dicom_reader.annotation_handle, '/Users/maxxyouu/Desktop/IAMLAB/Contour_outputs_test/{}_{}_mask_{}.nii.gz'.format(PATIENT_NAME, DAY_FOLDER_NAME, ROI_NAME))

# print('done')