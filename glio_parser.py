from src.DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
# importing neccessary libraries 

# file mangagment 
import os 
import zipfile
from six.moves import urllib

# array manipulation and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom as dicom

# medical image manipulation 
import SimpleITK as sitk

glio_data = '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/'

patients = os.listdir(glio_data)
patients = [p for p in patients if 'GBM' in p]

for patient_name in patients:

    patient_contours = {} # collect the rt_struct file path for each day of the same patient

    patient_dir = os.path.join(glio_data, patient_name)
    patient_days = os.listdir(patient_dir)
    patient_days = [d for d in patient_days if 'Recur' not in d] #NOTE: all patient day directory expects to have a struct for now
    for patient_day in patient_days:
        # root directory of the dicoms files within a patient day
        patient_day_dicoms = os.path.join(patient_dir, patient_day) # '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/GBM108/Day0T1c'

        Dicom_reader = DicomReaderWriter(description=patient_name, arg_max=True)
        Dicom_reader.walk_through_folders(patient_day_dicoms) # need to define in order to use all_roi method
        all_rois = Dicom_reader.return_rois(print_rois=True)  # Return a list of all rois present, and print them

        # filter out the contours that do not belong to this
        all_rois = [roi for roi in all_rois if ('day' not in roi) or 'day0' in roi]

        # convert the format of the dicom image to nii  TODO: move the files to the correct directory
        sitk.WriteImage(Dicom_reader.dicom_handle, '/Users/maxxyouu/Desktop/IAMLAB/Contour_outputs_test/{}_{}.nii.gz'.format(patient_name, patient_day))
        for roi in all_rois:
            print('processing contour {} for patient {} on day {}'.format(roi, patient_name, patient_day))
            ROI_NAME = roi
            # Print the locations of all RTs with a certain ROI name, automatically lower cased
            Dicom_reader.where_is_ROI(ROIName=ROI_NAME)
            Dicom_reader.set_contour_names_and_associations([ROI_NAME])
            # Dicom_reader.get_images()
            # Dicom_reader.get_mask()
            Dicom_reader.get_images_and_mask()
            # TODO: move the files to the correct directory
            sitk.WriteImage(Dicom_reader.annotation_handle, '/Users/maxxyouu/Desktop/IAMLAB/Contour_outputs_test/{}_{}_mask_{}.nii.gz'.format(PATIENT_NAME, DAY_FOLDER_NAME, ROI_NAME))
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