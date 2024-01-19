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

PATIENT_NAME = 'GBM149'
DAY_FOLDER_NAME = 'Day0FLAIR'
# DAY_FOLDER_NAME = 'Day10T1c'

patient_day_dicoms = '/Users/maxxyouu/Desktop/IAMLAB/GBMforAprilV2-origin/{}/{}'.format(PATIENT_NAME, DAY_FOLDER_NAME)

Dicom_reader = DicomReaderWriter(description='GBM108', arg_max=True)
Dicom_reader.walk_through_folders(patient_day_dicoms) # need to define in order to use all_roi method
all_rois = Dicom_reader.return_rois(print_rois=True)  # Return a list of all rois present, and print them

# Print the locations of all RTs with a certain ROI name, automatically lower cased
Dicom_reader.where_is_ROI(ROIName='T1_GTV')

Dicom_reader.set_contour_names_and_associations(['T1_GTV'])
# Dicom_reader.get_mask()
Dicom_reader.get_images()
Dicom_reader.get_mask()

sitk.WriteImage(Dicom_reader.dicom_handle, '/Users/maxxyouu/Desktop/IAMLAB/Contour_outputs_test/{}_{}.nii.gz'.format(PATIENT_NAME, DAY_FOLDER_NAME))
sitk.WriteImage(Dicom_reader.annotation_handle, '/Users/maxxyouu/Desktop/IAMLAB/Contour_outputs_test/{}_{}_mask.nii.gz'.format(PATIENT_NAME, DAY_FOLDER_NAME))
print('done')