__author__ = 'Brian M Anderson'
# Created on 4/16/2020


Contour_Names = ['Lung (Left)', 'Lung (Right)']
image_path = r'\\mymdafiles\di_data1\Morfeus\Lung_Exports\From_Raystation'
'''
This will print if any rois are missing at certain locations
'''
check_rois = False
if check_rois:
    from .Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack
    Dicom_Reader = Dicom_to_Imagestack(get_images_mask=False,Contour_Names=Contour_Names)
    Dicom_Reader.down_folder(image_path)

'''
This will turn the dicom into niftii files
'''
nifti_path = r'\\mymdafiles\di_data1\Morfeus\Lung_Exports\Nifti_Files'
write_files = False
if write_files:
    from .Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, os
    Dicom_Reader = Dicom_to_Imagestack(get_images_mask=False, Contour_Names=Contour_Names, desc='Test')
    Dicom_Reader.down_folder(image_path)
    Dicom_Reader.write_parallel(out_path=nifti_path, excel_file=os.path.join('.', 'MRN_Path_To_Iteration.xlsx'))

'''
Distribute the nifti files to other folders
'''