from src.DicomRTTool import DicomReaderWriter, plot_scroll_Image, sitk
import os
from NiftiResampler.ResampleTools import ImageResampler

out_path = r'C:\Users\b5anderson\Desktop\Modular_Projects\Dicom_RT_and_Images_to_Mask\Examples\Example_Data\Image_Data'
reader = DicomReaderWriter(description='Examples', arg_max=True, verbose=False)
path = r'O:\DICOM\Test2'
reader.down_folder(path)
for series_uid in reader.images_dictionary.keys():
    os.makedirs(os.path.join(path, series_uid))
    files = reader.return_files_from_UID(series_uid)
    for file in files:
        file_name = os.path.split(file)[-1]
        os.rename(file, os.path.join(path, series_uid, file_name))