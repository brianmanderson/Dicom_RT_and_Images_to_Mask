from src.DicomRTTool.ReaderWriter import DicomReaderWriter, plot_scroll_Image


path = r'C:\Users\b5anderson\Desktop\Modular_Projects\Dicom_RT_and_Images_to_Mask\Examples\Example_Data\Image_Data'
reader = DicomReaderWriter(description='Examples', arg_max=True)
reader.walk_through_folders(path)