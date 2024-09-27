from src.DicomRTTool.ReaderWriter import DicomReaderWriter
import os
import pydicom

reader = DicomReaderWriter()
base_path = r'C:\Users\Markb\OneDrive\Desktop\20240927__MIM7_RADONC__100094327796'
output_path = r'C:\Users\Markb\OneDrive\Desktop\100094327796'
reader.down_folder()
for i in reader.series_instances_dictionary.values():
    series_uid = i.SeriesInstanceUID

    for f in i.files:

    x = 1