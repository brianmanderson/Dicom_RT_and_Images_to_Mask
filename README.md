## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!
### Please consider using https://github.com/brianmanderson/Dicom_Data_to_Numpy_Arrays if you are wanting a more parallel approach
# Dicom_Utilities

Various utilities created to help with the interpretation of dicom images/RT Structures

RT Structure and dicom conversion to numpy arrays

This code is designed to receive an input path to a folder which contains both dicom images and a single RT structure file

For example, assume a folder exists with dicom files and an RT structure located at 'C:\users\brianmanderson\Patient_1\CT1\' with the roi 'Liver'

The performed action would be Dicom_Image = DicomImagestoData(path='C:\users\brianmanderson\Patient_1\CT1\')

Assume there are 100 images, the generated data will be:
Dicom_Image.ArrayDicom is the image numpy array in the format [# images, rows, cols]

You can then call it to return a mask based on contour names called DicomImage.get_mask(), this takes in a list of Contour Names

You can see the available contour names with

    for roi in DicomImage.rois_in_case:

        print(roi)
    

Example:

    from Image_Array_And_Mask_From_Dicom import DicomImagestoData

    Path = 'C:\users\brianmanderson\Patient_1\CT1\'

    Contour_Names = ['Liver']

    DicomImage = DicomImagestoData(path=Path)
    for roi in DicomImage.rois_in_case:
        print(roi)
    DicomImage.get_mask(Contour_Names)

    mask = DicomImage.mask
