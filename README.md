# This code provides functionality for turning dicom images and RT structures into nifti files as well as turning prediction masks back into RT structures
# Installation guide
    pip install DicomRTTool
# Highly recommend to go through the jupyter notebook in the Examples folder

#### If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!
##### Please consider using the .write_parallel if you have many patients
##### Ring update allows for multiple rings to be represented correctly

![multiple_rings.png](./Images/multiple_rings.png)

Various utilities created to help with the interpretation of dicom images/RT Structures

RT Structure and dicom conversion to numpy arrays

Works on oblique images for masks and predictions*
