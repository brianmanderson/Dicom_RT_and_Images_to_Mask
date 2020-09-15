__author__ = 'Brian M Anderson'
# Created on 9/15/2020


from setuptools import setup

setup(
    name='DicomRTTools',
    author='Brian Mark Anderson',
    email='bmanderson@mdanderson.org',
    version='0.0.2',
    description='Tools for reading dicom files, RT structures, and dose files, as well as tools for '
                'converting numpy prediction masks back to an RT structure',
    py_modules=['DicomRTTool'],
    package_dir={'': 'src'}
)
