__author__ = 'Brian M Anderson'
# Created on 9/15/2020


from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='DicomRTTool',
    author='Brian Mark Anderson',
    author_email='bmanderson@mdanderson.org',
    version='0.2.13',
    description='Tools for reading dicom files, RT structures, and dose files, as well as tools for '
                'converting numpy prediction masks back to an RT structure',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'DicomRTTool': 'src/DicomRTTool'},
    packages=['DicomRTTool'],
    include_package_data=True,
    url='https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    install_requires=required,
)
