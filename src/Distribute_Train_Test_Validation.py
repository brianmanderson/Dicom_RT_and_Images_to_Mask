__author__ = 'Brian M Anderson'
# Created on 4/16/2020

import numpy as np
import os
import typing
import pandas as pd


def define_folders_for_excel_sheet(excel_file: typing.Union[str, bytes, os.PathLike],
                                   validation_fraction: float, test_fraction=0,
                                   patientIDcolumnname='PatientID') -> pd.DataFrame:
    """
    The goal of this is to provide a Folder name (Train, Test, or Validation) for each image set based
    on the patient ID
    :param patientIDcolumnname: string of the column name that you want to distribute across
    :param excel_file: the excel file which was created during the niftii writing process
    :param validation_fraction: fraction [0-1] of data to be placed in the validation set
    :param test_fraction: fraction [0-1] of the data to be placed in the test set
    :return: a pandas dataframe of the loaded excel sheet
    """
    assert os.path.exists(excel_file), FileExistsError("File not found")
    data_df = pd.read_excel(excel_file, engine='openpyxl')

    unique_patients = np.unique(data_df[patientIDcolumnname].values)
    total_patients = len(unique_patients)
    patient_image_dictionary = {}
    not_distributed = []
    validation_mrns = []
    test_mrns = []
    for patient_MRN in unique_patients:
        patient_indexes = data_df.loc[data_df[patientIDcolumnname] == patient_MRN].index.values
        patient_image_dictionary[patient_MRN] = {'Indexes': patient_indexes, 'Folder': None}
        """
        Check to see if we have already assigned a Train/Test/Validation folder to this patient's MRN
        We do not want to break up a patient's images!
        """
        folder = None
        for patient_folder in data_df.Folder[patient_indexes]:
            if patient_folder == 'Train':
                folder = 'Train'
                break
            elif patient_folder == 'Validation':
                folder = 'Validation'
                validation_mrns.append(patient_MRN)
                break
            elif patient_folder == 'Test':
                folder = 'Test'
                test_mrns.append(patient_MRN)
                break
        patient_image_dictionary[patient_MRN]['Folder'] = folder
        """
        If this patient was already assigned a folder, assign that folder to the other images and save
        """
        if folder is not None:
            rewrite = False
            for index in patient_indexes:
                if data_df.Folder[index] != folder:
                    data_df.loc[data_df.index == index, 'Folder'] = folder
                    rewrite = True
            if rewrite:
                data_df.to_excel(excel_file, index=0)
        else:
            not_distributed.append(patient_MRN)
    not_distributed = np.asarray(not_distributed)
    if len(not_distributed) > 0:
        """
        Shuffle the indexes up that haven't been distributed
        """
        perm = np.arange(len(not_distributed))
        np.random.shuffle(perm)
        not_distributed = not_distributed[perm]
        """
        For each patient, check to see if they should go in validation or test, overflow goes into train
        """
        for patient_MRN in not_distributed:
            patient_indexes = patient_image_dictionary[patient_MRN]['Indexes']
            number_to_validation = int(total_patients * validation_fraction) - len(validation_mrns)
            number_to_test = int(total_patients * test_fraction) - len(test_mrns)
            folder = 'Train'
            if number_to_validation > 0:
                folder = 'Validation'
                validation_mrns.append(patient_MRN)
            elif number_to_test > 0:
                folder = 'Test'
                test_mrns.append(patient_MRN)
            for index in patient_indexes:
                data_df.loc[data_df.index == index, 'Folder'] = folder
        data_df.to_excel(excel_file, index=0)
    return data_df


def distribute(niftii_path: typing.Union[str, bytes, os.PathLike],
               excel_file: typing.Union[str, bytes, os.PathLike],
               validation_fraction: float, test_fraction=0, patientIDcolumnname='PatientID'):
    """
    :param patientIDcolumnname: string of the column name that you want to distribute across
    :param niftii_path: path to the niftii files
    :param excel_file: the excel file which was created during the niftii writing process
    :param validation_fraction: fraction [0-1] of data to be placed in the validation set
    :param test_fraction: fraction [0-1] of the data to be placed in the test set
    :return:
    """
    train_path = os.path.join(niftii_path, 'Train')
    test_path = os.path.join(niftii_path, 'Test')
    validation_path = os.path.join(niftii_path, 'Validation')
    for out_path in [train_path, test_path, validation_path]:
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    data_df = define_folders_for_excel_sheet(excel_file=excel_file, patientIDcolumnname=patientIDcolumnname,
                                             validation_fraction=validation_fraction, test_fraction=test_fraction)

    file_list = [i for i in os.listdir(niftii_path) if i.find('Overall_Data') == 0]
    '''
    Group all of the images up based on their MRN, we don't want to contaminate other groups
    '''
    for image_file in file_list:
        iteration = image_file.split('_')[-1].split('.')[0]
        out_folder = data_df.Folder[data_df.Iteration == int(iteration)].values[0]
        os.rename(os.path.join(niftii_path, image_file), os.path.join(niftii_path, out_folder, image_file))
        label_file = image_file.replace('_{}'.format(iteration), '_y{}'.format(iteration)).replace('Overall_Data',
                                                                                                   'Overall_mask')
        os.rename(os.path.join(niftii_path, label_file), os.path.join(niftii_path, out_folder, label_file))
    return None


if __name__ == '__main__':
    pass
