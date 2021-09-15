__author__ = 'Brian M Anderson'
# Created on 4/16/2020

import numpy as np
import os
import typing
import pandas as pd


def distribute(niftii_path: typing.Union[str, bytes, os.PathLike],  excel_file: typing.Union[str, bytes, os.PathLike],
               validation_fraction: float, test_fraction=0):
    """
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

    assert os.path.exists(excel_file), FileExistsError("File not found")
    data_df = pd.read_excel(excel_file, engine='openpyxl')

    total_patients = np.unique(data_df.PatientID.values)
    patient_image_dictionary = {}
    for patient_MRN in total_patients:
        patient_indexes = data_df.loc[data_df.PatientID == patient_MRN].index.values
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
                break
            elif patient_folder == 'Test':
                folder = 'Test'
                break
        patient_image_dictionary[patient_MRN]['Folder'] = folder
        """
        If this patient was already assigned a folder, assign that folder to the other images and save
        """
        if folder is not None:
            for index in patient_indexes:
                data_df.Folder[index] = folder
            data_df.to_excel(excel_file, index=0)
        xxx = 1
    not_distributed = data_df.loc[pd.isnull(data_df.Folder)]
    indexes_not_distributed = not_distributed.index.values
    validation_indexes = data_df.loc[data_df.Folder == 'Validation'].index.values
    test_indexes = data_df.loc[data_df.Folder == 'Test'].index.values

    if len(indexes_not_distributed) > 0:
        """
        Shuffle the indexes up that haven't been distributed
        """
        perm = np.arange(len(indexes_not_distributed))
        np.random.shuffle(perm)
        indexes_not_distributed = indexes_not_distributed[perm]
        number_to_validation = int(total_patients * validation_fraction) - len(validation_indexes)
        number_to_test = int(total_patients * test_fraction) - len(test_indexes)
        for index in indexes_not_distributed:
            if number_to_validation > 0:
                data_df['Folder'][index] = 'Validation'
                number_to_validation -= 1
            elif number_to_test > 0:
                data_df['Folder'][index] = 'Test'
                number_to_test -= 1
            else:
                data_df['Folder'][index] = 'Train'
        data_df.to_excel(excel_file, index=0)

    file_list = [i for i in os.listdir(niftii_path) if i.find('Overall_Data') == 0]
    '''
    Group all of the images up based on their MRN, we don't want to contaminate other groups
    '''

    image_dict = dict()
    file_dict = dict()
    for file in file_list:
        iteration = file.split('_')[-1].split('.')[0]
        file_dict[iteration] = file
        index = final_out_dict['Iteration'].index(iteration)
        patient_id = final_out_dict['MRN'][index]
        if patient_id not in image_dict:
            image_dict[patient_id] = [iteration]
        else:
            image_dict[patient_id].append(iteration)

    patient_image_keys = list(image_dict.keys())
    perm = np.arange(len(patient_image_keys))
    np.random.shuffle(perm)
    patient_image_keys = list(np.asarray(patient_image_keys)[perm])
    total_patients = len(patient_image_keys)
    split_train = int(total_patients * train_faction)
    split_validation = int(total_patients * validation_fraction)
    for xxx in range(split_train):
        for iteration in image_dict[patient_image_keys[xxx]]:
            image_file = file_dict[iteration]
            os.rename(os.path.join(niftii_path,image_file),os.path.join(test_path,image_file))
            label_file = image_file.replace('_{}'.format(iteration),'_y{}'.format(iteration)).replace('Overall_Data','Overall_mask')
            os.rename(os.path.join(niftii_path, label_file), os.path.join(test_path, label_file))
    for xxx in range(split_train, split_train + split_validation):
        for iteration in image_dict[patient_image_keys[xxx]]:
            image_file = file_dict[iteration]
            os.rename(os.path.join(niftii_path,image_file),os.path.join(validation_path,image_file))
            label_file = image_file.replace('_{}'.format(iteration),'_y{}'.format(iteration)).replace('Overall_Data','Overall_mask')
            os.rename(os.path.join(niftii_path, label_file), os.path.join(validation_path, label_file))
    for xxx in range(split_train + split_validation, total_patients):
        for iteration in image_dict[patient_image_keys[xxx]]:
            image_file = file_dict[iteration]
            os.rename(os.path.join(niftii_path,image_file),os.path.join(train_path,image_file))
            label_file = image_file.replace('_{}'.format(iteration),'_y{}'.format(iteration)).replace('Overall_Data','Overall_mask')
            os.rename(os.path.join(niftii_path, label_file), os.path.join(train_path, label_file))
    output_dict = {'MRN':[],'Path':[],'Iteration':[],'Folder':[]}
    keys = ['MRN','Path','Iteration']
    for title, folder in zip(['Train','Validation','Test'],[train_path, validation_path, test_path]):
        file_list = [i for i in os.listdir(folder) if i.find('Overall_Data') == 0]
        for file in file_list:
            iteration = file.split('_')[-1].split('.')[0]
            index = final_out_dict['Iteration'].index(iteration)
            for key in keys:
                output_dict[key].append(final_out_dict[key][index])
            output_dict['Folder'].append(title)
    output_dict['Iteration'] = [int(i) for i in output_dict['Iteration']]
    df = pd.DataFrame(output_dict)
    df.to_excel(excel_file,index=0)
    return None


if __name__ == '__main__':
    pass
