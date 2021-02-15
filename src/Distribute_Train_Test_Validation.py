__author__ = 'Brian M Anderson'
# Created on 4/16/2020

import numpy as np
import os
import pandas as pd


def distribute(description,niftii_path,excel_file):
    final_out_dict = {'MRN': [], 'Path': [], 'Iteration': [], 'Folder': []}
    if os.path.exists(excel_file):
        data = pd.read_excel(excel_file, engine='openpyxl')
        data = data.to_dict()
        for key in final_out_dict.keys():
            for index in data[key]:
                final_out_dict[key].append(str(data[key][index]))

    train_path = os.path.join(niftii_path, description,'Train')
    test_path = os.path.join(niftii_path, description,'Test')
    validation_path = os.path.join(niftii_path,description,'Validation')
    for out_path in [train_path,test_path,validation_path]:
        if not os.path.exists(out_path):
            os.makedirs(out_path)

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
    split_train = int(len(patient_image_keys)/6)
    for xxx in range(split_train):
        for iteration in image_dict[patient_image_keys[xxx]]:
            image_file = file_dict[iteration]
            os.rename(os.path.join(niftii_path,image_file),os.path.join(test_path,image_file))
            label_file = image_file.replace('_{}'.format(iteration),'_y{}'.format(iteration)).replace('Overall_Data','Overall_mask')
            os.rename(os.path.join(niftii_path, label_file), os.path.join(test_path, label_file))
    for xxx in range(split_train,int(split_train*2)):
        for iteration in image_dict[patient_image_keys[xxx]]:
            image_file = file_dict[iteration]
            os.rename(os.path.join(niftii_path,image_file),os.path.join(validation_path,image_file))
            label_file = image_file.replace('_{}'.format(iteration),'_y{}'.format(iteration)).replace('Overall_Data','Overall_mask')
            os.rename(os.path.join(niftii_path, label_file), os.path.join(validation_path, label_file))
    for xxx in range(int(split_train*2),len(perm)):
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
