# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:55:24 2023

@author: tuank
"""
import glob
import pandas as pd

def merge_data(meta_path, annotaion_path):
    dict_folder={'folder_location':[], 'folder_name':[], 'xml_location':[], 'xml_name':[]}
    for folder_location in glob.glob(meta_path + '/*'):
        for file_name in glob.glob(folder_location + '/*/*'):
            if 'PET' in file_name.split('Lung-PET-CT-Dx')[-1] or 'pet' in file_name.split('Lung-PET-CT-Dx')[-1]:
                folder_name = folder_location.split('\\')[-1]
                xml_name = folder_name.split('-')[-1]
                xml_location = glob.glob(annotaion_path + '/' + xml_name)
                if len(xml_location) > 0 :
                    xml_location = xml_location[0]
                    dict_folder['folder_location'].append(folder_location)
                    dict_folder['folder_name'].append(folder_name)
                    dict_folder['xml_location'].append(xml_location)
                    dict_folder['xml_name'].append(xml_name)
                    break
    df = pd.DataFrame.from_dict(dict_folder)
    df.to_csv('PET-CT-merged.csv')

if __name__ == "__main__":
    meta_path='D:/Spyder/Final_Project/LUNG-PET/manifest-1608669183333/Lung-PET-CT-Dx'
    annotaion_path = 'D:/Spyder/Final_Project/LUNG-PET/Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020/Annotation'
    merge_data(meta_path, annotaion_path)