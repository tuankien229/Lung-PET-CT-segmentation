import pydicom as pydcm
import torchio as tio
import pandas as pd
import glob
import os
from bs4 import BeautifulSoup

def get_tumor_box(xml_path: str, normalize = False):
    '''
    

    Parameters
    ----------
    data_path : String
        path of annotation.
    img_info : pandas.DataFrame
        DESCRIPTION.
    normalize : bool, optional
        Set normalize of bounding box. The default is False.

    Returns
    -------
    img_info : pandas.DataFrame
        Dataframe after save bounding box of cancer .

    '''
    file = open(xml_path[0], 'r').read()
    bs = BeautifulSoup(file, 'xml')
    width = float(bs.find('width').get_text())
    height = float(bs.find('height').get_text())
    xmin = float(bs.find('xmin').get_text())
    ymin = float(bs.find('ymin').get_text())
    xmax = float(bs.find('xmax').get_text())
    ymax = float(bs.find('ymax').get_text())
    if normalize:
        xmin = xmin/width
        ymin = ymin/height
        xmax = xmax/width
        ymax = ymax/height
    tumor_box = [xmin, xmax, ymin, ymax]
    return tumor_box


df = pd.read_csv('D:\Spyder\Final_Project\LUNG-PET\PET-CT-merged.csv')
for i in range(len(df)):
    folder_path = df['folder_location'][i]
    xml_path = df['xml_location'][i]
    for folder in glob.glob(folder_path + '/*'):
        if len(glob.glob(folder + '/*')) > 3:
            for dcm_folder in glob.glob(folder + '/*'):
                for file_name in glob.glob(dcm_folder + '/*'):
                    ds = pydcm.read_file(file_name)
                    xml_file_match = str(ds.SOPInstanceUID)
                    xml_files = glob.glob(xml_path + '/' + xml_file_match + '*')
                    if len(xml_files) != 0:
                        print(get_tumor_box(glob.glob(xml_path + '/' + xml_file_match + '*')))
                        

    break

