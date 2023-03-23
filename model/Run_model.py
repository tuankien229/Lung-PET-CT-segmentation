# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 15:54:15 2022

@author: tuank
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchio as tio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from model.UNET3D import Unet3DPP
def Normalize(image : np.ndarray):
    return (image - np.min(image))/(np.max(image) - np.min(image))

def load_model(in_channels = 1, out_channels = 32, n_classes = 1):
    model = Unet3DPP(in_channels=in_channels, out_channels=out_channels, n_classes=n_classes)
    device = 'cuda' 
    if torch.cuda.device_count() >= 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load('checkpoint_88.pth'))
    return model

def input_process(dicom_array):
    # print('Success load dicom')
    dicom_tensor = torch.Tensor(dicom_array)
    dicom_tensor = torch.unsqueeze(dicom_tensor, dim = 0)
    dicom_tio = tio.ScalarImage(tensor = dicom_tensor)
    resize_to_128 = tio.Resize((128,128,128))
    dicom_tio = resize_to_128(dicom_tio)
    dicom_np = np.array(dicom_tio)
    dicom_np = np.squeeze(dicom_np, axis = 0)
    dicom_np = np.moveaxis(dicom_np, (0,1,2), (2,1,0))
    dicom_np = Normalize(dicom_np)
    dicom_tensor = torch.Tensor(dicom_np)
    dicom_tensor = torch.unsqueeze(dicom_tensor, dim = 0)
    subjects = tio.Subject(image = tio.ScalarImage(tensor = dicom_tensor))
    list_subjects = []
    list_subjects.append(subjects)
    subject_dataset = tio.SubjectsDataset(list_subjects)
    data_loader = DataLoader(subject_dataset,
                                 batch_size=1,
                                 num_workers=4,
                                 pin_memory=True,
                                 shuffle=True, 
                                )
    data = next(iter(data_loader))
    return data

def resize_to_og(dicom_array, images):
    print(dicom_array.shape)
    print(images.shape)
    images = tio.ScalarImage(tensor = images.squeeze(0).cpu())
    resize_to_og = tio.Resize((dicom_array.shape[2], dicom_array.shape[1], dicom_array.shape[0]))
    images = resize_to_og(images)
    images_np = np.array(images)
    images_tensor = np.squeeze(images_np, axis = 0)
    images_tensor = np.moveaxis(images_tensor, (0,1,2), (2,1,0))
    print(images_tensor.shape)
    return images_tensor

def run_model(dicom_array, in_channels = 1, out_channels = 32, n_classes = 1):
    data = input_process(dicom_array)
    # print(data)
    # os.system('mkdir image')
    model = load_model(in_channels, out_channels, n_classes)
    device = 'cuda'
    with torch.no_grad():
        image = data['image'][tio.DATA].to(device)
        predict = model(image)
        image_og = resize_to_og(dicom_array, image)
        predict_og = resize_to_og(dicom_array, predict)
        predict_thresh = np.zeros(predict_og.shape)
        for i in range(predict_og.shape[2]):
            _, thresh = cv2.threshold(image_og[:,:,i], 0, 255, cv2.THRESH_BINARY)
            predict_og[:,:,i] = np.where(thresh, predict_og[:,:,i], 0)
            predict_thresh[:,:,i] = predict_og[:,:,i]
            predict_thresh[:,:,i][predict_thresh[:,:,i] > 0] = 255
            predict_thresh[:,:,i][predict_thresh[:,:,i] <= 0] = 0
        predict_thresh = np.array(predict_thresh, dtype = np.uint8)
        for i in range(predict_og.shape[2]):
            if predict_thresh[:,:,i].max() != 0:
              predict_thresh[:,:,i] = np.array(predict_thresh[:,:,i], dtype = np.uint32)
              output  = cv2.connectedComponentsWithStats(predict_thresh[:,:,i], 8, cv2.CV_32S)
              numLabels, labels, stats, centroids = output
              lis_cnt = []
              mask = np.zeros(predict_thresh[:,:,i].shape, np.uint8)
              count = 0
              for j in range(1, numLabels):
                  x = stats[j, cv2.CC_STAT_LEFT]
                  y = stats[j, cv2.CC_STAT_TOP]
                  w = stats[j, cv2.CC_STAT_WIDTH]
                  h = stats[j, cv2.CC_STAT_HEIGHT]
                  area = stats[j, cv2.CC_STAT_AREA]
                  if w*h > 100:
                    count = 1
                    componentMask = (labels == j).astype("uint8") * 255
                    componentMask = np.array(componentMask)
                    mask = cv2.bitwise_or(mask, componentMask)
              if count == 1:
                mask_new = cv2.bitwise_and(predict_thresh[:,:,i], mask)
                predict_thresh[:,:,i] = mask_new
        return predict_og, predict_thresh
