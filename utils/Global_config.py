import os
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
class GlobalConfig():
    # root = '../input/brats20-dataset-training-validation'
    # root_system = str(os.getcwd())
    # train_path = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
    # val_path = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'
    # name_mapping_train = 'name_mapping.csv'
    # survival_info_train = 'survival_info.csv'
    # name_mapping_test = 'name_mapping_validation_data.csv'
    # survival_info_test = 'survival_evaluation.csv'
    # checkpoint = 'best_checkpoint/'
    # best_checkpoint = 'best_checkpoint.pth'
    # highest_checkpoint = 'highest_checkpoint.pth'
    # train_df = 'train_df.csv'
    # val_df = 'test_df.csv'
    # pretrained_model_path = '../input/test-check/last_checkpoint.pth'
    seed = 55