#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:56:39 2021

@author: edward
"""
#!/usr/bin/env python3
import math
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from MiscFunctions import plot_img
from augmentation import random_affine, random_flip, random_perspective
from read_svs_class import svs_file_w_labels
from model_set_parameter_dicts import set_params_ed
params_gen_ed = set_params_ed()




## Dbueg read svs

files_test  = ['/home/edward/WIMM/Decryptics_train/train/KM1/KM1S_437769.svs',
  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11M_442932.svs',
  '/home/edward/WIMM/Decryptics_train/train/KM1/KM1M_428306.svs',
  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs',
  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs',
  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs',
  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs',
  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs']

# files_test  = ['/home/edward/WIMM/Decryptics_train/train/KM1/KM1S_437769.svs',
#  '/home/edward/WIMM/Decryptics_train/train/KM1/KM1S_437769.svs',
#  '/home/edward/WIMM/Decryptics_train/train/KM1/KM1S_437769.svs',
#  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs',
#  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs',
#  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs',
#  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs',
#  '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs']

already_curated = pd.read_csv("manual_curation_files/curated_files_summary.txt", names = ["file_name", "slide_crtd"])
already_curated = already_curated[already_curated['slide_crtd'] != "cancel"]
train_data      = already_curated.sample(150,random_state=22)
keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
val_data        = already_curated[keep_indx] #.sample(100, random_state=223)



mpp_try = [0.6, 1.1, 1.5, 1.7, 2.1, 3]

all_out1 = []
for mpp_i in mpp_try:
    sld_i = svs_file_w_labels(files_test[0], 1024, mpp_i)
    np.random.seed(seed=23) # (seed=22)
    img_cnt, mask = sld_i.fetch_clone()
    # img_cnt = sld_i.fetch_crypt()
    all_out1.append(img_cnt)
plot_img(tuple(all_out1), nrow = 2)


# mpp_try = [0.6, 1, 1.1, 1.7, 2.1, 3]

# # Speed test
# ii = 0
# sld_i = svs_file_w_labels(files_test[ii], 1024, mpp_i)
# print("Lowest mpp " + str(sld_i.mpp0))
# for mpp_i in mpp_try:
#     sld_i = svs_file_w_labels(files_test[ii], 1024, mpp_i)
#     print("## mpp " + str(mpp_i) + " downsample level used " + str(sld_i.dwnsmpl_lvl))
#     # img_cnt, mask = sld_i.fetch_clone(1)
#     %timeit img_cnt = sld_i.fetch_crypt(100)
    

# [0.6, 1, 1.1, 1.7, 2.1, 3]
# # '/home/edward/WIMM/Decryptics_train/train/KM11/KM11S_442912.svs'







