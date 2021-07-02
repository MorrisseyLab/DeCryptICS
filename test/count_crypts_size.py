#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 08:12:23 2021

@author: edward
"""
import cv2
import numpy      as np
import pandas as pd
from augmentation import plot_img
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from gen_ed      import DataGen_curt, CloneGen_curt
from tensorflow.keras import mixed_precision
from unet_sep_multiout import unet_sep

from model_set_parameter_dicts import set_params_ed
params_gen_ed = set_params_ed()

dnnfolder = "/home/edward/WIMM/Decryptics_train/decryptics_code/DNN/fullres_unet/"
logs_folder = dnnfolder + "/logs"

mixed_precision.set_global_policy('mixed_float16')

params_gen_ed['umpp']       = 1.1 #7 #0.8 #1.7 #1.8, 1.1 # probably faster if larger than 1.01
params_gen_ed['num_bbox']   = 200
params_gen_ed['batch_size'] = 8
params_gen_ed["just_clone"] = True
params_gen_ed["shuffle"] = False
params_gen_ed["aug"] = False
params_gen_ed["normalize"] = False

params_val = params_gen_ed.copy()

nsteps        = 2
nclone_mult   = 4

# Read curated data and filter bad ones
already_curated = pd.read_csv("manual_curation_files/curated_files_summary.txt", names = ["file_name", "slide_crtd"])
already_curated = already_curated[already_curated['slide_crtd'] != "cancel"]
train_data      = already_curated.sample(150,random_state=22)
keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
val_data        = already_curated[keep_indx] #.sample(100, random_state=223)


train_datagen = DataGen_curt(params_gen_ed, train_data, nsteps, nclone_mult)


all_crypt = []
for btch_i in range(20):
    kk = train_datagen[btch_i]
    for sub_j in range(8):
        msks_i = kk[1][0]
        crypt_mask  = msks_i[sub_j].astype(np.uint8)
        contours, hierarchy = cv2.findContours(crypt_mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        all_crypt.append(len(contours))
print(np.percentile(all_crypt, [50, 90, 95, 99]))

# np.percentile(all_crypt, [50, 90, 95, 99])
# 1.7 mpp  [172.5  301.1  399.95 541.37]
# 1.5 mpp  [130.5  276.1  336.05 440.62]
# 1.1mpp   [ 89.   157.3  181.45 258.22]

# def plot_box_prob(img_in, bbox, probs, norm = False):
#     # img        = (255*(img_in*norm_std + norm_mean)).astype(np.uint8)
    
#     img    = (255*img_in.copy()).astype(np.uint8)
        
#     # for j in range(max_crypts):
#     for bbx_j, pi_j  in zip(bbox, probs):
#         if bbx_j[2] <0:
#             continue
#         bbx_j = img.shape[0]*bbx_j
#         y = int(bbx_j[0]); x =  int(bbx_j[1]); y2 = int(bbx_j[2]); x2 =  int(bbx_j[3])
#         # Rectangle   
#         col = [0,255,0]
#         if pi_j>0.5: col[0] += 255
#         cv2.rectangle(img,(x,y),(x2,y2),tuple(col),2)
#     return img


# def plot_batch(batch_i):
#     dt1_img = batch_i[0][0]
#     dt1_bbox = batch_i[0][1]
#     # dt1_mask = batch_i[1][0]
#     dt1_p    = batch_i[1][1]
#     img_out = []
#     for ii in range(dt1_img.shape[0]):
#         img_i = plot_box_prob(dt1_img[ii], dt1_bbox[ii], dt1_p[ii])
#         img_out.append(img_i)
#     return np.array(img_out)



# all_out1 = []
# for mpp_i in [0.6, 1, 1.1, 1.7, 2.1, 3]:
#     np.random.seed(seed=44) # (seed=22)
#     params_gen_ed['umpp']         =  mpp_i
#     params_gen_ed['dilate_masks'] =  True
#     print("## mpp" + str(mpp_i))
#     #CloneGen_curt DataGen_curt(params_gen_ed, train_data, nsteps, nclone_mult)  
#     # train_datagen = CloneGen_curt (params_gen_ed, train_data)  
#     train_datagen = DataGen_curt(params_gen_ed, train_data, nsteps, nclone_mult)  
#     dt1_img       = plot_batch(train_datagen[0])    
#     all_out1.append(dt1_img)

# jj = 1 # 4
# img_tuple = [im_i[jj] for im_i in  all_out1]
# plot_img(tuple(img_tuple), nrow = 2)


# for mpp_i in [0.6, 1, 1.1, 1.7, 2.1, 3]:
#     np.random.seed(seed=22)
#     params_gen_ed['umpp']       = mpp_i
#     print("## mpp" + str(mpp_i))
#     train_datagen = DataGen_curt(params_gen_ed, train_data, nsteps, nclone_mult)
#     %timeit uu = train_datagen[0][0][0][1]


# %timeit for btch_i in range(4): kk = train_datagen[btch_i]
# 9.09 s ± 264 ms # mpp 0.6
# 10.1 s ± 264 ms # mpp 0.8
# 12   s ± 503 ms # mpp 1.1
# 14.9 s ± 573 ms # mpp 1.7 
# Must go from 0.5 to 2




