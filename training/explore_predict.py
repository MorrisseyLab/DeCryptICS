#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 08:54:17 2021

@author: edward
"""

import cv2
import numpy      as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from training.augmentation import plot_img
from training.gen_ed import DataGen_curt, CloneGen_curt
from tensorflow.keras import mixed_precision
from unet_sep_multiout import unet_sep

from model_set_parameter_dicts import set_params
params = set_params()

dnnfolder = "/home/edward/WIMM/Decryptics_train/decryptics_code/DNN/fullres_unet/"
logs_folder = dnnfolder + "/logs"

mixed_precision.set_global_policy('mixed_float16')

# Run paramaters
max_epochs                  = 200
params_gen_ed['umpp']       = 1.5 # 7 # 0.6  0.8 #1.7 #1.7  #1.1 # probably faster if larger than 1.01
params_gen_ed['num_bbox']   = 400 # 400
params_gen_ed['batch_size'] = 7
params_gen_ed["just_clone"] = True

nsteps        = 2
nclone_mult   = 4

# Read curated data and filter bad ones
already_curated = pd.read_csv("manual_curation_files/curated_files_summary.txt", 
                              names = ["file_name", "slide_crtd"])
already_curated = already_curated[already_curated['slide_crtd'] != "cancel"]
train_data      = already_curated.sample(150,random_state=22)
keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
val_data        = already_curated[keep_indx] #.sample(100, random_state=223)

train_datagen = DataGen_curt(params_gen_ed, train_data, nsteps, nclone_mult)
# val_datagen   = CloneGen_curt(params_gen_ed, val_data)    
 
model, just_trnsf, just_unet = unet_sep(params_gen_ed, weight_clone = 100)
weights_name = dnnfolder + "/weights/slmrnet_transf_400bb_15mpp_bnpos_clonedilate.hdf5"
model.load_weights(weights_name)


#TP, FP, FN, TN, num_uncertain
def get_confusion_matrix_values(y_true, y_pred):
    num_uncertain = np.sum(np.bitwise_and(y_pred > 0.2, y_pred < 0.7))
    y_true = y_true < 0.5
    y_pred = y_pred < 0.5
    cm = confusion_matrix(y_true, y_pred)
    
    if y_true.shape[0] == 0:
        return 0, 0, 0, 0, 0
    # Avoid crash if only one class
    if np.mean(y_true) == 1:
        return 0, 0, 0, len(y_true), num_uncertain
    if np.mean(y_true) == 0:
        return len(y_true), 0, 0, 0, num_uncertain

    return cm[0][0], cm[0][1], cm[1][0], cm[1][1], num_uncertain

norm_mean = np.array([0.485, 0.456, 0.406])
norm_std = np.array([0.229, 0.224, 0.225])
       
batch_pred = []
file_saves = []
svs_files  = []
crypts_all  = []
pred_crypts_all  = []
for batch_num in range(len(train_datagen)):
    (im, b_bboxes), (b_ma, b_p_i) = train_datagen[batch_num]
    files_used = train_datagen.pathused
    pred_i = model.predict([im, b_bboxes])
    
    svs_files += (files_used)    

    max_crypts = b_bboxes.shape[1]
    batch_size = b_bboxes.shape[0]
    
    
    for i in range(batch_size):
        ba         = i
        bboxes     = b_bboxes[ba]
        # crypt index, always same bounding boxes
        indx_crypt = np.where(bboxes[:,2] > 0)[0]
        
        true_lab = b_p_i[i]
        pred_lab = pred_i[1][i,:,0]
        metrics_out = get_confusion_matrix_values(true_lab[indx_crypt], pred_lab[indx_crypt])
        batch_pred.append(metrics_out)
        
        # num crypts
        crypts = (255*( cv2.pyrUp(b_ma[ba, :,:]) > 0.5)).astype(np.uint8)
        contours, hierarchy = cv2.findContours(crypts, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        crypts_all.append(len(contours))

        # predicted crypts
        crypts = (255*( cv2.pyrUp(pred_i[0][ba][:,:, 0]) > 0.5)).astype(np.uint8)
        contours, hierarchy = cv2.findContours(crypts, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        pred_crypts_all.append(len(contours))

        if (metrics_out[1] + metrics_out[2])>0 or metrics_out[4] > 2:
            ## Plot predictions
            img        = (255*(im[ba]*norm_std + norm_mean)).astype(np.uint8)
            img_cnt    = img.copy()
            bboxes     = b_bboxes[ba]
            p_i        = b_p_i[ba]
            m_p_i      = np.round(pred_i[1][ba], 2)
            for j in indx_crypt:                
                bbx_j    = 1024*bboxes[j, :]
                pi_j     = p_i[j]
                pred_p_i = m_p_i[j][0]
                y = int(bbx_j[0]); x =  int(bbx_j[1]); y2 = int(bbx_j[2]); x2 =  int(bbx_j[3])
                # Rectangle   
                col = [0,255,0]
                if pi_j>0.5: col[0] += 255
                if pred_p_i>0.5: col[2] += 255
                cv2.rectangle(img,(x,y),(x2,y2),tuple(col),2)
                col = (0,255,0) if pred_p_i else (255,0,0) 
                if pred_p_i > 1e-2:
                    cv2.putText(img, str(pred_p_i),(x2+10,y2),0,0.3, col)
            
            # plot_img(img, hold_plot=True, nameWindow = "full")
            
            crypt_fufi_pred = pred_i[0][ba] 
            
            crypts = (255*( cv2.pyrUp(crypt_fufi_pred[:,:]) > 0.5)).astype(np.uint8)
            contours, hierarchy = cv2.findContours(crypts, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            img_cnt = cv2.drawContours(img_cnt, contours, -1, (255,0,0), 2)
            file_name = "predict_plots/img" + str([batch_num, i])
            cv2.imwrite(file_name + "_clone.png", img)
            cv2.imwrite(file_name + "_crypts.png", img_cnt)
            file_saves.append(file_name)
        else: 
            file_saves.append("None")
    # plot_img(img_cnt, hold_plot=True, nameWindow = "full")
    
    # if batch_num == 3: break

df_out = pd.DataFrame(batch_pred, columns=["TP", "FP", "FN", "TN", "num_uncertain"])

df_out["file_saves"] = file_saves
df_out["svs_files"] = svs_files
df_out["crypts_all"] = crypts_all
df_out["pred_crypts_all"] = pred_crypts_all
df_out.to_csv("summary_analysis.csv")


