#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue May 11 14:43:54 2021

@author: edward
'''
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from training.augmentation import plot_img
from training.gen_v2 import DataGen_curt, CloneGen_curt
from tensorflow.keras import mixed_precision
from unet_sep_multiout import unet_sep
from model_set_parameter_dicts import set_params
params = set_params()

dnnfolder = '/home/doran/Work/py_code/new_DeCryptICS/'
logs_folder = dnnfolder + '/training/logs'

mixed_precision.set_global_policy('mixed_float16')

# Run paramaters
epochs               = 200
params['umpp']       = 1.1
params['num_bbox']   = 400
params['batch_size'] = 8
params['cpfr_frac']  = [2,1,1,1]
nsteps        = 2
nclone_factor = 4
npartial_mult = 5
weight_ccpf = [1., 2.2, 2.2, 1.25, 1.25] # unet, clone, partial, fufi, crypt
params['crypt_class']  = False

# Read curated data and filter bad ones
already_curated = pd.read_csv('./training/manual_curation_files/curated_files_summary.txt', 
                              names = ['file_name', 'slide_crtd'])
already_curated = already_curated[already_curated['slide_crtd'] != 'cancel']

# remove poorly stained slides or radiotherapy patients
bad_slides = pd.read_csv('./training/manual_curation_files/slidequality_scoring.csv')
radiother = np.where(bad_slides['quality_label']==2)[0]
staining = np.where(bad_slides['quality_label']==0)[0]
dontuse = np.asarray(bad_slides['path'])[np.hstack([radiother, staining])]
dontuse = pd.DataFrame({'file_name':list(dontuse)}).drop_duplicates(keep='first')
inds = ~already_curated.file_name.isin(dontuse.file_name)
already_curated = already_curated[inds]

#train_data      = already_curated.sample(150, random_state=22)
#keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
#val_data        = already_curated[keep_indx].sample(100, random_state=223)

train_data      = already_curated.sample(already_curated.shape[0]-100, random_state=22)
keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
val_data        = already_curated[keep_indx]

train_datagen = DataGen_curt(params, train_data, nsteps, nclone_factor, npartial_mult)
val_datagen   = CloneGen_curt(params, val_data, fufis=True)

model, just_trnsf, just_unet = unet_sep(params, weight_ccpf = weight_ccpf)

print('Loading weights!!')
weights_name = dnnfolder + '/weights/pixshuf_genbb_balanceweights_11mpp.hdf5'
model.load_weights(weights_name)
weights_name_next = dnnfolder + '/weights/pixshuf_genbb_jitter_fufidilate_11mpp.hdf5'

callbacks = [EarlyStopping(monitor='loss', patience=25, verbose=1, min_delta=1e-9),
             ReduceLROnPlateau(monitor='loss', factor=0.075, patience=8, verbose=1, min_delta=1e-9),
             ModelCheckpoint(monitor='val_loss', mode='min', filepath=weights_name_next, 
                             save_best_only=True, save_weights_only=True, verbose=1),
             CSVLogger(logs_folder + '/hist_'+weights_name_next.split('/')[-1].split('.')[0]+'.csv')]
         
res = model.fit(
         x = train_datagen,
         validation_data = val_datagen,
         verbose = 1,
         epochs = epochs,
         callbacks = callbacks,
         workers = 2,
         )


# CLR parameters
#if use_CLR:
#   max_lr = 0.0005
#   base_lr = max_lr/10
#   max_m = 0.98
#   base_m = 0.85
#   cyclical_momentum = True
#   cycles = 2.35
#   iterations = round(len(train_datagen)/params['batch_size']*epochs)
#   iterations = list(range(0,iterations+1))
#   step_size = len(iterations)/(cycles)
#   clr =  CyclicLR(base_lr=base_lr,
#                max_lr=max_lr,
#                step_size=step_size,
#                max_m=max_m,
#                base_m=base_m,
#                cyclical_momentum=cyclical_momentum)    
#   callbacks = [clr,
#                ModelCheckpoint(monitor='val_loss', mode='min', filepath=weights_name_next, 
#                                save_best_only=True, save_weights_only=True, verbose=1),
#                CSVLogger(logs_folder + '/hist_'+weights_name_next.split('/')[-1].split('.')[0]+'.csv')]
#else: 
