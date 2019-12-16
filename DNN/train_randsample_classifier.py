#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:16:23 2018

@author: doran
"""
import cv2
import glob
import io
import random
import keras
import numpy               as np
import matplotlib.pyplot   as plt
import DNN.u_net           as unet
import DNN.params          as params
from DNN.augmentation   import plot_img, randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip, fix_mask
from DNN.losses         import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss
from DNN.losses         import dice_coeff, MASK_VALUE, build_masked_loss, masked_accuracy, masked_dice_coeff
from keras.callbacks    import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers   import RMSprop
from PIL                import Image
from keras.preprocessing.image import img_to_array

samples = []
samples_cr = []

if keras.backend._BACKEND=="tensorflow":
   import tensorflow as tf
   num_cores = 16
   GPU = True
   CPU = False

   if GPU:
       num_GPU = 1
       num_CPU = 1
   if CPU:
       num_CPU = 1
       num_GPU = 0
       import os
       os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

   config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
           inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
           device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
   session = tf.Session(config=config)
   keras.backend.set_session(session)

if keras.backend._BACKEND=="mxnet":
   import mxnet
   input_shape = (3, params.input_size_train, params.input_size_train)
   chan_num = 1
else:
   input_shape = (params.input_size_train, params.input_size_train, 3)
   chan_num = 3
   
input_size = params.input_size_train
SIZE = (input_size, input_size)
epochs = params.max_epochs
batch_size = params.batch_size

def train_process_events(data):
   img_f, mask_f = data
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], 5]) # for crypt, fufis + 3 mark types
   # Order clone channels: crypts, fufis, (KDM6A, MAOA, NONO, HDAC6, STAG2), p53, mPAS
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]

   if (mname[-4:]=="fufi"):
      mask[:,:,1] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
      dontmask = 1
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-100, 100),
                                     sat_shift_limit=(-25, 25),
                                     val_shift_limit=(-25, 25))
   elif (mname[-5:]=="clone"):
      mname_broken = mask_f.upper().split('/')[-1].split('_')
      if "KDM6A" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "MAOA" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "NONO" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "HDAC6" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "STAG2" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "P53" in mname_broken:
         mask[:,:,3] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 3
      if "MPAS" in mname_broken:
         mask[:,:,4] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 4
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-5, 5),
                                     sat_shift_limit=(-15, 15),
                                     val_shift_limit=(-15, 15))


   img, mask = randomShiftScaleRotate(img, mask,
                                    shift_limit=(-0.0125, 0.0125),
                                    scale_limit=(-0.1, 0.1))
   img, mask = randomHorizontalFlip(img, mask)
   fix_mask(mask)
   
   ## Need to make masking values on outputs in float32 space, as uint8 arrays can't deal with it
   img = img.astype(np.float32) / 255
   mask = mask.astype(np.float32) / 255
   # choose which channel to mask (i.e. all other channels are masked)
   for i in range(mask.shape[2]):
      if (not i==dontmask):
         mask[:,:,i].fill(MASK_VALUE) 
   return (img, mask)

def train_process_random_crypts():   
   img_f, mask_f = samples_cr[random.randint(0, len(samples_cr)-1)]
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], 5]) # for crypt, fufis + 3 mark types
   # Order clone channels: crypts, fufis, (KDM6A, MAOA, NONO, HDAC6, STAG2), p53, mPAS
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]
   
   mask[:,:,0] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
   dontmask = 0
   img = randomHueSaturationValue(img,
                                  hue_shift_limit=(-100, 100),
                                  sat_shift_limit=(-25, 25),
                                  val_shift_limit=(-25, 25))

   img, mask = randomShiftScaleRotate(img, mask,
                                    shift_limit=(-0.0125, 0.0125),
                                    scale_limit=(-0.1, 0.1))
   img, mask = randomHorizontalFlip(img, mask)
   fix_mask(mask)
   
   ## Need to make masking values on outputs in float32 space, as uint8 arrays can't deal with it
   img = img.astype(np.float32) / 255
   mask = mask.astype(np.float32) / 255
   # choose which channel to mask (i.e. all other channels are masked)
   for i in range(mask.shape[2]):
      if (not i==dontmask):
         mask[:,:,i].fill(MASK_VALUE) 
   return (img, mask)

def train_generator():
    while True:
        for start in range(0, len(samples), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(samples))
            ids_train_batch = samples[start:end]
            key = 0
            for ids in ids_train_batch:
               if key%3==0: # set the ratio of crypts to events
                  img, mask = train_process_events(ids)                  
               else:
                  img, mask = train_process_random_crypts()
               x_batch.append(img)
               y_batch.append(mask)
               key += 1
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            if keras.backend._BACKEND=="mxnet":
               x_batch = keras.utils.to_channels_first(x_batch)
               y_batch = keras.utils.to_channels_first(y_batch)
            yield x_batch, y_batch

if __name__=="__main__":
   base_folder = "/home/doran/Work/py_code/DeCryptICS/DNN/" # as training data is in DeCryptICS folder
   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/"
   
   # Define network
   model = params.model_factory(input_shape=input_shape, num_classes=5, chan_num=chan_num)
   
   # Set up training data   
   training_base_folder = "/home/doran/Work/py_code/DeCryptICS/DNN/"
   imgfolder = training_base_folder + "/input/train/"
   maskfolder = training_base_folder + "/input/train_masks/"
   crypts = glob.glob(imgfolder + "*_crypt.png")
   fufis = glob.glob(imgfolder + "*_fufi.png")
   clones = glob.glob(imgfolder + "*_clone.png")

   # get crypt samples from which to sample from
   for i in range(len(crypts)):
      mask = maskfolder+"mask"+crypts[i][(len(imgfolder)+3):]
      sample = (crypts[i], mask)
      samples_cr.append(sample)
   # get event samples
   samples_fu = []
   for i in range(len(fufis)):
      mask = maskfolder+"mask"+fufis[i][(len(imgfolder)+3):]
      sample = (fufis[i], mask)
      samples_fu.append(sample)
   samples_cl = []
   for i in range(len(clones)):
      mask = maskfolder+"mask"+clones[i][(len(imgfolder)+3):]
      sample = (clones[i], mask)
      samples_cl.append(sample)

   # load mouse samples?
#   imgfolder_m = training_base_folder + "/input/mouse/train/"
#   maskfolder_m = training_base_folder + "/input/mouse/train_masks/"
#   crypts_m = glob.glob(imgfolder_m + "*_crypt.png")
#   fufis_m = glob.glob(imgfolder_m + "*_fufi.png")
#   clones_m = glob.glob(imgfolder_m + "*_clone.png")   
#   for i in range(len(crypts_m)):
#      mask = maskfolder_m+"mask"+crypts_m[i][(len(imgfolder_m)+3):]
#      sample = (crypts_m[i], mask)
#      samples_cr.append(sample)
#   for i in range(len(fufis_m)):
#      mask = maskfolder+"mask"+fufis_m[i][(len(imgfolder_m)+3):]
#      sample = (fufis_m[i], mask)
#      samples_fu.append(sample)
#   for i in range(len(clones_m)):
#      mask = maskfolder_m+"mask"+clones_m[i][(len(imgfolder_m)+3):]
#      sample = (clones_m[i], mask)
#      samples_cl.append(sample)
            
   samples += samples_cl
   samples += samples_fu
   random.shuffle(samples)
   
   curr_weights = "/weights/cryptfuficlone_weights.hdf5"
   weights_name = dnnfolder+"/weights/cryptfuficlone_weights2.hdf5"
   model.load_weights(dnnfolder+curr_weights)
   logs_folder = dnnfolder+"/logs"
   
   callbacks = [EarlyStopping(monitor='loss', patience=500, verbose=1, min_delta=1e-11),
                ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50, verbose=1, min_delta=1e-9),
                ModelCheckpoint(monitor='loss', filepath=weights_name, save_best_only=True, save_weights_only=True),
                TensorBoard(log_dir=logs_folder)]
                #TensorBoardImage(log_dir=logs_folder, tags=test_tags, test_image_batches=test_batches)]
                
   model.fit_generator(generator=train_generator(), steps_per_epoch=np.ceil(float(len(samples)) / float(batch_size)), 
                       epochs=epochs, verbose=1, callbacks=callbacks, validation_data=None)

