#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:16:23 2018

@author: doran
"""
import cv2
import glob
import io
import keras
import numpy               as np
import matplotlib.pyplot   as plt
import DNN.u_net           as unet
import DNN.params          as params
from random             import shuffle
from DNN.augmentation   import plot_img, randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip, fix_mask
from DNN.losses         import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss
from DNN.losses         import dice_coeff, MASK_VALUE, build_masked_loss, masked_accuracy, masked_dice_coeff
from keras.callbacks    import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers   import RMSprop
from PIL                import Image
from keras.preprocessing.image import img_to_array

samples = []
samples_hu = []

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

def train_process(data):
   img_f, mask_f = data
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], 5]) # for crypt, fufis + 3 mark types
   # Order clone channels: crypts, fufis, (KDM6A, MAOA, NONO, HDAC6, STAG2), p53, mPAS
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]
   if (mname[-5:]=="crypt"):
      mask[:,:,0] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
      dontmask = 0
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-100, 100),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-25, 25))
   elif (mname[-4:]=="fufi"):
      mask[:,:,1] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
      dontmask = 1
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-100, 100),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-25, 25))
   elif (mname[-5:]=="clone"):
      mname_broken = mask_f.split('/')[-1].split('_')
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
      if "p53" in mname_broken:
         mask[:,:,3] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 3
      if "mPAS" in mname_broken:
         mask[:,:,4] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 4
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-25, 25),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-15, 15))


   img, mask = randomShiftScaleRotate(img, mask,
                                    shift_limit=(-0.0625, 0.0625),
                                    scale_limit=(-0.1, 0.1),
                                    rotate_limit=(-20, 20))
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

def train_process_random():   
   img_f, mask_f = samples_hu[random.randint(0, len(samples_hu)-1)]
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], 5]) # for crypt, fufis + 3 mark types
   # Order clone channels: crypts, fufis, (KDM6A, MAOA, NONO, HDAC6, STAG2), p53, mPAS
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]
   if (mname[-5:]=="crypt"):
      mask[:,:,0] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
      dontmask = 0
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-100, 100),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-25, 25))
   elif (mname[-4:]=="fufi"):
      mask[:,:,1] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
      dontmask = 1
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-100, 100),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-25, 25))
   elif (mname[-5:]=="clone"):
      mname_broken = mask_f.split('/')[-1].split('_')
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
      if "p53" in mname_broken:
         mask[:,:,3] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 3
      if "mPAS" in mname_broken:
         mask[:,:,4] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 4
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-25, 25),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-15, 15))


   img, mask = randomShiftScaleRotate(img, mask,
                                    shift_limit=(-0.0625, 0.0625),
                                    scale_limit=(-0.1, 0.1),
                                    rotate_limit=(-20, 20))
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
               if key%5==0:
                  img, mask = train_process_random()
               else:
                  img, mask = train_process(ids)
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
   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/"

   # Redefine new network with new classification
   model = params.model_factory(input_shape=(params.input_size, params.input_size, 3), num_classes=5)
   model.load_weights(dnnfolder+"/weights/mousecrypt_weights.hdf5")

   # Set up training data   
   imgfolder = dnnfolder + "/input/mouse/train/"
   maskfolder = dnnfolder + "/input/mouse/train_masks/"
   crypts = glob.glob(imgfolder + "*_crypt.png")
   fufis = glob.glob(imgfolder + "*_fufi.png")
   clones = glob.glob(imgfolder + "*_clone.png")
   samples_cr = []
   for i in range(len(crypts)):
      mask = maskfolder+"mask"+crypts[i][(len(imgfolder)+3):]
      sample = (crypts[i], mask)
      samples_cr.append(sample)
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
   # add crypt samples
   samples += samples_cr
   samples += samples_cl
   samples += samples_fu

   # Get shrivelled-crypt human radiotherapy data    
   imgfolder_hu = dnnfolder + "/input/train/"
   maskfolder_hu = dnnfolder + "/input/train_masks/"
   rad_blocks = ["KM9M", "KM9S"]
   crypts_rad = []
   fufis_rad = []
   clones_rad = []
   for rb in rad_blocks:
      crypts_rad += glob.glob(imgfolder_hu + "*_" + rb + "*_crypt.png")
      fufis_rad += glob.glob(imgfolder_hu + "*_" + rb + "*_fufi.png")
      clones_rad += glob.glob(imgfolder_hu + "*_" + rb + "*_clone.png")
   samples_cr_rad = []
   for i in range(len(crypts_rad)):
      mask = maskfolder_hu+"mask"+crypts_rad[i][(len(imgfolder_hu)+3):]
      sample = (crypts_rad[i], mask)
      samples_cr_rad.append(sample)
   samples_fu_rad = []
   for i in range(len(fufis_rad)):
      mask = maskfolder_hu+"mask"+fufis_rad[i][(len(imgfolder_hu)+3):]
      sample = (fufis_rad[i], mask)
      samples_fu_rad.append(sample)
   samples_cl_rad = []
   for i in range(len(clones_rad)):
      mask = maskfolder_hu+"mask"+clones_rad[i][(len(imgfolder_hu)+3):]
      sample = (clones_rad[i], mask)
      samples_cl_rad.append(sample)
   # add rad samples to mouse data
   samples += samples_cr_rad
   samples += samples_cl_rad
   samples += samples_fu_rad
   
   ## WE WANT TO INCLUDE THE NEW MOUSE DATA, AND THEN EACH EPOCH
   ## TOP UP WITH THE SAME NUMBER OF HUMAN CRYPT DATA THAT IS 
   ## SAMPLED RANDOMLY FROM THE HUMAN DATA SET.
   # so now add human samples to global list samples_hu
   crypts_hu = glob.glob(imgfolder_hu + "*_crypt.png")
   fufis_hu = glob.glob(imgfolder_hu + "*_fufi.png")
   clones_hu = glob.glob(imgfolder_hu + "*_clone.png")
   samples_cr_hu = []
   for i in range(len(crypts_hu)):
      mask = maskfolder_hu+"mask"+crypts_hu[i][(len(imgfolder_hu)+3):]
      sample = (crypts_hu[i], mask)
      samples_cr_hu.append(sample)
   samples_fu_hu = []
   for i in range(len(fufis_hu)):
      mask = maskfolder_hu+"mask"+fufis_hu[i][(len(imgfolder_hu)+3):]
      sample = (fufis_hu[i], mask)
      samples_fu_hu.append(sample)
   samples_cl_hu = []
   for i in range(len(clones_hu)):
      mask = maskfolder_hu+"mask"+clones_hu[i][(len(imgfolder_hu)+3):]
      sample = (clones_hu[i], mask)
      samples_cl_hu.append(sample)
   # add crypt samples
   samples_hu += samples_cr_hu
   # add repeats of clone and fufi data to get desired ratios
   n1 = int(len(samples_cr_hu)/len(samples_cl_hu)/1.5)
   n2 = int(len(samples_cr_hu)/len(samples_fu_hu)/1.5)
   for i in range(n1): samples_hu += samples_cl_hu
   for i in range(n2): samples_hu += samples_fu_hu
      
   weights_name = dnnfolder+"/weights/mousecrypt_weights2.hdf5"
   logs_folder = dnnfolder+"/logs"
   
   callbacks = [ModelCheckpoint(monitor='loss', filepath=weights_name, save_best_only=True, 
                save_weights_only=True), TensorBoard(log_dir=logs_folder)]
                
   model.fit_generator(generator=train_generator(), steps_per_epoch=np.ceil(float(len(samples)) / float(batch_size)),
                       epochs=epochs, verbose=1, callbacks=callbacks, validation_data=None)

