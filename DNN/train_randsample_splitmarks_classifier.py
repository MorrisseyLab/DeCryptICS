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

samples_cl_t = []
samples_cl_f = []
samples_cr = []
samples_fu = []

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
num_classes = 6
num_rand_crypts = 4
num_rand_fufis = 2
num_false_clones = 4
batch_size = params.batch_size - (num_rand_crypts + num_rand_fufis + num_false_clones)
# [crypts, fufis, (KDM6A, STAG2, NONO), (MAOA, HDAC6), p53, mPAS]

def train_process_events(data):
   img_f, mask_f = data
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], num_classes]) # for crypt, fufis + 4 mark types
   # Order clone channels: crypts, fufis, (KDM6A, STAG2, NONO), (MAOA, HDAC6), p53, mPAS
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]

   if (mname[-4:]=="fufi"):
      dontmask = 1
      mask[:,:,dontmask] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)      
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-100, 100),
                                     sat_shift_limit=(-25, 25),
                                     val_shift_limit=(-25, 25))
   elif (mname[-5:]=="clone"):
      mname_broken = mask_f.upper().split('/')[-1].split('_')
      dontmask = np.nan
      if "KDM6A" in mname_broken:
         dontmask = 2 
      if "STAG2" in mname_broken:
         dontmask = 2
      if "NONO" in mname_broken:
         dontmask = 2
      if "MAOA" in mname_broken:
         dontmask = 3
      if "HDAC6" in mname_broken:
         dontmask = 3
      if "P53" in mname_broken:
         dontmask = 4
      if "MPAS" in mname_broken:
         dontmask = 5
      mask[:,:,dontmask] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)   
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-5, 5),
                                     sat_shift_limit=(-10, 10),
                                     val_shift_limit=(-10, 10))

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

def train_process_random_fufi():
   img_f, mask_f = samples_fu[random.randint(0, len(samples_fu)-1)]
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], num_classes])
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]

   if (mname[-4:]=="fufi"):
      dontmask = 1
      mask[:,:,dontmask] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)      
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-30, 30),
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

def train_process_random_crypt():   
   img_f, mask_f = samples_cr[random.randint(0, len(samples_cr)-1)]
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], num_classes]) # for crypt, fufis + 3 mark types
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

def train_process_random_falseclone():
   img_f, mask_f = samples_cl_f[random.randint(0, len(samples_cl_f)-1)]
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], num_classes]) # for crypt, fufis + 4 mark types
   # Order clone channels: crypts, fufis, (KDM6A, STAG2, NONO), (MAOA, HDAC6), p53, mPAS
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]

   mname_broken = mask_f.upper().split('/')[-1].split('_')
   dontmask = np.nan
   if "KDM6A" in mname_broken:
      dontmask = 2 
   if "STAG2" in mname_broken:
      dontmask = 2
   if "NONO" in mname_broken:
      dontmask = 2
   if "MAOA" in mname_broken:
      dontmask = 3
   if "HDAC6" in mname_broken:
      dontmask = 3
   if "P53" in mname_broken:
      dontmask = 4
   if "MPAS" in mname_broken:
      dontmask = 5
   mask[:,:,dontmask] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)   
   img = randomHueSaturationValue(img,
                                  hue_shift_limit=(-5, 5),
                                  sat_shift_limit=(-10, 10),
                                  val_shift_limit=(-10, 10))

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
        for start in range(0, len(samples_cl_t), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(samples_cl_t))
            ids_train_batch = samples_cl_t[start:end]
            for ids in ids_train_batch:
               img, mask = train_process_events(ids)
               x_batch.append(img)
               y_batch.append(mask)
            for ii in range(num_false_clones):
               img, mask = train_process_random_falseclone()
               x_batch.append(img)
               y_batch.append(mask)
            for ii in range(num_rand_fufis):
               img, mask = train_process_random_fufi()
               x_batch.append(img)
               y_batch.append(mask)
            for ii in range(num_rand_crypts):
               img, mask = train_process_random_crypt()
               x_batch.append(img)
               y_batch.append(mask)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            if keras.backend._BACKEND=="mxnet":
               x_batch = keras.utils.to_channels_first(x_batch)
               y_batch = keras.utils.to_channels_first(y_batch)
            yield x_batch, y_batch

if __name__=="__main__":
   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/"
   
   # Define network
   model_split = params.model_factory(input_shape=input_shape, num_classes=num_classes, chan_num=chan_num)
   
   # Set up training data   
   training_base_folder3 = "/home/doran/Work/py_code/DeCryptICS/DNN/"
   imgfolder3 = training_base_folder3 + "/input/train/"
   maskfolder3 = training_base_folder3 + "/input/train_masks/"
   crypts = glob.glob(imgfolder3 + "*_crypt.png")
   fufis = glob.glob(imgfolder3 + "*_fufi.png")
#   clones = glob.glob(imgfolder + "*_clone.png")

   # get crypt samples from which to sample from
   for i in range(len(crypts)):
      mask = maskfolder3+"mask"+crypts[i][(len(imgfolder3)+3):]
      sample = (crypts[i], mask)
      samples_cr.append(sample)
   # get event samples
   for i in range(len(fufis)):
      mask = maskfolder3+"mask"+fufis[i][(len(imgfolder3)+3):]
      sample = (fufis[i], mask)
      samples_fu.append(sample)
   
   ##########
   ## New training set
   # abstract this out...
   training_base_folder = dnnfolder + '/input/new_train_set/'
   imgfolder = training_base_folder + "/img/"
   maskfolder = training_base_folder + "/mask/"
   imgfolder2 = dnnfolder + "/input/train/"
   maskfolder2 = dnnfolder + "/input/train_masks/"
   #####
   # find true clones
   t_clones_s = glob.glob(imgfolder + '/set_*/slide_*/img*STAG2*T_clone.png')
   t_clones_k = glob.glob(imgfolder + '/set_*/slide_*/img*KDM6A*T_clone.png')
   t_clones_m = glob.glob(imgfolder + '/set_*/slide_*/img*MAOA*T_clone.png')
   t_clones_n = glob.glob(imgfolder + '/set_*/slide_*/img*NONO*T_clone.png')
   t_clones_hd = glob.glob(imgfolder + '/set_*/slide_*/img*HDAC6*T_clone.png')
   t_clones_mp = glob.glob(imgfolder + '/set_*/slide_*/img*mPAS*T_clone.png')
   t_clones_p = glob.glob(imgfolder + '/set_*/slide_*/img*P53*T_clone.png')
   t_clones_s2 = glob.glob(imgfolder2 + '/img*STAG2*T_clone.png')
   t_clones_k2 = glob.glob(imgfolder2 + '/img*KDM6A*T_clone.png')
   t_clones_m2 = glob.glob(imgfolder2 + '/img*MAOA*T_clone.png')
   t_clones_n2 = glob.glob(imgfolder2 + '/img*NONO*T_clone.png')
   t_clones_hd2 = glob.glob(imgfolder2 + '/img*HDAC6*T_clone.png')
   t_clones_mp2 = glob.glob(imgfolder2 + '/img*mPAS*T_clone.png')
   t_clones_p2 = glob.glob(imgfolder2 + '/img*P53*T_clone.png') # not enough T here
   # combine marks into channels
   t_clones_set1_ch1 = t_clones_s + t_clones_k + t_clones_n
   t_clones_set1_ch2 = t_clones_m + t_clones_hd
   t_clones_set1_ch3 = t_clones_p
   t_clones_set1_ch4 = t_clones_mp
   t_clones_set2_ch1 = t_clones_s2 + t_clones_k2 + t_clones_n2
   t_clones_set2_ch2 = t_clones_m2 + t_clones_hd2
   t_clones_set2_ch3 = t_clones_p2
   t_clones_set2_ch4 = t_clones_mp2
   # find corresponding masks
   mt_clones_set1_ch1 = [s.replace('img', 'mask') for s in t_clones_set1_ch1]
   mt_clones_set1_ch2 = [s.replace('img', 'mask') for s in t_clones_set1_ch2]
   mt_clones_set1_ch3 = [s.replace('img', 'mask') for s in t_clones_set1_ch3]
   mt_clones_set1_ch4 = [s.replace('img', 'mask') for s in t_clones_set1_ch4]
   mt_clones_set2_ch1 = [maskfolder2+"mask"+s[(len(imgfolder2)+3):] for s in t_clones_set2_ch1]
   mt_clones_set2_ch2 = [maskfolder2+"mask"+s[(len(imgfolder2)+3):] for s in t_clones_set2_ch2]
   mt_clones_set2_ch3 = [maskfolder2+"mask"+s[(len(imgfolder2)+3):] for s in t_clones_set2_ch3]
   mt_clones_set2_ch4 = [maskfolder2+"mask"+s[(len(imgfolder2)+3):] for s in t_clones_set2_ch4]
   # combine sets
   t_clones_ch1  = t_clones_set1_ch1  + t_clones_set2_ch1
   t_clones_ch2  = t_clones_set1_ch2  + t_clones_set2_ch2
   t_clones_ch3  = t_clones_set1_ch3  + t_clones_set2_ch3
   t_clones_ch4  = t_clones_set1_ch4  + t_clones_set2_ch4
   mt_clones_ch1 = mt_clones_set1_ch1 + mt_clones_set2_ch1
   mt_clones_ch2 = mt_clones_set1_ch2 + mt_clones_set2_ch2
   mt_clones_ch3 = mt_clones_set1_ch3 + mt_clones_set2_ch3
   mt_clones_ch4 = mt_clones_set1_ch4 + mt_clones_set2_ch4
   # make positive clone samples sets
   tsamps_clones_ch1 = [(im, msk) for im, msk in zip(t_clones_ch1, mt_clones_ch1)]
   tsamps_clones_ch2 = [(im, msk) for im, msk in zip(t_clones_ch2, mt_clones_ch2)]
   tsamps_clones_ch3 = [(im, msk) for im, msk in zip(t_clones_ch3, mt_clones_ch3)]
   tsamps_clones_ch4 = [(im, msk) for im, msk in zip(t_clones_ch4, mt_clones_ch4)]
   
   #####
   # find false clones to sample from
   f_clones_s = glob.glob(imgfolder + '/set_*/slide_*/img*STAG2*F_clone.png')
   f_clones_k = glob.glob(imgfolder + '/set_*/slide_*/img*KDM6A*F_clone.png')
   f_clones_m = glob.glob(imgfolder + '/set_*/slide_*/img*MAOA*F_clone.png')
   f_clones_n = glob.glob(imgfolder + '/set_*/slide_*/img*NONO*F_clone.png')
   f_clones_hd = glob.glob(imgfolder + '/set_*/slide_*/img*HDAC6*F_clone.png')
   f_clones_mp = glob.glob(imgfolder + '/set_*/slide_*/img*mPAS*F_clone.png')
   f_clones_p = glob.glob(imgfolder + '/set_*/slide_*/img*P53*F_clone.png')
   f_clones_s2 = glob.glob(imgfolder2 + '/img*STAG2*_clone.png')
   f_clones_k2 = glob.glob(imgfolder2 + '/img*KDM6A*_clone.png')
   f_clones_m2 = glob.glob(imgfolder2 + '/img*MAOA*_clone.png')
   f_clones_n2 = glob.glob(imgfolder2 + '/img*NONO*_clone.png')
   f_clones_hd2 = glob.glob(imgfolder2 + '/img*HDAC6*_clone.png')
   f_clones_mp2 = glob.glob(imgfolder2 + '/img*mPAS*_clone.png')
   f_clones_p2 = glob.glob(imgfolder2 + '/img*P53*_clone.png') # included above
   # combine marks into channels
   f_clones_set1_ch1 = f_clones_s + f_clones_k + f_clones_n
   f_clones_set1_ch2 = f_clones_m + f_clones_hd
   f_clones_set1_ch3 = f_clones_p
   f_clones_set1_ch4 = f_clones_mp
   f_clones_set2_ch1 = f_clones_s2 + f_clones_k2 + f_clones_n2
   f_clones_set2_ch2 = f_clones_m2 + f_clones_hd2
   f_clones_set2_ch3 = f_clones_p2
   f_clones_set2_ch4 = f_clones_mp2
   # find corresponding masks
   mf_clones_set1_ch1 = [s.replace('img', 'mask') for s in f_clones_set1_ch1]
   mf_clones_set1_ch2 = [s.replace('img', 'mask') for s in f_clones_set1_ch2]
   mf_clones_set1_ch3 = [s.replace('img', 'mask') for s in f_clones_set1_ch3]
   mf_clones_set1_ch4 = [s.replace('img', 'mask') for s in f_clones_set1_ch4]
   mf_clones_set2_ch1 = [maskfolder2+"mask"+s[(len(imgfolder2)+3):] for s in f_clones_set2_ch1]
   mf_clones_set2_ch2 = [maskfolder2+"mask"+s[(len(imgfolder2)+3):] for s in f_clones_set2_ch2]
   mf_clones_set2_ch3 = [maskfolder2+"mask"+s[(len(imgfolder2)+3):] for s in f_clones_set2_ch3]
   mf_clones_set2_ch4 = [maskfolder2+"mask"+s[(len(imgfolder2)+3):] for s in f_clones_set2_ch4]
   # combine sets
   f_clones_ch1 = f_clones_set1_ch1 + f_clones_set2_ch1
   f_clones_ch2 = f_clones_set1_ch2 + f_clones_set2_ch2
   f_clones_ch3 = f_clones_set1_ch3 + f_clones_set2_ch3
   f_clones_ch4 = f_clones_set1_ch4 + f_clones_set2_ch4
   mf_clones_ch1 = mf_clones_set1_ch1 + mf_clones_set2_ch1
   mf_clones_ch2 = mf_clones_set1_ch2 + mf_clones_set2_ch2
   mf_clones_ch3 = mf_clones_set1_ch3 + mf_clones_set2_ch3
   mf_clones_ch4 = mf_clones_set1_ch4 + mf_clones_set2_ch4
   # make positive clone samples sets
   fsamps_clones_ch1 = [(im, msk) for im, msk in zip(f_clones_ch1, mf_clones_ch1)]
   fsamps_clones_ch2 = [(im, msk) for im, msk in zip(f_clones_ch2, mf_clones_ch2)]
   fsamps_clones_ch3 = [(im, msk) for im, msk in zip(f_clones_ch3, mf_clones_ch3)]
   fsamps_clones_ch4 = [(im, msk) for im, msk in zip(f_clones_ch4, mf_clones_ch4)]

   #####
   # equalise lengths
   def equalise_length(vec_in, maxlength):
      numtimes = max(1, int(maxlength // len(vec_in)))
      vec_out = []
      for i in range(numtimes): vec_out += vec_in
      return vec_out
   
   max_lengtht = max(len(tsamps_clones_ch1), len(tsamps_clones_ch2), len(tsamps_clones_ch3), len(tsamps_clones_ch4))
   max_lengthf = max(len(fsamps_clones_ch1), len(fsamps_clones_ch2), len(fsamps_clones_ch3), len(fsamps_clones_ch4))
   # true
   tsamps_clones_ch1 = equalise_length(tsamps_clones_ch1, max_lengtht)
   tsamps_clones_ch2 = equalise_length(tsamps_clones_ch2, max_lengtht)
   tsamps_clones_ch3 = equalise_length(tsamps_clones_ch3, max_lengtht)
   tsamps_clones_ch4 = equalise_length(tsamps_clones_ch4, max_lengtht)
   # false
   fsamps_clones_ch1 = equalise_length(fsamps_clones_ch1, max_lengthf)
   fsamps_clones_ch2 = equalise_length(fsamps_clones_ch2, max_lengthf)
   fsamps_clones_ch3 = equalise_length(fsamps_clones_ch3, max_lengthf)
   fsamps_clones_ch4 = equalise_length(fsamps_clones_ch4, max_lengthf)

   # join up
   samples_cl_t += tsamps_clones_ch1 + tsamps_clones_ch2 + tsamps_clones_ch3 + tsamps_clones_ch4
   samples_cl_f += fsamps_clones_ch1 + fsamps_clones_ch2 + fsamps_clones_ch3 + fsamps_clones_ch4

   ## Train   
   curr_weights = "/weights/cryptfuficlone_split_weights2.hdf5"
   weights_name = dnnfolder+"/weights/cryptfuficlone_split_weights3.hdf5"
   model_split.load_weights(dnnfolder+curr_weights)
   logs_folder = dnnfolder+"/logs"
   
   callbacks = [EarlyStopping(monitor='loss', patience=500, verbose=1, min_delta=1e-11),
                ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50, verbose=1, min_delta=1e-9),
                ModelCheckpoint(monitor='loss', filepath=weights_name, save_best_only=True, save_weights_only=True),
                TensorBoard(log_dir=logs_folder)]
                #TensorBoardImage(log_dir=logs_folder, tags=test_tags, test_image_batches=test_batches)]
                
   model_split.fit_generator(generator=train_generator(), steps_per_epoch=np.ceil(float(len(samples_cl_t)) / float(batch_size)), epochs=epochs, verbose=1, callbacks=callbacks, validation_data=None)

