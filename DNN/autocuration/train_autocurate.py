##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:16:23 2018

@author: doran
"""
import cv2
import glob
import io
import pickle
import keras
import numpy as np
import os
from random                      import shuffle
from DNN.augmentation            import plot_img, randomHueSaturationValue,\
                                        ReproducerandomHueSaturationValue,\
                                        HorizontalFlip
from keras.callbacks             import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,\
                                        TensorBoard
from keras.optimizers            import RMSprop
from keras.preprocessing.image   import img_to_array
from PIL                         import Image
from pathlib                     import Path
from DeCryptICS.MiscFunctions    import read_cnt_text_file, add_offset, rescale_contours
#from DNN.autocuration.mutant_net import *
from DNN.autocuration.context_net import *
from DNN.autocuration.datagen    import *
from DNN.autocuration.train_generator3 import *

positive_data = read_data(read_new = False, read_negative = False)
validation_data = np.load(datapath+"validation_data.npy", allow_pickle=True)

#if os.path.exists(datapath+"train_inds_p.npy"):
#   train_inds_p = np.load(datapath+"train_inds_p.npy")
#   val_inds_p = np.load(datapath+"val_inds_p.npy")
#   train_inds_n = np.load(datapath+"train_inds_n.npy")
#   val_inds_n = np.load(datapath+"val_inds_n.npy")
#else:
#   ## take random sample and output as validation set for reproduceability
#   val_size = 10.
#   all_inds_p = np.array(range(positive_data.shape[0]))
#   val_inds_p = np.random.choice(all_inds_p, size=int(positive_data.shape[0]/val_size), replace=False)
#   train_inds_p = np.setdiff1d(all_inds_p, val_inds_p)
#   np.save(datapath+"train_inds_p.npy", train_inds_p)
#   np.save(datapath+"val_inds_p.npy", val_inds_p)
#   all_inds_n = np.array(range(negative_data.shape[0]))
#   val_inds_n = np.random.choice(all_inds_n, size=int(negative_data.shape[0]/val_size), replace=False)
#   train_inds_n = np.setdiff1d(all_inds_n, val_inds_n)
#   np.save(datapath+"train_inds_n.npy", train_inds_n)
#   np.save(datapath+"val_inds_n.npy", val_inds_n)

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

epochs = 20000
batch_size = 10
tilesize = 512 # 50
#sizesmall = 384
#nn = 30
#nn_sampsize = 200
if keras.backend._BACKEND=="mxnet":
   import mxnet
   input_size = (3, tilesize, tilesize)
#   input_size1 = (3*nn, tilesize, tilesize)
#   input_size2 = (3, sizesmall, sizesmall)
   chan_num = 1
else:
   input_size = (tilesize, tilesize, 3)
#   input_size1 = (tilesize, tilesize, 3*nn)
#   input_size2 = (sizesmall, sizesmall, 3)
   chan_num = 3
 
if __name__=="__main__":

   ## ideas:
   #     - check it's using both branches (black out each independently)
   #     - new network structure: subtract both branches then dense
   #     - validation set from OTHER slides not in training set, plus more WT crypts
   #     - weighted loss function?
   #     - might be too many reduction layers, especially in context map (~16x16 at end)

   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/autocuration/"
#   model = rcnet3(input_shape=input_size, chan_num=chan_num)
   model = test(input_shape=input_size, chan_num=chan_num)
  
#   weights_name_curr = dnnfolder + "/weights/autocurate_contextnet_rc1.hdf5"
#   model.load_weights(weights_name_curr)
   
   weights_name = dnnfolder + "/weights/autocurate_contextnet_rc1.hdf5"
   logs_folder = dnnfolder + "/logs"
   
   batch_size = 1
#   train_datagen = train_generator(tilesize, sizesmall, positive_data, batch_size, nn=nn, nn_sampsize=nn_sampsize)
#   valid_datagen = validation_generator(tilesize, sizesmall, validation_data, batch_size)
   train_datagen = train_generator(tilesize, positive_data[:30,:], batch_size)
   valid_datagen = validation_generator(tilesize, validation_data[:30,:], batch_size)
   
   callbacks = [EarlyStopping(monitor='loss', patience=350, verbose=1, min_delta=1e-9),
                ReduceLROnPlateau(monitor='loss', factor=0.075, patience=25, verbose=1, min_delta=1e-9),
                ModelCheckpoint(monitor='loss', filepath=weights_name, save_best_only=True, save_weights_only=True, verbose=1)]
#                TensorBoard(log_dir=logs_folder)]

   res = model.fit_generator(
            generator=train_datagen,
            steps_per_epoch=len(train_datagen),
            validation_data=valid_datagen,
            validation_steps=len(valid_datagen),
            verbose = 1,
            epochs = epochs,
            workers = 14,
            use_multiprocessing = True,
            callbacks=callbacks
            )

   ## testing
   i = 60
   # train set
   b1, m1 = train_datagen[i]
   p1 = model.predict(b1)
   badinds1 = np.where(abs(m1-p1)>0.1)[0]
   for j in badinds1:
      print("pred: %1.2f, mask: %1.2f" % (p1[j,0], m1[j,0]))
#      plot_img(b1[0][j,:,:,:], hold_plot=False, nameWindow="a")
      plot_img(b1[1][j,:,:,:])
         
   # validation set
   b2, m2 = valid_datagen[i]
   p2 = model.predict(b2)
   badinds2 = np.where(abs(m2-p2)>0.1)[0]
   for j in badinds2:
      print("pred: %1.2f, mask: %1.2f" % (p2[j,0], m2[j,0]))
      plot_img(b2[0][j,:,:,:3], hold_plot=False, nameWindow="a")
      plot_img(b2[1][j,:,:,:])
   
   ## cycle through batch
   for k in range(m1.shape[0]):
      print("pred: %1.2f, mask: %1.2f" % (p1[k,0], m1[k,0]))
      plot_img(b1[0][k,:,:,:3], hold_plot=False, nameWindow="a")
      plot_img(b1[1][k,:,:,:])
   for k in range(m2.shape[0]):
      print("pred: %1.2f, mask: %1.2f" % (p2[k,0], m2[k,0]))
      plot_img(b2[0][k,:,:,:3], hold_plot=False, nameWindow="a")
      plot_img(b2[1][k,:,:,:])
   
   ## black out one image
   b1, m1 = train_datagen[i]
   for k in range(b1[0].shape[0]):
      b1[0][k,:,:,:] = 0 # black-out an image
   p11 = model.predict(b1)
#   p22 = model2.predict(b1)
   for k in range(m1.shape[0]):
      print("pred: %1.2f, blankpred: %1.2f,  mask: %1.2f" % (p1[k,0], p11[k,0], m1[k,0]))
#      print("pred2: %1.2f, mask: %1.2f" % (p22[k,0], m2[k,0]))
      plot_img(b1[0][k,:,:,:3], hold_plot=False, nameWindow="a")
      plot_img(b1[1][k,:,:,:])
   
   k = 1
   plot_img(b1[0][k,:,:,:], hold_plot=False, nameWindow="a")
   plot_img(b1[1][k,:,:,:], hold_plot=False, nameWindow="b")      
   
   
   ## checking training and validation set for loading errors
   for i in range(len(train_datagen)):
      if (i%100==0): print(i)
      a = train_datagen[i]
   
   for i in range(len(valid_datagen)):
      if (i%10==0): print(i)
      a = valid_datagen[i]
   
   
