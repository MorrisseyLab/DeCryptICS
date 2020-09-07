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
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
from random                      import shuffle
from DNN.augmentation            import plot_img, randomHueSaturationValue,\
                                        ReproducerandomHueSaturationValue,\
                                        HorizontalFlip
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers            import RMSprop
from tensorflow.keras.preprocessing.image   import img_to_array
from PIL                               import Image
from pathlib                           import Path
from DeCryptICS.MiscFunctions          import read_cnt_text_file, add_offset, rescale_contours
#from DNN.autocuration.mutant_net       import *
from DNN.autocuration.context_net      import *
from DNN.autocuration.datagen          import *
from DNN.autocuration.train_generator5 import *

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

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

epochs = 20000
batch_size = 2
tilesize = 512
crsize = 384
#nn = 30
#nn_sampsize = 200

chan_num = 3
input_size1 = (tilesize, tilesize, 3)
input_size2 = (crsize, crsize, 3)
#   input_size1 = (tilesize, tilesize, 3*nn)
#   input_size2 = (sizesmall, sizesmall, 3)
 
if __name__=="__main__":

   ## ideas:
   #     - check it's using both branches (black out each independently)
   #     - new network structure: subtract both branches then dense
   #     - validation set from OTHER slides not in training set, plus more WT crypts
   #     - weighted loss function?
   #     - might be too many reduction layers, especially in context map (~16x16 at end)

   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/autocuration/"
   model = att_roi_net2(input_shape1=input_size1, input_shape2=input_size2,
                       d_model=64, depth_k=6, depth_v=8, num_heads=2, dff=128, dropout_rate=0.3)
#   model = rcnet3(input_shape=input_size, chan_num=chan_num)
#   model = rcnet_reducedtest(input_shape=input_size, chan_num=chan_num)
#   model = test(input_shape=input_size, chan_num=chan_num)
  
#   weights_name_curr = dnnfolder + "/weights/autocurate_contextnet_rc2.hdf5"
#   model.load_weights(weights_name_curr)
   
   weights_name = dnnfolder + "/weights/att_roi_net_w2.hdf5"
   logs_folder = dnnfolder + "/logs"
   
   train_datagen = train_generator(tilesize, crsize, positive_data, batch_size)
   valid_datagen = validation_generator(tilesize, crsize, validation_data, 2*batch_size)
   vd1 = valid_datagen[0]
   
   callbacks = [EarlyStopping(monitor='loss', patience=350, verbose=1, min_delta=1e-9),
                ReduceLROnPlateau(monitor='loss', factor=0.075, patience=25, verbose=1, min_delta=1e-9),
                ModelCheckpoint(monitor='loss', filepath=weights_name, save_best_only=True, save_weights_only=True, verbose=1)]

   res = model.fit(
            x=train_datagen,
            steps_per_epoch=len(train_datagen),
            validation_data=vd1,
            verbose = 1,
            epochs = epochs,
            callbacks=callbacks,
            workers = 14
            )

   ## testing
   i = 2000
   # train set
   b1, m1 = train_datagen[i]
   p1 = model.predict(b1)
   ## cycle through batch
   for k in range(m1.shape[0]):
      print("pred: %1.2f, mask: %1.2f" % (p1[k,0], m1[k,0]))
#      plot_img(b1[0][k,:,:,:3], hold_plot=False, nameWindow="a")
      plot_img(b1[0][k,:,:,:])
   ## cycle through bad preds
   badinds1 = np.where(abs(m1-p1)>0.1)[0]
   for k in badinds1:
      print("pred: %1.2f, mask: %1.2f" % (p1[k,0], m1[k,0]))
      plot_img(b1[1][k,:,:,:])
         
   # validation set
   b2, m2 = valid_datagen[10]
   p2 = model.predict(b2)
   ## cycle through batch
   for k in range(m2.shape[0]):
      print("pred: %1.2f, mask: %1.2f" % (p2[k,0], m2[k,0]))
#      plot_img(b2[0][k,:,:,:3], hold_plot=False, nameWindow="a")
      plot_img(b2[0][k,:,:,:])
   ## cycle through bad preds
   badinds2 = np.where(abs(m2-p2)>0.1)[0]
   for k in badinds2:
      print("pred: %1.2f, mask: %1.2f" % (p2[k,0], m2[k,0]))
      plot_img(b2[1][k,:,:,:])
   
   
   ## black out one image
   b1, m1 = train_datagen[i]
   for k in range(b1[0].shape[0]):
      b1[0][k,:,:,:] = 0 # black-out an image
   p11 = model.predict(b1)
   for k in range(m1.shape[0]):
      print("pred: %1.2f, blankpred: %1.2f,  mask: %1.2f" % (p1[k,0], p11[k,0], m1[k,0]))
      plot_img(b1[0][k,:,:,:3], hold_plot=False, nameWindow="a")
      plot_img(b1[1][k,:,:,:])
      
   
