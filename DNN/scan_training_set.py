#!/usr/bin/env python3

import cv2, os, sys
import openslide as osl
import pandas as pd
import numpy as np
import glob
import keras
import pickle
import DNN.u_net  as unet
import DNN.params as params
from keras.preprocessing.image import img_to_array
from knn_prune                 import *
from MiscFunctions             import simplify_contours, write_clone_image_snips,\
                                     convert_to_local_clone_indices, mkdir_p,\
                                     getROI_img_osl, add_offset, write_cnt_text_file,\
                                     rescale_contours, write_score_text_file, plot_img
from cnt_Feature_Functions     import joinContoursIfClose_OnlyKeepPatches, contour_Area,\
                                     contour_EccMajorMinorAxis, contour_xy
from GUI_ChooseROI_class       import getROI_svs

if keras.backend._BACKEND=="tensorflow":
   import tensorflow as tf
   input_shape = (params.input_size_train, params.input_size_train, 3)
   chan_num = 3
elif keras.backend._BACKEND=="mxnet":
   import mxnet
   input_shape = (3, params.input_size_train, params.input_size_train)
   chan_num = 1
num_classes = 6
model = params.model_factory(input_shape=input_shape, num_classes=num_classes, chan_num=chan_num)
model2 = params.model_factory(input_shape=input_shape, num_classes=5, chan_num=chan_num)
maindir = '/home/doran/Work/py_code/DeCryptICS/'
weightsin = os.path.join(maindir, 'DNN', 'weights', 'cryptfuficlone_split_weights.hdf5')
weightsin2 = os.path.join(maindir, 'DNN', 'weights', 'cryptfuficlone_weights.hdf5')
model.load_weights(weightsin)
model2.load_weights(weightsin2)

trainfolder = maindir + '/DNN/input/new_train_set/'

imgs_in = glob.glob(trainfolder + 'img/*/*/*.png')
with open(trainfolder + 'performance_database.csv', 'w') as fo:
   fo.write('Image_path\tpred_diff1\tpred_diff2\tpred_diff3\n')
   for i in range(len(imgs_in)):
      im = cv2.imread(imgs_in[i], cv2.IMREAD_COLOR)
      curmask = cv2.imread(imgs_in[i].replace('img', 'mask'), cv2.IMREAD_GRAYSCALE)
      newmask = model.predict(np.array([im/255], dtype=np.float32))
      newmask2 = model2.predict(np.array([im/255], dtype=np.float32))
      
      # choose channel
      if 'T_clone.png' in imgs_in[i]:
         mark = imgs_in[i].split('/')[-1].split('_T_')[0].split('_')[-1]
      elif 'F_clone.png' in imgs_in[i]:
         mark = imgs_in[i].split('/')[-1].split('_F_')[0].split('_')[-1]
      if mark.upper()=="STAG2" or mark.upper()=="KDM6A" or mark.upper()=="NONO":
         chan1 = 2
         chan2 = 2
      if mark.upper()=="MAOA" or mark.upper()=="HDAC6":
         chan1 = 3
         chan2 = 2
      if mark.upper()=="P53":
         chan1 = 4
         chan2 = 3
      if mark.upper()=="MPAS":
         chan1 = 5
         chan2 = 4
      
      _, newmask_t = cv2.threshold((newmask[0,:,:,chan1]*255).astype(np.uint8), 0.01*255, 255, cv2.THRESH_BINARY)
      _, newmask2_t = cv2.threshold((newmask2[0,:,:,chan2]*255).astype(np.uint8), 0.25*255, 255, cv2.THRESH_BINARY)
      
      # find difference between newmasks 1&2, and curmask
      diff1 = cv2.countNonZero(newmask2_t - newmask_t)
      diff2 = cv2.countNonZero(newmask2_t - curmask)
      diff3 = cv2.countNonZero(newmask_t - curmask)
      fo.write('%s\t%1.8g\t%1.8g\t%1.8g\n' % (imgs_in[i], diff1, diff2, diff3) )
      if i%500==0: print(i)
   
   

   
