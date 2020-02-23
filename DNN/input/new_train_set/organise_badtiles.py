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
                                     contour_EccMajorMinorAxis, contour_xy, st_3
from GUI_ChooseROI_class       import getROI_svs

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def lower_intensity_by_one(img):
    inds = np.where(img>0)
    img[inds] = img[inds] - 1
    return img

dat = np.loadtxt('/home/doran/Work/py_code/DeCryptICS/DNN/input/new_train_set/bad_tiles_need_check.tsv', dtype=str, skiprows=1)

outbase = '/home/doran/Work/py_code/DeCryptICS/DNN/input/new_train_set/premask/'
numtiles = dat.shape[0]
spltby = 50

#for i in range(int(np.ceil(numtiles/spltby)+1)):
#   mkdir_p(outbase + '/set_' + str(i))
#   for j in range(spltby):
#      if (i*spltby + j)>=numtiles: break
#      imname = dat[i*spltby + j, 0].split('/')[-1]
#      outname = outbase + '/set_' + str(i) + '/' + imname
#      shutil.copy(dat[i*spltby + j, 0] , outname)
      
## actually run old model on *T_clone.png files, and use as initial premask.
## then just manually curate those + *F_clone.png files.

if keras.backend._BACKEND=="tensorflow":
   import tensorflow as tf
   input_shape = (params.input_size_train, params.input_size_train, 3)
   chan_num = 3
elif keras.backend._BACKEND=="mxnet":
   import mxnet
   input_shape = (3, params.input_size_train, params.input_size_train)
   chan_num = 1
model = params.model_factory(input_shape=input_shape, num_classes=5, chan_num=chan_num)
maindir = '/home/doran/Work/py_code/DeCryptICS/'
weightsin = os.path.join(maindir, 'DNN', 'weights', 'cryptfuficlone_weights.hdf5')
model.load_weights(weightsin)

for i in range(int(np.ceil(numtiles/spltby)+1)):
   print(i)
   outfolder = '/set_' + str(i)
   mkdir_p(outbase + outfolder)
   for j in range(spltby):
      if (i*spltby + j)>=numtiles: break
      imname = dat[i*spltby + j].split('/')[-1]
      outname = outbase + outfolder + '/' + imname
      im = cv2.imread(dat[i*spltby + j], cv2.IMREAD_COLOR)
      
      if 'F_clone.png' in imname:
         # we only want help with true clone tiles
         im_l = lower_intensity_by_one(im)

      if 'T_clone.png' in imname:      
         newmask = model.predict(np.array([im/255], dtype=np.float32))
          
         # choose channel
         mark = imname.split('/')[-1].split('_T_')[0].split('_')[-1]
         if mark.upper()=="STAG2" or mark.upper()=="KDM6A" or mark.upper()=="NONO":
            chan2 = 2
         if mark.upper()=="MAOA" or mark.upper()=="HDAC6":
            chan2 = 2
         if mark.upper()=="P53":
            chan2 = 3
         if mark.upper()=="MPAS":
            chan2 = 4
         
         nmsk = cv2.morphologyEx(newmask[0,:,:,chan2], cv2.MORPH_DILATE, st_3, iterations = 1)
      
         # lower from 255 then set mask elements to white
         im_l = lower_intensity_by_one(im)
         minds = np.where(nmsk>0.25)
         im_l[minds[0], minds[1], :] = 255
      
      # output in premask
      cv2.imwrite(outname, im_l)
      
      

