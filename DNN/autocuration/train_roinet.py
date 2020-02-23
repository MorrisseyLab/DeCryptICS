## test ROI context network
import cv2
import glob
import io
import pickle
import keras
import numpy as np
import os
from random                      import shuffle
from DNN.augmentation            import plot_img
from keras.callbacks             import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers            import RMSprop
from keras.preprocessing.image   import img_to_array
from PIL                         import Image
from pathlib                     import Path
from MiscFunctions               import read_cnt_text_file, add_offset, rescale_contours
from DNN.autocuration.datagen    import *
from DNN.autocuration.context_net import *

positive_data = read_data(read_new = False, read_negative = False)
validation_data = np.load(datapath+"validation_data.npy", allow_pickle=True)

## for a query crypt, find all crypts whose bounding box lie inside tile

data = positive_data[0,:]
XY = data[:2]
imgpath = data[5]
cntpath = data[6]
ind_m = int(data[2])
clone_bool = int(data[3])
      thisname = self.datapath + imgpath.split('/')[-1].split('.svs')[0]
      slide_data = np.load(thisname + '_data.npy')
      cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)

def pull_centered_img(XY, imgpath, tilesize, ROTATE=False, RT_m=0, dwnsample_lvl=1):
   # get slide maximum dims
   slide = osl.OpenSlide(imgpath)
   maxdims = slide.level_dimensions[dwnsample_lvl]
   # Fix tiles to specific DNN size
   xy_m = XY/slide.level_downsamples[dwnsample_lvl]
   if ROTATE==False:
      xy_vals_m_ds_out = centred_tile(xy_m, tilesize, maxdims, edge_adjust = False)
      # pull out images
      Lx = np.maximum(0,int(np.around(xy_vals_m_ds_out[0])))
      Rx = int(np.around((xy_vals_m_ds_out[0]+tilesize)))
      Ty = np.maximum(0,int(np.around(xy_vals_m_ds_out[1])))
      By = int(np.around((xy_vals_m_ds_out[1]+tilesize)))
      img = getROI_img_osl(imgpath, (Lx,Ty), (tilesize, tilesize), dwnsample_lvl)
      # pad if required
      img = pad_edge_image(img.copy(), tilesize//2, xy_m, maxdims)
      return img
   else:
      xy_vals_m_ds_out = centred_tile(xy_m, 2*tilesize, maxdims, edge_adjust = False)
      # pull out images
      Lx = np.maximum(0,int(np.around(xy_vals_m_ds_out[0])))
      Rx = int(np.around((xy_vals_m_ds_out[0]+2*tilesize)))
      Ty = np.maximum(0,int(np.around(xy_vals_m_ds_out[1])))
      By = int(np.around((xy_vals_m_ds_out[1]+2*tilesize)))
      img = getROI_img_osl(imgpath, (Lx,Ty), (2*tilesize, 2*tilesize), dwnsample_lvl)
      # pad if required
      img = pad_edge_image(img.copy(), tilesize, xy_m, maxdims)
      img = cv2.warpAffine(img.copy(), RT_m[:2,:], img.shape[1::-1])
      img = img[(tilesize//2):-(tilesize//2), (tilesize//2):-(tilesize//2), :]
      return img
