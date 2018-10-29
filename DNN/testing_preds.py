#!/usr/bin/env python3
import tensorflow as tf
from keras import backend as K
import cv2, os, time, glob
import numpy as np
import keras
import pickle
from keras.preprocessing.image import img_to_array
import DNN.u_net as unet
import DNN.params as params
from deconv_mat               import *
from automaticThresh_func     import calculate_deconvolution_matrix_and_ROI, find_deconmat_fromtiles
from MiscFunctions            import simplify_contours, col_deconvol_and_blur2, mkdir_p, write_clone_image_snips, convert_to_local_clone_indices
from MiscFunctions            import getROI_img_osl, add_offset, write_cnt_text_file, plot_img, rescale_contours, write_score_text_file
from cnt_Feature_Functions    import joinContoursIfClose_OnlyKeepPatches, st_3, contour_Area, plotCnt
from multicore_morphology     import getForeground_mc
from GUI_ChooseROI_class      import getROI_svs
from Segment_clone_from_crypt import find_clone_statistics, combine_feature_lists, determine_clones, determine_clones_gridways
from Segment_clone_from_crypt import subset_clone_features, add_xy_offset_to_clone_features, write_clone_features_to_file
from knn_prune                import remove_tiling_overlaps_knn
from DNN_segment              import mask_to_contours

# Load DNN model
model = params.model_factory(input_shape=(params.input_size, params.input_size, 3), num_classes=5)
model.load_weights("/home/doran/Work/py_code/experimental_DeCryptICS/DNN/weights/cryptfuficlone_weights5.hdf5")

training_base_folder = "/home/doran/Work/py_code/DeCryptICS/DNN/"
# Set up training data   
imgfolder = training_base_folder + "/input/train/"
maskfolder = training_base_folder + "/input/train_masks/"
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

clones_stag = glob.glob(imgfolder + "*STAG2*_clone.png")
samples_stag = []
for i in range(len(clones_stag)):
   mask = maskfolder+"mask"+clones_stag[i][(len(imgfolder)+3):]
   sample = (clones_stag[i], mask)
   samples_stag.append(sample)
   
# crypts
i = 8
img = cv2.imread(samples_cr[i][0], cv2.IMREAD_COLOR)
mask = cv2.imread(samples_cr[i][1], cv2.IMREAD_GRAYSCALE)
batch = np.array([np.array(img, dtype=np.float32) / 255])
pred = model.predict(batch)
pp = pred[0,:,:,:]
plot_img(img, hold_plot=False, nameWindow="img")
plot_img((pp[:,:,0],pp[:,:,1],pp[:,:,2],pp[:,:,3],pp[:,:,4]))

# fufis
i = 100
img = cv2.imread(samples_fu[i][0], cv2.IMREAD_COLOR)
mask = cv2.imread(samples_fu[i][1], cv2.IMREAD_GRAYSCALE)
batch = np.array([np.array(img, dtype=np.float32) / 255])
pred = model.predict(batch)
pp = pred[0,:,:,:]
plot_img(img, hold_plot=False, nameWindow="img")
plot_img((pp[:,:,0],pp[:,:,1],pp[:,:,2],pp[:,:,3],pp[:,:,4]))

# clones
i = 7
img = cv2.imread(samples_cl[i][0], cv2.IMREAD_COLOR)
mask = cv2.imread(samples_cl[i][1], cv2.IMREAD_GRAYSCALE)
batch = np.array([np.array(img, dtype=np.float32) / 255])
pred = model.predict(batch)
pp = pred[0,:,:,:]
print(samples_cl[i][0])
plot_img(img, hold_plot=False, nameWindow="img")
plot_img((pp[:,:,0],pp[:,:,1],pp[:,:,2],pp[:,:,3],pp[:,:,4]))

for i in range(len(samples_cl)):
   img = cv2.imread(samples_cl[i][0], cv2.IMREAD_COLOR)
   mask = cv2.imread(samples_cl[i][1], cv2.IMREAD_GRAYSCALE)
   batch = np.array([np.array(img, dtype=np.float32) / 255])
   pred = model.predict(batch)
   pp = pred[0,:,:,:]
   print(samples_cl[i][0])
   plot_img(img, hold_plot=False, nameWindow="img")
   plot_img((pp[:,:,0],pp[:,:,1],pp[:,:,2],pp[:,:,3],pp[:,:,4]))

for i in range(len(samples_stag)):
   img = cv2.imread(samples_stag[i][0], cv2.IMREAD_COLOR)
   mask = cv2.imread(samples_stag[i][1], cv2.IMREAD_GRAYSCALE)
   batch = np.array([np.array(img, dtype=np.float32) / 255])
   pred = model.predict(batch)
   pp = pred[0,:,:,:]
   print(samples_stag[i][1])
   plot_img(img, hold_plot=False, nameWindow="img")
   plot_img((pp[:,:,0],pp[:,:,1],pp[:,:,2],pp[:,:,3],pp[:,:,4]))

## exporting the crypt predictions for all the clone tiles for manual curation and addition to training set
#for i in range(len(samples_cl)):
#   img = cv2.imread(samples_cl[i][0], cv2.IMREAD_COLOR)
#   batch = np.array([np.array(img, dtype=np.float32) / 255])
#   pred = model.predict(batch)
#   newcnts = mask_to_contours(pred, 0.4)
#   for k in range(pred.shape[3]):
#      newcnts[k] = [cc for cc in newcnts[k] if len(cc)>4]
#   crypt_contours  = newcnts[0]
#   print(i)
#   # Save img output
#   newname = samples_cl[i][0].split('.')[0] + '_iter3_crypt.png'
#   cv2.imwrite(newname, img)
#   # Reduce img below 255 and draw on contours
#   img[img<21] -= 20
#   for j in range(len(crypt_contours)):
#      cv2.drawContours(img, [crypt_contours[j]], 0, (255,255,255), -1)
#   # Save as premask
#   premaskname = newname.split('/train/img_')[0] + '/pre-mask/premask_' + newname.split('/train/img_')[1]
#   cv2.imwrite(premaskname, img)

'''
Notes:
   - KDM6A, NONO and MAOA tend to fire at the same time, lump training data together?
   - Fufis firing on clones, need to either make sure training data is assigning correctly
      or add in more fufi data with variety of clone staining as negatives/positives
   - mPAS clones not firing in clone channel, and mPAS pos crypts not registering as crypts!
   - In fact mPAS training set basically all missing?!
   - I ruined all tiling data by not saving raw images and overwriting premasks.... 
'''

#############################################################
## Testing mask assignment to check why fufis are firing on clones
def train_process(data):
   outlist = []
   img_f, mask_f = data
   # Order clone channels: KDM6A, MAOA, NONO, STAG2, mPAS
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]
   if (mname[-5:]=="crypt"):
      dontmask = 0
   elif (mname[-4:]=="fufi"):
      dontmask = 1
   elif (mname[-5:]=="clone"):
      mname_broken = mask_f.split('/')[-1].split('_')
      if "KDM6A" in mname_broken:
         dontmask = 2
      if "MAOA" in mname_broken:
         dontmask = 3
      if "NONO" in mname_broken:
         dontmask = 4
      if "STAG2" in mname_broken:
         dontmask = 5
      if "mPAS" in mname_broken:
         dontmask = 6
   print(dontmask)

#   for i in range(mask.shape[2]):
#      if (not i==dontmask):
#         mask[:,:,i].fill(MASK_VALUE)
#   return outlist
