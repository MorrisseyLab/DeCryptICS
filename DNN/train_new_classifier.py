#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:16:23 2018

@author: doran
"""
import tensorflow as tf
from keras import backend as K
#import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import DNN.u_net as unet
import DNN.params as params
from DNN.augmentation import plot_img, randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip, fix_mask
from DNN.losses       import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from keras.callbacks  import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import img_to_array
from keras.optimizers import RMSprop
from random import shuffle

num_cores = 12
GPU = True
CPU = False

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

input_size = params.input_size
SIZE = (input_size, input_size)
epochs = params.max_epochs
batch_size = params.batch_size

# Processing function for the training data
def train_process(data, check_class=True):
   img_f, mask_f = data
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   if (not check_class==True):
      mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
      if (not mask.shape==SIZE): mask = cv2.resize(mask, SIZE)
   if (check_class==True):
      mask = np.zeros([img.shape[0], img.shape[1], 2]) # for two classifications
      mc = cv2.imread(mask_f+"crypt.png", cv2.IMREAD_GRAYSCALE)
      mf = cv2.imread(mask_f+"fufi.png", cv2.IMREAD_GRAYSCALE)
      mfb = cv2.imread(mask_f+"fufi_black.png", cv2.IMREAD_GRAYSCALE)
      mask[:,:,0] = mc # 1 is reversed
      #mask[:,:,0] = mf # 0 is reversed
      mask[:,:,1].fill(np.nan) # 0 is reversed
      
#      if (mask_f.split('/')[-1][:4]=="mask"):
#         mask[:,:,0] = mm
#         mask[:,:,1].fill(np.nan)
#      elif (mask_f.split('/')[-1][:4]=="fufi"):
#         mask[:,:,0].fill(np.nan)
#         mask[:,:,1] = mm
   img = randomHueSaturationValue(img,
                                hue_shift_limit=(-100, 100),
                                sat_shift_limit=(0, 0),
                                val_shift_limit=(-25, 25))
   img, mask = randomShiftScaleRotate(img, mask,
                                    shift_limit=(-0.0625, 0.0625),
                                    scale_limit=(-0.1, 0.1),
                                    rotate_limit=(-20, 20))
   img, mask = randomHorizontalFlip(img, mask)
   fix_mask(mask)
   #mask = np.expand_dims(mask, axis=2)
   return (img, mask)

def train_generator():
    while True:
        for start in range(0, len(samples), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(samples))
            ids_train_batch = samples[start:end]
            for ids in ids_train_batch:
                img, mask = train_process(ids)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

if __name__=="__main__":
   base_folder = "/home/doran/Work/py_code/DeCryptICS/DNN/"
   
   ## Loading old weights into all but the final layer
   model = params.model_factory(input_shape=(params.input_size, params.input_size, 3))
   model.load_weights("./DNN/weights/tile256_for_X_best_weights.hdf5")

   # Getting weights layer by layer
   weights_frozen = [l.get_weights() for l in model.layers]

   # Redefine new network with new classification (here we are using just one class, to then copy into a 2-class network later)
   model = params.model_factory(input_shape=(params.input_size, params.input_size, 3), num_classes=2)

   # Add in old weights, not including final layer
   numlayers = len(model.layers)
   for i in range(numlayers-1):
      model.layers[i].set_weights(weights_frozen[i])

   w_elems = []
   w_f_elems = weights_frozen[-1]
   for i in range(len(model.layers[-1].get_weights())):
      w_elems.append(model.layers[-1].get_weights()[i])   
   w_elems[0][:,:,:,0] = w_f_elems[0][:,:,:,0]
   w_elems[1][0] = w_f_elems[1][0]
   # set new class weights to zero as known base line
   w_elems[0][:,:,:,1] = 0
   w_elems[1][1] = 0
   # and cryptfinding to zero for test
   w_elems[0][:,:,:,0] = 0
   w_elems[1][0] = 0
   
   model.layers[-1].set_weights(w_elems)
   old_weights = model.layers[-1].get_weights()

   # Freeze all layer but the last classification convolution (as difficult to freeze a subset of parameters within a layer -- but can load them back in afterwards)
   for layer in model.layers[:-1]:
      layer.trainable = False
   # To check whether we have successfully frozen layers, check model.summary() before and after re-compiling
   model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

   # testing missing data masking
   test_img_f = "/home/doran/Work/py_code/DeCryptICS/DNN/input/test_img.png"
   test_mask_f = "/home/doran/Work/py_code/DeCryptICS/DNN/input/test_mask_"
   test_img             = cv2.imread("./DNN/input/test_img.png")
   test_mask_crypt      = cv2.imread("./DNN/input/test_mask_crypt.png", cv2.IMREAD_GRAYSCALE)
   test_mask_fufi       = cv2.imread("./DNN/input/test_mask_fufi.png", cv2.IMREAD_GRAYSCALE)
   test_mask_fufi_black = cv2.imread("./DNN/input/test_mask_fufi_black.png", cv2.IMREAD_GRAYSCALE)
   samples = [(test_img_f, test_mask_f)]
   
   model.load_weights(base_folder+'/weights/fufi_weights_nan.hdf5')
   nanweights = model.layers[-1].get_weights()
   model.load_weights(base_folder+'/weights/fufi_weights_f.hdf5')
   fweights = model.layers[-1].get_weights()
   #model.load_weights(base_folder+'/weights/fufi_weights_fb.hdf5')
   #fbweights = model.layers[-1].get_weights()
   model.load_weights(base_folder+'/weights/fufi_weights_nan_r.hdf5')
   nanweights_r = model.layers[-1].get_weights()
   model.load_weights(base_folder+'/weights/fufi_weights_f_r.hdf5')
   fweights_r = model.layers[-1].get_weights()
   #model.load_weights(base_folder+'/weights/fufi_weights_fb_r.hdf5')
   #fbweights_r = model.layers[-1].get_weights()
   
   diffnan1 = old_weights[0][0,0,:,1] - nanweights[0][0,0,:,1]
   diffnan2 = old_weights[1][1] - nanweights[1][1]
   difff1 = old_weights[0][0,0,:,1] - fweights[0][0,0,:,1]
   difff2 = old_weights[1][1] - fweights[1][1]
   #difffb1 = old_weights[0][0,0,:,1] - fbweights[0][0,0,:,1]
   #difffb2 = old_weights[1][1] - fbweights[1][1]
   
   diffnan1_r = old_weights[0][0,0,:,1] - nanweights_r[0][0,0,:,1]
   diffnan2_r = old_weights[1][1] - nanweights_r[1][1]
   difff1_r = old_weights[0][0,0,:,1] - fweights_r[0][0,0,:,1]
   difff2_r = old_weights[1][1] - fweights_r[1][1]
   #difffb1_r = old_weights[0][0,0,:,1] - fbweights_r[0][0,0,:,1]
   #difffb2_r = old_weights[1][1] - fbweights_r[1][1]
   
   cdiffnan1 = old_weights[0][0,0,:,0] - nanweights[0][0,0,:,0]
   cdiffnan2 = old_weights[1][0] - nanweights[1][0]
   cdifff1 = old_weights[0][0,0,:,0] - fweights[0][0,0,:,0]
   cdifff2 = old_weights[1][0] - fweights[1][0]
   #cdifffb1 = old_weights[0][0,0,:,0] - fbweights[0][0,0,:,0]
   #cdifffb2 = old_weights[1][0] - fbweights[1][0]   

   cdiffnan1_r = old_weights[0][0,0,:,0] - nanweights_r[0][0,0,:,0]
   cdiffnan2_r = old_weights[1][0] - nanweights_r[1][0]
   cdifff1_r = old_weights[0][0,0,:,0] - fweights_r[0][0,0,:,0]
   cdifff2_r = old_weights[1][0] - fweights_r[1][0]
   #cdifffb1_r = old_weights[0][0,0,:,0] - fbweights_r[0][0,0,:,0]
   #cdifffb2_r = old_weights[1][0] - fbweights_r[1][0]   

   # We see that for the NaN output masking, both end weights are changed
   # by the same amount, whereas for the non NaN ouputs the change in the
   # weights is different.  Then-- is the NaN masking doing what we want,
   # and the relative change of zero is to preserve the gradients in the
   # classification?

   # Set up training data
   
   imgfolder = base_folder + "/input/fufis/train/"
   maskfolder = base_folder + "/input/fufis/train_masks/"
   images = glob.glob(imgfolder + "*.png")
   #masks = glob.glob(maskfolder + "*.png")
   samples = []
   #for i in range(len(masks)):
   for i in range(len(images)):
      #mask = maskfolder+"mask"+images[i][(len(imgfolder)+3):] # when images have "img_" prefix
      #sample = (images[i], mask)
      #img = imgfolder+"img"+masks[i][(len(maskfolder)+4):]
      #sample = (img, masks[i])
      mask = maskfolder+"mask_"+images[i][len(imgfolder):] # when images don't have "img_" prefix
      sample = (images[i], mask)
      samples.append(sample)
   shuffle(samples)

   callbacks = [EarlyStopping(monitor='loss',
                            patience=10,
                            verbose=1,
                            min_delta=1e-8),
              ReduceLROnPlateau(monitor='loss',
                                factor=0.1,
                                patience=20,
                                verbose=1,
                                epsilon=1e-8),
              ModelCheckpoint(monitor='loss',
                              filepath=base_folder+'/weights/fufi_weights_nansubset.hdf5',
                              save_best_only=True,
                              save_weights_only=True),
              TensorBoard(log_dir='logs')]

   model.fit_generator(generator=train_generator(),
                     steps_per_epoch=np.ceil(float(len(samples)) / float(batch_size)),
                     epochs=epochs,
                     verbose=1,
                     callbacks=callbacks,
                     validation_data=None
                     )
     
