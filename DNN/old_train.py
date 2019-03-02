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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import img_to_array
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
def train_process(data):
    img_f, mask_f = data
    img = cv2.imread(img_f, cv2.IMREAD_COLOR)
    if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
    mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
    if (not mask.shape==SIZE): mask = cv2.resize(mask, SIZE)
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
    mask = np.expand_dims(mask, axis=2)
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
    
    # Load model and parameters
    model = params.model_factory()
    model.summary()
    
    # Set up training data
    base_folder = "/home/doran/Work/py_code/zoomed_out_DeCryptICS/DNN/"
    imgfolder = base_folder + "/input/train/"
    maskfolder = base_folder + "/input/train_masks/"
    images = glob.glob(imgfolder + "*.png")
    #masks = glob.glob(maskfolder + "*.png")
    samples = []
    #for i in range(len(masks)):
    for i in range(len(images)):
        mask = maskfolder+"mask"+images[i][(len(imgfolder)+3):]
        sample = (images[i], mask)
        #img = imgfolder+"img"+masks[i][(len(maskfolder)+4):]
        #sample = (img, masks[i])
        samples.append(sample)
    shuffle(samples)

    callbacks = [EarlyStopping(monitor='loss',
                               patience=8,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4),
                 ModelCheckpoint(monitor='loss',
                                 filepath=base_folder+'/weights/tile256_for_2048_best_weights.hdf5',
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
        
