#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:16:23 2018

@author: doran
"""
import tensorflow as tf
import keras
import cv2
import numpy as np
from DNN.model.augmentation import plot_img, randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip, fix_mask
import DNN.model.u_net as unet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import glob
import DNN.params

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) # define hardware usage
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)

# Processing function for the training data
def train_process(data):
    img_f, mask_f = data
    img = cv2.imread(img_f, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (input_size, input_size))
    mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)    
    mask = cv2.resize(mask, (input_size, input_size))
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-50, 50),
                                   sat_shift_limit=(0, 0),
                                   val_shift_limit=(-15, 15))
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
    input_size = params.input_size
    epochs = params.max_epochs
    batch_size = params.batch_size
    model = params.model_factory()
    model.summary()
    SIZE = (1024, 1024)
    
    # Set up training data
    base_folder = "/home/doran/Work/py_code/DNN_ImageMasking/"
    imgfolder = base_folder + "/input/train/"
    maskfolder = base_folder + "/input/train_masks/"
    images = glob.glob(imgfolder + "*.png")
    samples = []
    for i in range(len(images)):
        mask = maskfolder+"mask"+images[i][(len(imgfolder)+3):]
        sample = (images[i], mask)
        samples.append(sample)
    #samples = samples[:32] # test small sample
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
                                 filepath='weights/best_weights.hdf5',
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
        
