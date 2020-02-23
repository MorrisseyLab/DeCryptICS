#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
import cv2, os
from keras.utils import Sequence
from keras.applications import nasnet
#import feather
import efficientnet.keras as efn
from DNN.autocuration.datagen import pull_centered_img, pull_crypt
from DNN.augmentation            import plot_img, randomHueSaturationValue,\
                                        ReproducerandomHueSaturationValue,\
                                        HorizontalFlip

class train_generator(Sequence):
   def __init__(self, img_size1, img_size2, pos_dat, neg_dat, 
                batch_size, shuffle=True, aug=False):
      self.positive_data = pos_dat
      self.negative_data = neg_dat
      self.batch_size = batch_size
      self.tilesize = img_size1
      self.smallsize = img_size2
      self.shuffle = shuffle
      self.aug = aug
      self.pos_shape = self.positive_data.shape[0]
      self.neg_shape = self.negative_data.shape[0]
      self.shuffinds = np.asarray(range(2*batch_size), dtype=np.uint32)
      if self.shuffle:
         np.random.shuffle(self.positive_data)         
      if self.aug:
         pass    

   def __len__(self):
      return math.floor(self.pos_shape / self.batch_size)

   def __getitem__(self, idx):
      start = idx * self.batch_size
      end   = min(start + self.batch_size, self.pos_shape)
      x_batch, y_batch = self.read_batch(start, end)
      return x_batch, y_batch

   def read_batch(self, start, end):
      x_batcha = []
      x_batchb = []
      y_batch = []
      for ids in range(start, end):
         ii = ids
         # get true clone sample
         got_good_c = False
         while not got_good_c:            
            try:
               img_out, img_cr, mask = self.get_image_pair(self.positive_data[ii,:])
               got_good_c = True
            except:
               ii = np.random.randint(self.pos_shape)
               got_good_c = False
         x_batcha.append(img_out)
         x_batchb.append(img_cr)
         y_batch.append(mask)
         # get random WT sample
         got_good = False
         while not got_good:
            ind = np.random.randint(self.neg_shape)
            try:
               img_out, img_cr, mask = self.get_image_pair(self.negative_data[ind,:])
               got_good = True
            except:
               got_good = False
         x_batcha.append(img_out)
         x_batchb.append(img_cr)
         y_batch.append(mask)
      y_batch = np.array(y_batch)
      # shuffle order of batch      
      np.random.shuffle(self.shuffinds)
      y_batch = y_batch[self.shuffinds]
      x_batcha = np.array(x_batcha)[self.shuffinds,:,:,:]
      x_batchb = np.array(x_batchb)[self.shuffinds,:,:,:]
      return [x_batcha, x_batchb], y_batch
         
   def get_image_pair(self, data):
      XY = data[:2]
      imgpath = data[5]
      cntpath = data[6]
      ind_m = int(data[2])
      clone_bool = int(data[3])
      
      # decide on rotation
      ROTATE = False
      RT_m = 0
      #RT_m = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
      
      img_out = pull_centered_img(XY, imgpath, self.tilesize, ROTATE=ROTATE, 
                                  RT_m=RT_m, dwnsample_lvl=1)
      img_cr = pull_crypt(XY, imgpath, cntpath, ind_m, self.smallsize,
                          ROTATE=ROTATE, RT_m=RT_m, dwnsample_lvl=0)
      mask = np.array([clone_bool], dtype=np.float32)
                          
      ## perform any augmentation
      if np.random.random() < 0.25:
         hue_shift = np.random.uniform(-5, 5)
         sat_shift = np.random.uniform(-5, 5)
         val_shift = np.random.uniform(-15, 15)
         img_out = ReproducerandomHueSaturationValue(img_out, hue_shift, sat_shift, val_shift)
         img_cr = ReproducerandomHueSaturationValue(img_cr, hue_shift, sat_shift, val_shift)

      if np.random.random() < 0.5:
         img_out = HorizontalFlip(img_out)
         img_cr = HorizontalFlip(img_cr)
         
      ## convert to float32 space (done in efn.preprocess_input?)
      img_out  = img_out.astype(np.float32) / 255
      img_cr  = img_cr.astype(np.float32) / 255
      return img_out, img_cr, mask

   def on_epoch_end(self):
      np.random.shuffle(self.positive_data)

class validation_generator(Sequence):
   def __init__(self, img_size1, img_size2, val_dat, 
                batch_size, shuffle=True, aug=False):
      self.validation_data = val_dat
      self.batch_size = batch_size
      self.tilesize = img_size1
      self.smallsize = img_size2
      self.val_shape = self.validation_data.shape[0]

   def __len__(self):
      return math.floor(self.val_shape / self.batch_size)

   def __getitem__(self, idx):
      start = idx * self.batch_size
      end   = min(start + self.batch_size, self.val_shape)
      x_batch, y_batch = self.read_batch(start, end)
      return x_batch, y_batch

   def read_batch(self, start, end):
      x_batcha = []
      x_batchb = []
      y_batch = []
      for ids in range(start, end):
         ii = ids
         # get sample
         got_good_c = False
         while not got_good_c:            
            try:
               img_out, img_cr, mask = self.get_image_pair(self.validation_data[ii,:])
               got_good_c = True
            except:
               ii = np.random.randint(self.val_shape)
               got_good_c = False
         x_batcha.append(img_out)
         x_batchb.append(img_cr)
         y_batch.append(mask)
      y_batch = np.array(y_batch)
      x_batcha = np.array(x_batcha)
      x_batchb = np.array(x_batchb)
      return [x_batcha, x_batchb], y_batch
         
   def get_image_pair(self, data):
      XY = data[:2]
      imgpath = data[5]
      cntpath = data[6]
      ind_m = int(data[2])
      clone_bool = int(data[3])
      
      # decide on rotation
      ROTATE = False
      RT_m = 0
      #RT_m = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
      
      img_out = pull_centered_img(XY, imgpath, self.tilesize, ROTATE=ROTATE, 
                                  RT_m=RT_m, dwnsample_lvl=1)
      img_cr = pull_crypt(XY, imgpath, cntpath, ind_m, self.smallsize,
                          ROTATE=ROTATE, RT_m=RT_m, dwnsample_lvl=0)
      mask = np.array([clone_bool], dtype=np.float32)
                          
      ## perform any augmentation
      if np.random.random() < 0.25:
         hue_shift = np.random.uniform(-5, 5)
         sat_shift = np.random.uniform(-5, 5)
         val_shift = np.random.uniform(-15, 15)
         img_out = ReproducerandomHueSaturationValue(img_out, hue_shift, sat_shift, val_shift)
         img_cr = ReproducerandomHueSaturationValue(img_cr, hue_shift, sat_shift, val_shift)

      if np.random.random() < 0.5:
         img_out = HorizontalFlip(img_out)
         img_cr = HorizontalFlip(img_cr)
         
      ## convert to float32 space (done in efn.preprocess_input?)
      img_out  = img_out.astype(np.float32) / 255
      img_cr  = img_cr.astype(np.float32) / 255
      return img_out, img_cr, mask

