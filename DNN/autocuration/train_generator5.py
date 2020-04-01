#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
import math
import cv2, os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import nasnet
from DNN.autocuration.datagen import pull_centered_img, pull_crypt_from_cnt
from DNN.augmentation         import plot_img, randomHueSaturationValue,\
                                     ReproducerandomHueSaturationValue, HorizontalFlip

class train_generator(Sequence):
   def __init__(self, tilesize, crsize, data, batch_size,
               shuffle=True, aug=False, dpath="./DNN/autocuration/data/"):
      self.data = data
      self.datapath = dpath
      self.batch_size = batch_size
      self.tilesize = tilesize
      self.crsize = crsize
      self.shuffle = shuffle
      self.aug = aug
      self.pos_shape = self.data.shape[0]
      self.shuffinds = np.asarray(range(2*batch_size), dtype=np.uint32)
      if self.shuffle:
         np.random.shuffle(self.data)         
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
         got_good_c = False
         while not got_good_c:            
            try:
               img_tile, img_cr, mask1, img_tile2, img_cr2, mask2 = self.get_images(self.data[ii,:])
               got_good_c = True
            except:
               print("get_img_bbox failed!")
               print(ids)
               print(self.data[ii,:])
               ii = np.random.randint(self.pos_shape)
               got_good_c = False
         x_batcha.append(img_tile)
         x_batchb.append(img_cr)
         y_batch.append(mask1)
         x_batcha.append(img_tile2)
         x_batchb.append(img_cr2)
         y_batch.append(mask2)
      y_batch = np.array(y_batch)
      # shuffle order of batch      
      np.random.shuffle(self.shuffinds)
      y_batch = y_batch[self.shuffinds]
      x_batcha = np.array(x_batcha)[self.shuffinds,:,:,:]
      x_batchb = np.array(x_batchb)[self.shuffinds,:,:,:]
      return [x_batcha, x_batchb], y_batch
         
   def get_images(self, data):
      XY = data[:2]
      imgpath = data[5]
      cntpath = data[6]
      ind_m = int(data[2])
      clone_bool = int(data[3])
      
      ## load the relevant slide data
      thisname = self.datapath + imgpath.split('/')[-1].split('.svs')[0]
      slide_data = np.load(thisname + '_data.npy')
      cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)
      
      ## get hi-res crypt and bounding box, and low res tile
      img_cr = pull_crypt_from_cnt(XY, imgpath, cnts[ind_m], self.crsize, dwnsample_lvl=0)
#      img_cr = pull_centered_img(XY, imgpath, self.crsize, dwnsample_lvl=0)
      img_tile = pull_centered_img(XY, imgpath, self.tilesize, dwnsample_lvl=1)
      mask1 = np.array([clone_bool], dtype=np.float32)

      ## get negative/random sample
      got_good_c = False
      newind = np.random.randint(low = 0, high=slide_data.shape[0])
      while not got_good_c:
         try:
            ind_m2 = int(slide_data[newind,2])
            XY2 = slide_data[newind,:2]
            
            img_cr2 = pull_crypt_from_cnt(XY2, imgpath, cnts[ind_m2], self.crsize, dwnsample_lvl=0)
#            img_cr2 = pull_centered_img(XY2, imgpath, self.crsize, dwnsample_lvl=0)
            img_tile2 = pull_centered_img(XY2, imgpath, self.tilesize, dwnsample_lvl=1)
            mask2 = np.array([slide_data[newind,3]], dtype=np.float32)
            
            got_good_c = True
         except:
            print("get_negative failed!")
            print(imgpath)
            print(ind_m2)
            newind = np.random.randint(low = 0, high=slide_data.shape[0])
            got_good_c = False

      img_cr = img_cr.astype(np.float32) / 255
      img_tile = img_tile.astype(np.float32) / 255
      img_cr2 = img_cr2.astype(np.float32) / 255
      img_tile2 = img_tile2.astype(np.float32) / 255
      return img_tile, img_cr, mask1, img_tile2, img_cr2, mask2

   def on_epoch_end(self):
      np.random.shuffle(self.data)

class validation_generator(Sequence):
   def __init__(self, tilesize, crsize, val_dat, batch_size,
                shuffle=True, aug=False, dpath="./DNN/autocuration/data/"):
      self.validation_data = val_dat
      self.datapath = dpath
      self.batch_size = batch_size
      self.tilesize = tilesize
      self.crsize = crsize
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
               img_tile, img_cr, mask = self.get_images(self.validation_data[ii,:])
               got_good_c = True
            except:
               print("get_image_pair failed!")
               print(ids)
               print(self.validation_data[ii,:])
               ii = np.random.randint(self.val_shape)
               got_good_c = False
         x_batcha.append(img_tile)
         x_batchb.append(img_cr)
         y_batch.append(mask)
      y_batch = np.array(y_batch)
      x_batcha = np.array(x_batcha)
      x_batchb = np.array(x_batchb)
      return [x_batcha, x_batchb], y_batch
         
   def get_images(self, data):
      XY = data[:2]
      imgpath = data[5]
      cntpath = data[6]
      ind_m = int(data[2])
      clone_bool = int(data[3])
      
      ## load the relevant slide data
      thisname = self.datapath + imgpath.split('/')[-1].split('.svs')[0]
      slide_data = np.load(thisname + '_data.npy')
      cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)
      
      ## get hi-res crypt and bounding box, and low res tile
      img_cr = pull_crypt_from_cnt(XY, imgpath, cnts[ind_m], self.crsize, dwnsample_lvl=0)
#      img_cr = pull_centered_img(XY, imgpath, self.crsize, dwnsample_lvl=0)
      img_tile = pull_centered_img(XY, imgpath, self.tilesize, dwnsample_lvl=1)
      mask1 = np.array([clone_bool], dtype=np.float32)

      img_cr = img_cr.astype(np.float32) / 255
      img_tile = img_tile.astype(np.float32) / 255
      return img_tile, img_cr, mask1

