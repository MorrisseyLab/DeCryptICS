#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
import math
import cv2, os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import nasnet
from DNN.autocuration.datagen import pull_centered_img, pull_crypt_from_cnt

class run_generator(Sequence):
   def __init__(self, imgpath, dat, cnts, batch_size, tilesize=512, crsize=384):
      self.imgpath = imgpath
      self.data = dat
      self.cnts = cnts
      self.batch_size = batch_size
      self.tilesize = tilesize
      self.crsize = crsize
      self.dat_shape = self.data.shape[0]

   def __len__(self):
      return math.floor(self.dat_shape / self.batch_size)

   def __getitem__(self, idx):
      start = idx * self.batch_size
      end   = min(start + self.batch_size, self.dat_shape)
      x_batch, y_batch = self.read_batch(start, end)
      return x_batch, y_batch

   def read_batch(self, start, end):
      x_batcha = []
      x_batchb = []
      for ids in range(start, end):
         ii = ids
         # get sample
         got_good_c = False
         while not got_good_c:            
            try:
               img_tile, img_cr = self.get_images(self.data[ii,:])
               got_good_c = True
            except:
               print("get_image_pair failed!")
               print(ids)
               print(self.data[ii,:])
               ii = np.random.randint(self.dat_shape)
               got_good_c = False
         x_batcha.append(img_tile)
         x_batchb.append(img_cr)
      x_batcha = np.array(x_batcha)
      x_batchb = np.array(x_batchb)
      return [x_batcha, x_batchb]
         
   def get_images(self, data):
      XY = data[:2]
      ind_m = int(data[2])
            
      ## get hi-res crypt and bounding box, and low res tile
      img_cr = pull_crypt_from_cnt(XY, self.imgpath, self.cnts[ind_m], self.crsize, dwnsample_lvl=0)
#      img_cr = pull_centered_img(XY, imgpath, self.crsize, dwnsample_lvl=0)
      img_tile = pull_centered_img(XY, self.imgpath, self.tilesize, dwnsample_lvl=1)

      img_cr = img_cr.astype(np.float32) / 255
      img_tile = img_tile.astype(np.float32) / 255
      return img_tile, img_cr

