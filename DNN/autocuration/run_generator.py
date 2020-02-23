#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
import cv2, os
from keras.utils import Sequence
from keras.applications import nasnet
from DNN.autocuration.datagen import pull_centered_img, pull_crypt

class run_generator(Sequence):
   def __init__(self, img_size1, img_size2, dat, batch_size):
      self.data = dat
      self.batch_size = batch_size
      self.tilesize = img_size1
      self.smallsize = img_size2
      self.dat_shape = self.data.shape[0]

   def __len__(self):
      leftover = int(math.ceil((self.dat_shape % self.batch_size) / self.batch_size))
      return math.floor(self.dat_shape / self.batch_size) + leftover

   def __getitem__(self, idx):
      start = idx * self.batch_size
      end   = min(start + self.batch_size, self.dat_shape)
      x_batch = self.read_batch(start, end)
      return x_batch

   def read_batch(self, start, end):
      x_batcha = []; x_batchb = []
      for ids in range(start, end):
         ii = ids
         img_out, img_cr = self.get_image_pair(self.data[ii,:])
         x_batcha.append(img_out)
         x_batchb.append(img_cr)
      x_batcha = np.array(x_batcha)
      x_batchb = np.array(x_batchb)
      return [x_batcha, x_batchb]
         
   def get_image_pair(self, data):
      XY = data[:2]
      imgpath = data[5]
      cntpath = data[6]
      ind_m = int(data[2])
      clone_bool = int(data[3])
      
      ROTATE = False
      RT_m = 0
      
      img_out = pull_centered_img(XY, imgpath, self.tilesize, ROTATE=ROTATE, 
                                  RT_m=RT_m, dwnsample_lvl=1)
      img_cr = pull_crypt(XY, imgpath, cntpath, ind_m, self.smallsize,
                          ROTATE=ROTATE, RT_m=RT_m, dwnsample_lvl=0)
                          
      img_out  = img_out.astype(np.float32) / 255
      img_cr  = img_cr.astype(np.float32) / 255
      return img_out, img_cr


