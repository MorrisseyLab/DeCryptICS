#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
import cv2, os
from keras.utils import Sequence
from keras.applications import nasnet
from DNN.autocuration.datagen import create_nbr_stack, pull_crypt_from_cnt

class run_generator(Sequence):
   def __init__(self, imgpath, dat, cnts, batch_size, img_size1=50, img_size2=384, nn=30, nn_sampsize=200):
      ## create a generator for checking crypts of a single slide
      self.imgpath = imgpath
      self.data = dat
      self.cnts = cnts
      self.batch_size = batch_size
      self.scr_size = img_size1
      self.smallsize = img_size2
      self.dat_shape = self.data.shape[0]
      self.nn = nn
      self.nn_sampsize = nn_sampsize

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
      ind_m = int(data[2])
      clone_bool = int(data[3])

      ## get positive sample
      thiscnt = self.cnts[ind_m]
      print("Here")
      img_out1 = create_nbr_stack(XY, self.imgpath, self.cnts, self.data, ind_m, scr_size = self.scr_size, 
                                  nn = self.nn, sampsize = self.nn_sampsize, multicore=True)
      print("Here2")
      img_cr1 = pull_crypt_from_cnt(XY, self.imgpath, [thiscnt], self.smallsize, ROTATE=False, RT_m=0, dwnsample_lvl=0)
      print("Here3")
      img_out1 = img_out1.astype(np.float32) / 255
      img_cr1  = img_cr1.astype(np.float32) / 255
      return img_out1, img_cr1


