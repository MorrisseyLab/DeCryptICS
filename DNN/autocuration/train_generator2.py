#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
import math
import cv2, os
from keras.utils import Sequence
from keras.applications import nasnet
#import feather
import efficientnet.keras as efn
from DNN.autocuration.datagen import pull_centered_img, pull_crypt, create_nbr_stack, pull_crypt_from_cnt
from DNN.augmentation            import plot_img, randomHueSaturationValue,\
                                        ReproducerandomHueSaturationValue,\
                                        HorizontalFlip

class train_generator(Sequence):
   def __init__(self, img_size1, img_size2, data, batch_size, nn=30, nn_sampsize=200,
               shuffle=True, aug=False, dpath="./DNN/autocuration/data/"):
      self.data = data
      self.datapath = dpath
      self.batch_size = batch_size
      self.scr_size = img_size1
      self.smallsize = img_size2
      self.nn = nn
      self.nn_sampsize = nn_sampsize
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
      t3 = time.time()
      for ids in range(start, end):
         ii = ids
         got_good_c = False
         while not got_good_c:            
            try:
               img_out1, img_cr1, mask1, img_out2, img_cr2, mask2 = self.get_posneg_pair(self.data[ii,:])
               got_good_c = True
            except:
               print("get_posneg_pair failed!")
               print(ids)
               print(self.data[ii,:])
               ii = np.random.randint(self.pos_shape)
               got_good_c = False
         x_batcha.append(img_out1)
         x_batchb.append(img_cr1)
         y_batch.append(mask1)
         x_batcha.append(img_out2)
         x_batchb.append(img_cr2)
         y_batch.append(mask2)
      y_batch = np.array(y_batch)
      # shuffle order of batch      
      np.random.shuffle(self.shuffinds)
      y_batch = y_batch[self.shuffinds]
      x_batcha = np.array(x_batcha)[self.shuffinds,:,:,:]
      x_batchb = np.array(x_batchb)[self.shuffinds,:,:,:]
      t4 = time.time()
      time_taken = t4-t3
      print("Time taken for batch: %1.2f seconds" % (time_taken))
      return [x_batcha, x_batchb], y_batch
         
   def get_posneg_pair(self, data):
      XY = data[:2]
      imgpath = data[5]
      cntpath = data[6]
      ind_m = int(data[2])
      clone_bool = int(data[3])
      
      ## load the relevant slide data
      thisname = self.datapath + imgpath.split('/')[-1].split('.svs')[0]
      slide_data = np.load(thisname + '_data.npy')
      cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)
      
      ## decide on rotation
      ROTATE = False
      RT_m = 0
      
      ## get positive sample
      thiscnt = cnts[ind_m]
      print("Here1")
      img_out1 = create_nbr_stack(XY, imgpath, cnts, slide_data, ind_m, scr_size = self.scr_size, 
                                  nn = self.nn, sampsize = self.nn_sampsize, multicore=False)
      print("Here2")
      img_cr1 = pull_crypt_from_cnt(XY, imgpath, [thiscnt], self.smallsize, 
                                    ROTATE=ROTATE, RT_m=RT_m, dwnsample_lvl=0)
      mask1 = np.array([clone_bool], dtype=np.float32)

      ## get negative/random sample
      got_good_c = False
      newind = np.random.randint(low = 0, high=slide_data.shape[0])
      while not got_good_c:
         try:
            ind_m2 = int(slide_data[newind,2])
            thiscnt = cnts[ind_m2]
            XY2 = slide_data[newind,:2]
            print("Here3")
#            img_out2 = create_nbr_stack(XY2, imgpath, cnts, slide_data, ind_m2, scr_size = self.scr_size, nn = self.nn, sampsize = self.nn_sampsize)
            img_top2 = pull_crypt_from_cnt(XY2, imgpath, thiscnt, self.scr_size, 
                                          ROTATE=ROTATE, RT_m=RT_m, dwnsample_lvl=1)                       
            img_out2 = np.dstack([img_top2, img_out1[:,:,3:]])
            print("Here4")
            img_cr2 = pull_crypt_from_cnt(XY2, imgpath, thiscnt, self.smallsize, 
                                          ROTATE=ROTATE, RT_m=RT_m, dwnsample_lvl=0)
            mask2 = np.array([slide_data[newind,3]], dtype=np.float32)
            got_good_c = True
         except:
            print("get_negative_stack failed!")
            print(imgpath)
            print(ind_m2)
            newind = np.random.randint(low = 0, high=slide_data.shape[0])
            got_good_c = False

      img_out1  = img_out1.astype(np.float32) / 255
      img_cr1  = img_cr1.astype(np.float32) / 255
      img_out2  = img_out2.astype(np.float32) / 255
      img_cr2  = img_cr2.astype(np.float32) / 255
      return img_out1, img_cr1, mask1, img_out2, img_cr2, mask2

   def on_epoch_end(self):
      np.random.shuffle(self.data)

class validation_generator(Sequence):
   def __init__(self, img_size1, img_size2, val_dat, batch_size, nn=30, nn_sampsize=200, 
               shuffle=True, aug=False, dpath="./DNN/autocuration/data/"):
      self.validation_data = val_dat
      self.datapath = dpath
      self.batch_size = batch_size
      self.scr_size = img_size1
      self.smallsize = img_size2
      self.nn = nn
      self.nn_sampsize = nn_sampsize
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
               print("get_image_pair failed!")
               print(ids)
               print(self.validation_data[ii,:])
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
      
      ## load the relevant slide data
      thisname = self.datapath + imgpath.split('/')[-1].split('.svs')[0]
      slide_data = np.load(thisname + '_data.npy')
      cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)
      
      ## decide on rotation
      ROTATE = False
      RT_m = 0
      
      ## get positive sample
      thiscnt = cnts[ind_m]
      img_out1 = create_nbr_stack(XY, imgpath, cnts, slide_data, ind_m, scr_size = self.scr_size, 
                                  nn = self.nn, sampsize = self.nn_sampsize, multicore=False)
      img_cr1 = pull_crypt_from_cnt(XY, imgpath, [thiscnt], self.smallsize, 
                                    ROTATE=ROTATE, RT_m=RT_m, dwnsample_lvl=0)
      mask1 = np.array([clone_bool], dtype=np.float32)

      img_out1  = img_out1.astype(np.float32) / 255
      img_cr1  = img_cr1.astype(np.float32) / 255
      return img_out1, img_cr1, mask1

