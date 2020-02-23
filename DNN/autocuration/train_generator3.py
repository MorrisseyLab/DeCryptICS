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
from DNN.autocuration.datagen import pull_centered_img_keeporigincoords, get_bounding_box
from DNN.augmentation         import plot_img, randomHueSaturationValue,\
                                     ReproducerandomHueSaturationValue, HorizontalFlip

class train_generator(Sequence):
   def __init__(self, img_size, data, batch_size,
               shuffle=True, aug=False, dpath="./DNN/autocuration/data/"):
      self.data = data
      self.datapath = dpath
      self.batch_size = batch_size
      self.tilesize = img_size
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
               img1, bbox1, mask1, img2, bbox2, mask2 = self.get_image_and_bbox(self.data[ii,:])
               got_good_c = True
            except:
               print("get_img_bbox failed!")
               print(ids)
               print(self.data[ii,:])
               ii = np.random.randint(self.pos_shape)
               got_good_c = False
         x_batcha.append(img1)
         x_batchb.append(bbox1)
         y_batch.append(mask1)
         x_batcha.append(img2)
         x_batchb.append(bbox2)
         y_batch.append(mask2)
      y_batch = np.array(y_batch)
      # shuffle order of batch      
      np.random.shuffle(self.shuffinds)
      y_batch = y_batch[self.shuffinds]
      x_batcha = np.array(x_batcha)[self.shuffinds,:,:,:]
      x_batchb = np.array(x_batchb)[self.shuffinds,:,:]
      return [x_batcha, x_batchb], y_batch
         
   def get_image_and_bbox(self, data):
      XY = data[:2]
      imgpath = data[5]
      cntpath = data[6]
      ind_m = int(data[2])
      clone_bool = int(data[3])
      
      ## load the relevant slide data
      thisname = self.datapath + imgpath.split('/')[-1].split('.svs')[0]
      slide_data = np.load(thisname + '_data.npy')
      cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)
      
      dwnsamp = 1
      img, xy, pd = pull_centered_img_keeporigincoords(XY, imgpath, self.tilesize, dwnsample_lvl=dwnsamp)
      bbox = get_bounding_box(cnts[ind_m], imgpath, xy, pd, dwnsample_lvl=dwnsamp)
      # recast as (xmin, ymin, xmax, ymax) as fractions of the image size
      bbox_fc1 = np.array([bbox[0]/float(img.shape[1]),
                            bbox[1]/float(img.shape[0]),
                            (bbox[0] + bbox[2])/float(img.shape[1]),
                            (bbox[1] + bbox[3])/float(img.shape[0])])
      mask1 = np.array([clone_bool], dtype=np.float32)

      ## get negative/random sample
      got_good_c = False
      newind = np.random.randint(low = 0, high=slide_data.shape[0])
      while not got_good_c:
         try:
            ind_m2 = int(slide_data[newind,2])
            XY2 = slide_data[newind,:2]

            img2, xy2, pd2 = pull_centered_img_keeporigincoords(XY2, imgpath, self.tilesize, dwnsample_lvl=dwnsamp)
            bbox2 = get_bounding_box(cnts[ind_m2], imgpath, xy2, pd2, dwnsample_lvl=dwnsamp)
            # recast as (xmin, ymin, xmax, ymax) as fractions of the image size
            bbox_fc2 = np.array([bbox2[0]/float(img2.shape[1]),
                                  bbox2[1]/float(img2.shape[0]),
                                  (bbox2[0] + bbox2[2])/float(img2.shape[1]),
                                  (bbox2[1] + bbox2[3])/float(img2.shape[0])])
            mask2 = np.array([slide_data[newind,3]], dtype=np.float32)
            got_good_c = True
         except:
            print("get_negative failed!")
            print(imgpath)
            print(ind_m2)
            newind = np.random.randint(low = 0, high=slide_data.shape[0])
            got_good_c = False

      img = img.astype(np.float32) / 255
      bbox_fc1 = bbox_fc1.astype(np.float32)
      img2 = img2.astype(np.float32) / 255
      bbox_fc2 = bbox_fc2.astype(np.float32)
      return img, bbox_fc1, mask1, img2, bbox_fc2, mask2

   def on_epoch_end(self):
      np.random.shuffle(self.data)

class validation_generator(Sequence):
   def __init__(self, img_size, val_dat, batch_size,
                shuffle=True, aug=False, dpath="./DNN/autocuration/data/"):
      self.validation_data = val_dat
      self.datapath = dpath
      self.batch_size = batch_size
      self.tilesize = img_size
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
               img, bbox, mask = self.get_image_and_bbox(self.validation_data[ii,:])
               got_good_c = True
            except:
               print("get_image_pair failed!")
               print(ids)
               print(self.validation_data[ii,:])
               ii = np.random.randint(self.val_shape)
               got_good_c = False
         x_batcha.append(img)
         x_batchb.append(bbox)
         y_batch.append(mask)
      y_batch = np.array(y_batch)
      x_batcha = np.array(x_batcha)
      x_batchb = np.array(x_batchb)
      return [x_batcha, x_batchb], y_batch
         
   def get_image_and_bbox(self, data):
      XY = data[:2]
      imgpath = data[5]
      cntpath = data[6]
      ind_m = int(data[2])
      clone_bool = int(data[3])
      
      ## load the relevant slide data
      thisname = self.datapath + imgpath.split('/')[-1].split('.svs')[0]
      slide_data = np.load(thisname + '_data.npy')
      cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)
      
      dwnsamp = 1
      img, xy, pd = pull_centered_img_keeporigincoords(XY, imgpath, self.tilesize, dwnsample_lvl=dwnsamp)
      bbox = get_bounding_box(cnts[ind_m], imgpath, xy, pd, dwnsample_lvl=dwnsamp)
      # recast as (xmin, ymin, xmax, ymax) as fractions of the image size
      bbox_fc1 = np.array([bbox[0]/float(img.shape[1]),
                            bbox[1]/float(img.shape[0]),
                            (bbox[0] + bbox[2])/float(img.shape[1]),
                            (bbox[1] + bbox[3])/float(img.shape[0])])
      mask1 = np.array([clone_bool], dtype=np.float32)

      img = img.astype(np.float32) / 255
      bbox_fc1 = bbox_fc1.astype(np.float32)
      return img, bbox_fc1, mask1

