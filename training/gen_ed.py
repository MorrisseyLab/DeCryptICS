#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:20:34 2021

@author: edward
"""
#!/usr/bin/env python3
import math
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tensorflow.keras.utils import Sequence
from MiscFunctions import plot_img
from training.augmentation import random_affine, random_flip, random_perspective
from training.read_svs_class import svs_file_w_labels
from model_set_parameter_dicts import set_params_ed
params_gen_ed = set_params_ed()


def contour_Area(cnt):
    if len(cnt) == 1: return 1        
    return(cv2.contourArea(cnt))

def contour_mean_Area(cnt, img):
    # Get mean colour of object
    roi           = cv2.boundingRect(cnt)
    Start_ij_ROI  = np.array(roi)[0:2] # get x,y of bounding box
    cnt_roi       = cnt - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill     = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill, [cnt_roi], 0, 255, -1) ## Get mask
    mean_col_ii   = cv2.mean(img_ROI, mask_fill)[0]/255.
    return(mean_col_ii)

def bbox_y1_x1_y2_x2(cnti):
    bb_cv2 = cv2.boundingRect(cnti)
    # x,y,w,h -> y1, x1, y2, x2
    return np.array([bb_cv2[1], bb_cv2[0], bb_cv2[1] + bb_cv2[3], bb_cv2[0]+ bb_cv2[2]])
    
def box_overlap(A,B):
    return (A[0] < B[2] and A[2] > B[0] and A[1] < B[3] and A[3] > B[1])
      
def get_bbox_prob_from_masks(masks, max_crypts = 400, max_gen_crypts = 40): # increase this?
#   img = x_batch[batch_i] * train_datagen.norm_std + train_datagen.norm_mean
#   imdown = cv2.pyrDown(np.around(img*255).astype(np.uint8))

   mask_cp     = masks[:, :, 0]
   crypt_mask  = mask_cp.astype(np.uint8) * 255
   contours, hierarchy = cv2.findContours(crypt_mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
   bboxes = np.array([bbox_y1_x1_y2_x2(cnti) for cnti in contours])

   dummy_mask = np.ones(crypt_mask.shape, dtype=np.uint8)*255
   for i in range(bboxes.shape[0]):
      dummy_mask = cv2.rectangle(dummy_mask, (bboxes[i,1], bboxes[i,0]), (bboxes[i,3], bboxes[i,2]), -255, -1)

   ## this is dependent on mpp~1.1, needs generalising
   st_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
   dummy_mask = cv2.morphologyEx(dummy_mask.copy(), cv2.MORPH_ERODE, st_5, iterations = 2)
   sz = crypt_mask.shape[0]
   x_chop = np.random.choice(range(sz), size=30)
   y_chop = np.random.choice(range(sz), size=30)
   dummy_mask[:,x_chop] = 0
   dummy_mask[y_chop,:] = 0
   dummy_mask = cv2.morphologyEx(dummy_mask.copy(), cv2.MORPH_ERODE, st_5, iterations = 1)

   gen_cnts, gen_hier = cv2.findContours(dummy_mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
   gen_bboxes = np.array([bbox_y1_x1_y2_x2(cnti) for cnti in gen_cnts])
   genb_areas = np.array([(box[2]-box[0])*(box[3]-box[1]) for box in gen_bboxes])
   # cut large boxes and reduce number for intersect comparison
   gen_bboxes = gen_bboxes[np.where(genb_areas<2000)[0],:]
   gen_bboxes = gen_bboxes[np.random.choice(range(len(gen_bboxes)), 3*max_gen_crypts),:]

   ## throw those bboxes that intersect real bboxes?
   intersects = []
   for i in range(len(gen_bboxes)):
      intersects.append(np.all([not box_overlap(gen_bboxes[i], box_j) for box_j in bboxes]))
   gen_bboxes = gen_bboxes[np.where(np.array(intersects))[0]]

#   for i in range(len(gen_bboxes)):
#      imdown = cv2.rectangle(imdown.copy(), (gen_bboxes[i,1], gen_bboxes[i,0]), (gen_bboxes[i,3], gen_bboxes[i,2]), (0,0,255), -1)
#   for i in range(len(contours)):
#      imdown = cv2.drawContours(imdown.copy(), contours, i, (0,255,0), 1)
#      imdown = cv2.rectangle(imdown.copy(), (bboxes[i,1], bboxes[i,0]), (bboxes[i,3], bboxes[i,2]), (255,0,0), 1)
#   plot_img(imdown)

   ## bbox x, y, w, h  
   # Find which channel is use, pull that channel
   stuff_mask = masks[:,:,1:]
   # make prob
   p_i = []
   for dim_i in range(stuff_mask.shape[2]):
     p_feat_j = np.array([1*(255*contour_mean_Area(cnti, stuff_mask[:,:,dim_i])>0.5) for cnti in contours])
     p_i.append(p_feat_j)
   p_i = np.stack(p_i, axis = 1)

   # assume square tiles and make coords [0 , 1]
   bboxes      = bboxes/crypt_mask.shape[0]
   gen_bboxes  = gen_bboxes/crypt_mask.shape[0]

   null_bbox = np.array([[0, 0, -1, -1]])
   null_p_i = np.zeros((1, stuff_mask.shape[2]))

   n_crypts = bboxes.shape[0]
   n_gen_crypts = np.minimum(gen_bboxes.shape[0], max_gen_crypts)
   if n_crypts == 0:
     bboxes   = null_bbox 
     p_i      = null_p_i
     n_crypts = 1
     
   # Shuffle box order, pad or chop
   # If too many, random subsample
   if n_crypts >= max_crypts:
      indx_subsmp = np.random.choice(n_crypts, max_crypts, replace=False)
      bboxes = bboxes[indx_subsmp,:]
      p_i = p_i[indx_subsmp,:]
   elif (n_crypts + n_gen_crypts) >= max_crypts:
      indx_subsmp = np.random.choice(range(gen_bboxes.shape[0]), max_crypts-n_crypts)
      vec_extend = np.zeros(max_crypts-n_crypts, dtype = np.uint8)
      gen_bboxes = gen_bboxes[indx_subsmp, :]
      p_ig = null_p_i[vec_extend,:]
      # stack them 
      bboxes      = np.vstack([bboxes, gen_bboxes])
      p_i         = np.vstack([p_i, p_ig])
      indx_shuffle = np.random.choice(max_crypts, max_crypts, replace=False)        
      bboxes = bboxes[indx_shuffle,:]
      p_i = p_i[indx_shuffle,:]
   else:  
      # pad with null and generated bounding boxes
      gen_bboxes = gen_bboxes[np.random.choice(range(gen_bboxes.shape[0]), n_gen_crypts),:]
      vec_extend = np.zeros(n_gen_crypts, dtype = np.uint8)
      p_ig = null_p_i[vec_extend,:]
      indx_subsmp = np.zeros(max_crypts-n_crypts-n_gen_crypts, dtype = np.uint8)
      bboxes0      = null_bbox[indx_subsmp,:]
      p_i0         = null_p_i[indx_subsmp,:]
      # stack them 
      bboxes      = np.vstack([bboxes, gen_bboxes, bboxes0])
      p_i         = np.vstack([p_i, p_ig, p_i0])
      indx_shuffle = np.random.choice(max_crypts, max_crypts, replace=False)        
      bboxes = bboxes[indx_shuffle,:]
      p_i = p_i[indx_shuffle,:]
     
   p_i = p_i.astype(np.float32)
   return mask_cp, bboxes, p_i



def batch_process_mask(batch_masks, max_crypts = 100):
    b_mask_cp = []
    b_bboxes  = []
    b_p_i     = []
    for batch_i in range(batch_masks.shape[0]):     
        mask_cp, bboxes, p_i = get_bbox_prob_from_masks(batch_masks[batch_i], max_crypts = max_crypts)
        b_mask_cp.append(mask_cp)
        b_bboxes.append(bboxes)
        b_p_i.append(p_i)            
    return np.array(b_mask_cp), np.array(b_bboxes), np.array(b_p_i)


class DataGen_curt(Sequence):
   def __init__(self, params, file_curated, n_steps, n_clone_img_factor, n_part_img_mult):
      imgpaths    = list(file_curated["file_name"]) # params['imgpaths']
      curated_inf = list(file_curated["slide_crtd"]) # params['curated_files']
      self.tilesize = params['tilesize_train'] # output image size
      self.um_per_pixel = params['umpp'] # target to downsample to
      self.num_bbox = params['num_bbox'] # target to downsample to
      self.open_all_svs(imgpaths, curated_inf) # Open all svs file in dictionary
      
      ## multiply images  with respect to clone and partial numbers
      self.nclones = [ii.clone_num for ii in self.all_svs_opened.values()]
      self.npartials = [ii.partial_num for ii in self.all_svs_opened.values()]      
      renormed_imgpaths = []
      for i in range(len(self.nclones)):
         for j in range(max(1, int(self.nclones[i] // n_clone_img_factor))):
            renormed_imgpaths.append(imgpaths[i])
      for i in range(len(self.npartials)):
         for j in range(self.npartials[i] * n_part_img_mult):
            renormed_imgpaths.append(imgpaths[i])
      while len(renormed_imgpaths)<params['batch_size']:
         renormed_imgpaths = renormed_imgpaths + renormed_imgpaths
      self.imgpaths = renormed_imgpaths
      
      self.dat_length = len(self.imgpaths)
      self.batch_size = params['batch_size']
      self.run_length = math.ceil(self.dat_length / self.batch_size)
      self.dilate_masks = params['dilate_masks'] # draw and dilate to enlarge contours and bboxes
      self.shuffle = params['shuffle']
      self.aug = params['aug']
      self.just_clone = params['just_clone']
      self.normalize = params['normalize']
      self.norm_mean = np.array([0.485, 0.456, 0.406])
      self.norm_std = np.array([0.229, 0.224, 0.225])
      self.stride_bool = params['stride_bool']
      self.n_steps = n_steps
      if self.shuffle:
         np.random.shuffle(self.imgpaths)         
      if params['reset_binaries']:
         self.save_new_binaries()
      self.cpfr_frac = params['cpfr_frac'] / np.sum(params['cpfr_frac']) # fraction of samples that contain clones, partials, fufis, random tiles
      
   def __len__(self):
      return self.n_steps * self.run_length

   def __getitem__(self, idx):
      start = (idx % self.run_length) * self.batch_size
      end   = min(start + self.batch_size, self.dat_length)
      x_batch, y_batch = self.read_batch(start, end)
      self.pathused    = [self.imgpaths[ids] for ids in range(start, end)]
      mask_crypt, bboxes, p_i = batch_process_mask(y_batch, max_crypts=self.num_bbox)
      return [x_batch, bboxes], [mask_crypt, p_i[:,:,0], p_i[:,:,1], p_i[:,:,2]] # in order: clone, partial, fufi

   def read_batch(self, start, end):
      x_batch = []
      y_batch = []
      for ids in range(start, end):
         ii = ids
         # get sample
         got_good_c = False
         while not got_good_c:            
            try:
               img, mask = self.get_images(self.imgpaths[ii])
               got_good_c = True
            except:
               print("get_images failed!")
               print(self.imgpaths[ii])
               ii = np.random.randint(len(self.imgpaths))
               got_good_c = False
         x_batch.append(img)
         y_batch.append(mask)
      y_batch = np.array(y_batch)
      x_batch = np.array(x_batch)
      return x_batch, y_batch
         
   def open_all_svs(self, img_paths_all, curated_files):
       self.all_svs_opened = {}
       for path_i, curated_i in zip(img_paths_all, curated_files):
          self.all_svs_opened[path_i] = svs_file_w_labels(path_i, self.tilesize, self.um_per_pixel, curated_i)
  
   def get_images(self, imgpath):
      # Pick random clone, crypt or fufi
      u01 = np.random.uniform()
      if u01<self.cpfr_frac[0]: img, mask = self.all_svs_opened[imgpath].fetch_clone(prop_displ = 0.45)
      elif u01<np.sum(self.cpfr_frac[0:2]): img, mask = self.all_svs_opened[imgpath].fetch_partial(prop_displ = 0.45)
      elif u01<np.sum(self.cpfr_frac[0:3]): img, mask = self.all_svs_opened[imgpath].fetch_fufi(prop_displ = 0.45)
      else: img, mask = self.all_svs_opened[imgpath].fetch_rndmtile() 

      if self.dilate_masks==True:
        st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        n_dil = int(5/self.um_per_pixel) # if mpp is one or less than 1        
        # dilate if desired
        for i in range(mask.shape[2]):
            mask[:,:,i] = cv2.morphologyEx(mask[:,:,i].copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)

      if self.aug==True:
          composition = A.Compose([
              A.HorizontalFlip(), A.VerticalFlip(), A.Rotate(border_mode = cv2.BORDER_CONSTANT),
              A.OneOf(
              [
                  A.ElasticTransform(alpha = 1000, sigma = 30,
                                     alpha_affine = 30, border_mode = cv2.BORDER_CONSTANT, p=1),
                  A.GridDistortion(border_mode = cv2.BORDER_CONSTANT, p = 1),
#                  A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5,
#                                      border_mode = cv2.BORDER_CONSTANT, p = 1),
              ],  p=0.7),
              A.CLAHE(p=0.3),
              A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=12, val_shift_limit=12, p=0.6),
              A.RandomBrightnessContrast(p=0.6),
              A.Posterize(p=0.2, num_bits=4),
              A.OneOf(
              [
                  A.JpegCompression(p=1),
                  A.MedianBlur(p=1),
                  A.Blur(p=1),
                  A.GlassBlur(p=1, max_delta = 2, sigma=0.4),
                  A.IAASharpen(p=1)
              ],  p=0.3)
          ],  p = 1)
          transformed = composition(image=img, mask=mask)
          img, mask = transformed['image'], transformed['mask']
      mask_list = [mask[:,:,ii] for ii in range(mask.shape[2])]
      
      if self.stride_bool:
          mask_list = [cv2.pyrDown(mask_ii.copy()) for mask_ii in mask_list]
      
      mask_list = [cv2.threshold(mask_ii, 120, 255, cv2.THRESH_BINARY)[1] for mask_ii in mask_list]
      
      ## convert to floating point space, normalize and mask non-used clones
      img       = img.astype(np.float32) / 255
      mask_list = [mask_ii.astype(np.float32) / 255 for mask_ii in mask_list]
      if self.normalize:
         img = (img - self.norm_mean) / self.norm_std
      
      return img, np.stack(mask_list, axis = 2)

   def on_epoch_end(self):
       if self.shuffle:
           np.random.shuffle(self.imgpaths)


class CloneGen_curt(Sequence):
   def __init__(self, params, file_curated):     
      imgpaths    = list(file_curated["file_name"]) # params['imgpaths']
      curated_inf = list(file_curated["slide_crtd"]) # params['curated_files']
      self.tilesize = params['tilesize_train'] # output image size
      self.um_per_pixel = params['umpp'] # target to downsample to
      self.num_bbox = params['num_bbox'] # target to downsample to
      self.open_all_svs(imgpaths, curated_inf) # Open all svs file in dictionary
      self.nclones = [ii.clone_num + ii.partial_num for ii in self.all_svs_opened.values()]

      renormed_imgpaths = []
      which_clone = []
      for i in range(len(self.nclones)):
         if self.nclones[i]>0:
            for j in range(self.nclones[i]):
               renormed_imgpaths.append(imgpaths[i])
               which_clone.append(j)
               
      self.imgpaths = renormed_imgpaths
      self.which_clone = which_clone
      self.dat_length = len(self.imgpaths)
      self.batch_size = params['batch_size']
      self.run_length = math.ceil(self.dat_length / self.batch_size)
      self.dilate_masks = params['dilate_masks'] # draw and dilate to enlarge contours and bboxes
      self.just_clone = params['just_clone']
      self.normalize = params['normalize']
      self.norm_mean = np.array([0.485, 0.456, 0.406])
      self.norm_std = np.array([0.229, 0.224, 0.225])
      self.stride_bool = params['stride_bool']
      if params['reset_binaries']:
         self.save_new_binaries()
      
   def __len__(self):
      return self.run_length

   def __getitem__(self, idx):
      start = idx * self.batch_size
      end   = min(start + self.batch_size, self.dat_length)
      x_batch, y_batch = self.read_batch(start, end)
      self.pathused    = [self.imgpaths[ids] for ids in range(start, end)]
      mask_crypt, bboxes, p_i = batch_process_mask(y_batch, max_crypts=self.num_bbox)
      return [x_batch, bboxes], [mask_crypt, p_i[:,:,0], p_i[:,:,1], p_i[:,:,2]]

   def read_batch(self, start, end):
      x_batch = []
      y_batch = []
      for ids in range(start, end):
         ii = ids
         # get sample
         got_good_c = False
         while not got_good_c:            
            try:
               img, mask = self.get_images(self.imgpaths[ii], self.which_clone[ii])
               got_good_c = True
            except:
               print("get_images failed!")
               print(self.imgpaths[ii])
               ii = np.random.randint(len(self.imgpaths))
               got_good_c = False
         x_batch.append(img)
         y_batch.append(mask)
      y_batch = np.array(y_batch)
      x_batch = np.array(x_batch)
      return x_batch, y_batch
         
   def open_all_svs(self, img_paths_all, curated_files):
       self.all_svs_opened = {}
       for path_i, curated_i in zip(img_paths_all, curated_files):
          self.all_svs_opened[path_i] = svs_file_w_labels(path_i, self.tilesize, self.um_per_pixel, curated_i)
  
   def get_images(self, imgpath, clone_ind):
      # Pick the clone
      img, mask = self.all_svs_opened[imgpath].fetch_clone(clone_ind)                
         
      if self.dilate_masks==True:
        st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        n_dil = int(5./self.um_per_pixel) # if mpp is one or less than 1
        for i in range(mask.shape[2]):
            mask[:,:,i] = cv2.morphologyEx(mask[:,:,i].copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)
#        mask[:,:,0] = cv2.morphologyEx(mask[:,:,0].copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)
     
      mask_list = [mask[:,:,ii] for ii in range(mask.shape[2])]
      
      if self.stride_bool:
          mask_list = [cv2.pyrDown(mask_ii.copy()) for mask_ii in mask_list]
          
      mask_list = [cv2.threshold(mask_ii, 120, 255, cv2.THRESH_BINARY)[1] for mask_ii in mask_list]
      ## convert to floating point space, normalize and mask non-used clones
      img       = img.astype(np.float32) / 255
      mask_list = [mask_ii.astype(np.float32) / 255 for mask_ii in mask_list]
      if self.normalize:
         img = (img - self.norm_mean) / self.norm_std
      
      return img, np.stack(mask_list, axis = 2)
      
class CloneFufiGen_curt(Sequence):
   def __init__(self, params, file_curated):     
      imgpaths    = list(file_curated["file_name"]) # params['imgpaths']
      curated_inf = list(file_curated["slide_crtd"]) # params['curated_files']
      self.tilesize = params['tilesize_train'] # output image size
      self.um_per_pixel = params['umpp'] # target to downsample to
      self.num_bbox = params['num_bbox'] # target to downsample to
      self.open_all_svs(imgpaths, curated_inf) # Open all svs file in dictionary
      self.nclones = [ii.clone_num + ii.partial_num for ii in self.all_svs_opened.values()]

      renormed_imgpaths = []
      which_clone = []
      for i in range(len(self.nclones)):
         if self.nclones[i]>0:
            for j in range(self.nclones[i]):
               renormed_imgpaths.append(imgpaths[i])
               which_clone.append(j)
               
      self.imgpaths = renormed_imgpaths
      self.which_clone = which_clone
      self.dat_length = len(self.imgpaths)
      self.batch_size = params['batch_size']
      self.run_length = math.ceil(self.dat_length / self.batch_size)
      self.dilate_masks = params['dilate_masks'] # draw and dilate to enlarge contours and bboxes
      self.just_clone = params['just_clone']
      self.normalize = params['normalize']
      self.norm_mean = np.array([0.485, 0.456, 0.406])
      self.norm_std = np.array([0.229, 0.224, 0.225])
      self.stride_bool = params['stride_bool']
      if params['reset_binaries']:
         self.save_new_binaries()
      
   def __len__(self):
      return self.run_length

   def __getitem__(self, idx):
      start = idx * self.batch_size
      end   = min(start + self.batch_size, self.dat_length)
      x_batch, y_batch = self.read_batch(start, end)
      self.pathused    = [self.imgpaths[ids] for ids in range(start, end)]
      mask_crypt, bboxes, p_i = batch_process_mask(y_batch, max_crypts=self.num_bbox)
      return [x_batch, bboxes], [mask_crypt, p_i[:,:,0], p_i[:,:,1], p_i[:,:,2]]

   def read_batch(self, start, end):
      x_batch = []
      y_batch = []
      for ids in range(start, end):
         ii = ids
         # get sample
         got_good_c = False
         while not got_good_c:            
            try:
               img, mask = self.get_images(self.imgpaths[ii], self.which_clone[ii])
               got_good_c = True
            except:
               print("get_images failed!")
               print(self.imgpaths[ii])
               ii = np.random.randint(len(self.imgpaths))
               got_good_c = False
         x_batch.append(img)
         y_batch.append(mask)
      y_batch = np.array(y_batch)
      x_batch = np.array(x_batch)
      return x_batch, y_batch
         
   def open_all_svs(self, img_paths_all, curated_files):
       self.all_svs_opened = {}
       for path_i, curated_i in zip(img_paths_all, curated_files):
          self.all_svs_opened[path_i] = svs_file_w_labels(path_i, self.tilesize, self.um_per_pixel, curated_i)
  
   def get_images(self, imgpath, clone_ind):
      # Pick the clone
      img, mask = self.all_svs_opened[imgpath].fetch_clone(clone_ind)                
         
      if self.dilate_masks==True:
        st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        n_dil = int(5./self.um_per_pixel) # if mpp is one or less than 1
        for i in range(mask.shape[2]):
            mask[:,:,i] = cv2.morphologyEx(mask[:,:,i].copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)
#        mask[:,:,0] = cv2.morphologyEx(mask[:,:,0].copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)
     
      mask_list = [mask[:,:,ii] for ii in range(mask.shape[2])]
      
      if self.stride_bool:
          mask_list = [cv2.pyrDown(mask_ii.copy()) for mask_ii in mask_list]
          
      mask_list = [cv2.threshold(mask_ii, 120, 255, cv2.THRESH_BINARY)[1] for mask_ii in mask_list]
      ## convert to floating point space, normalize and mask non-used clones
      img       = img.astype(np.float32) / 255
      mask_list = [mask_ii.astype(np.float32) / 255 for mask_ii in mask_list]
      if self.normalize:
         img = (img - self.norm_mean) / self.norm_std
      
      return img, np.stack(mask_list, axis = 2)
