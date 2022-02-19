#!/usr/bin/env python3
import math
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tensorflow.keras.utils import Sequence
from MiscFunctions import plot_img, contour_mean_Area, bbox_y1_x1_y2_x2, box_overlap
from training.read_svs_class import svs_file_w_labels
from model_set_parameter_dicts import set_params
params = set_params()
st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def generate_dummy_bboxes(crypt_mask, bboxes, max_gen_crypts, area_cap=4000):
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
   gen_bboxes = gen_bboxes[np.where(genb_areas<area_cap)[0],:]
   gen_bboxes = gen_bboxes[np.random.choice(range(len(gen_bboxes)), 3*max_gen_crypts),:]

   ## throw those bboxes that intersect real bboxes?
   intersects = []
   for i in range(len(gen_bboxes)):
      intersects.append(np.all([not box_overlap(gen_bboxes[i], box_j) for box_j in bboxes]))
   gen_bboxes = gen_bboxes[np.where(np.array(intersects))[0]]
   return gen_bboxes

def merge_close_contours(thismask, cnts, dummy_mask, n_dil):   
   thismask = cv2.morphologyEx(thismask.copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)
   cnts_dil, _ = cv2.findContours(thismask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
   bimg = dummy_mask.copy()
   for n, cd in enumerate(cnts_dil):
      cv2.drawContours(bimg, [cd], 0, (n+1,  0,   0),  -1)
   # find the contour numbers underneath orig contours
   meancols = [bimg[cnts[i][0,0,1], cnts[i][0,0,0]] for i in range(len(cnts))]
   # do several contours overlay same number (i.e. have been joined)?
   uniq, nums = np.unique(meancols, return_counts=True)
   new_cnt_inds = uniq[np.where(nums>1)[0]] - 1 # undo the 1-starting from drawing cnts
   orig_cnt_inds = [np.where(meancols==(ni+1))[0].astype(np.int32) for ni in new_cnt_inds]
   extra_cnts = [cnts_dil[ni] for ni in new_cnt_inds]
   return extra_cnts, orig_cnt_inds

def add_potential_fufi_cnts(cnts, mask):   
   n_dil = int(np.around(2.5/params['umpp']))
   extra_cnts, orig_cnt_inds = merge_close_contours(mask, cnts, np.zeros(mask.shape[:2], dtype=np.uint16), n_dil)
   n_new_cnts = len(extra_cnts)
   return extra_cnts, orig_cnt_inds
   
def shuffle_boxes(bboxes, max_crypts, n_crypts, n_gen_crypts):
   shufinds = np.array([], dtype=np.int32)
   if n_crypts > 0:
      to_pad = max_crypts - n_crypts - n_gen_crypts
      if to_pad<=0:
         shufinds = np.random.choice(n_crypts, n_crypts)
         bboxes = bboxes + np.around(np.random.normal(0, 2.5, size=(n_crypts,4))).astype(np.int32)
         bboxes = bboxes[shufinds,:]
         n_crypts = bboxes.shape[0]
      else:
         # stack noised bounding boxes up to maximum
         shufinds = np.hstack([range(n_crypts),np.random.choice(n_crypts, to_pad)])
         np.random.shuffle(shufinds)
         bboxes = bboxes[shufinds,:] + np.around(np.random.normal(0, 2.5, size=(n_crypts+to_pad,4))).astype(np.int32)
         n_crypts = bboxes.shape[0]
   return bboxes, n_crypts, shufinds

#def test_plot_bboxes(im, bboxes, p_i, ind):
#   img        = (255*(im*train_datagen.norm_std + train_datagen.norm_mean)).astype(np.uint8)
#   img_cnt    = img.copy()
#   bboxes     = bboxes
#   for j in range(bboxes.shape[0]):                
#       bbx_j    = 1024*bboxes[j, :]
#       y = int(bbx_j[0]); x =  int(bbx_j[1]); y2 = int(bbx_j[2]); x2 =  int(bbx_j[3])
#       col = (0,255,0) if p_i[j,ind] else (255,0,0) 
#       cv2.rectangle(img_cnt,(x,y),(x2,y2),tuple(col),2)
#   plot_img(img_cnt)

def get_bbox_prob_from_masks(masks, max_crypts = 400, max_gen_crypts = 200, crypt_class = False):
   mask_cp     = masks[:, :, 0]
   crypt_mask  = mask_cp.astype(np.uint8) * 255
   contours, hierarchy = cv2.findContours(crypt_mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

   # add true and false fufi boxes by dilation
   extra_cnts, orig_cnt_inds = add_potential_fufi_cnts(contours, crypt_mask)
   contours = contours + extra_cnts

   # Find which channel is use, pull that channel
   stuff_mask = masks[:,:,1:]   
   # make prob
   p_i = []
   for dim_i in range(stuff_mask.shape[2]):
      p_feat_j = np.array([1*(255*contour_mean_Area(cnti, stuff_mask[:,:,dim_i])>0.25) for cnti in contours])
      p_i.append(p_feat_j)
   if crypt_class is True:
      p_feat_j = np.array([1*(255*contour_mean_Area(cnti, masks[:,:,0])>0.25) for cnti in contours])
      p_i.append(p_feat_j)
   p_i = np.stack(p_i, axis = 1)   

   # make bounding boxes
   bboxes = np.array([bbox_y1_x1_y2_x2(cnti) for cnti in contours])
   gen_bboxes = generate_dummy_bboxes(crypt_mask, bboxes, max_gen_crypts, area_cap=4000)

   n_crypts = bboxes.shape[0]
   try:
      n_gen_crypts = np.random.choice(range(1, np.minimum(gen_bboxes.shape[0], max_gen_crypts)), 1)[0]
   except:
      n_gen_crypts = 0

   # shuffle
   bboxes, n_crypts, shufinds = shuffle_boxes(bboxes, max_crypts, n_crypts, n_gen_crypts)
   p_i = p_i[shufinds,:]

   # assume square tiles and make coords [0 , 1]
   bboxes      = bboxes/crypt_mask.shape[0]
   gen_bboxes  = gen_bboxes/crypt_mask.shape[0]
   bboxes[bboxes<0] = 0
   bboxes[bboxes>1] = 1

   null_bbox = np.array([[0, 0, -1, -1]])
   if crypt_class is True: null_p_i = np.zeros((1, stuff_mask.shape[2]+1))
   else: null_p_i = np.zeros((1, stuff_mask.shape[2]))
   
   if n_crypts == 0:
     bboxes   = null_bbox 
     p_i      = null_p_i
     n_crypts = 1
     
   if n_crypts >= max_crypts:
      indx_subsmp = np.random.choice(n_crypts, max_crypts, replace=False)
      bboxes = bboxes[indx_subsmp,:]
      p_i = p_i[indx_subsmp,:]
   elif (n_crypts + n_gen_crypts) >= max_crypts:
      indx_subsmp = np.random.choice(gen_bboxes.shape[0], max_crypts-n_crypts)
      vec_extend = np.zeros(max_crypts-n_crypts, dtype = np.uint8)
      gen_bboxes = gen_bboxes[indx_subsmp, :]
      p_ig = null_p_i[vec_extend,:]
      bboxes      = np.vstack([bboxes, gen_bboxes])
      p_i         = np.vstack([p_i, p_ig])
      indx_shuffle = np.random.choice(max_crypts, max_crypts, replace=False)        
      bboxes = bboxes[indx_shuffle,:]
      p_i = p_i[indx_shuffle,:]
   else:  
      # pad with null and generated bounding boxes
      gen_bboxes = gen_bboxes[np.random.choice(gen_bboxes.shape[0], n_gen_crypts),:]
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

def batch_process_mask(batch_masks, max_crypts = 400, crypt_class = False):
    b_mask_cp = []
    b_bboxes  = []
    b_p_i     = []
    for batch_i in range(batch_masks.shape[0]):     
        mask_cp, bboxes, p_i = get_bbox_prob_from_masks(batch_masks[batch_i], max_crypts = max_crypts, crypt_class = crypt_class)
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
      self.normalize = params['normalize']
      self.norm_mean = np.array([0.485, 0.456, 0.406])
      self.norm_std = np.array([0.229, 0.224, 0.225])
      self.stride_bool = params['stride_bool']
      self.crypt_class = params['crypt_class']
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
      mask_crypt, bboxes, p_i = batch_process_mask(y_batch, max_crypts=self.num_bbox, crypt_class=self.crypt_class)
      if self.crypt_class:
         # in order:                            clone, partial, fufi, crypt
         return [x_batch, bboxes], [mask_crypt, p_i[:,:,0], p_i[:,:,1], p_i[:,:,2], p_i[:,:,3]]
      else:
          # in order:                           clone, partial, fufi
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
              ],  p=0.5),
              A.CLAHE(p=0.2),
              A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=12, val_shift_limit=12, p=0.3),
              A.RandomBrightnessContrast(p=0.3),
              A.Posterize(p=0.1, num_bits=4),
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
   def __init__(self, params, file_curated, fufis=False):     
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
      self.normalize = params['normalize']
      self.norm_mean = np.array([0.485, 0.456, 0.406])
      self.norm_std = np.array([0.229, 0.224, 0.225])
      self.stride_bool = params['stride_bool']
      self.crypt_class = params['crypt_class']
      self.fufis = fufis # do we want to generate fufis?
      self.aug = params['aug']
      if params['reset_binaries']:
         self.save_new_binaries()
      
   def __len__(self):
      return self.run_length

   def __getitem__(self, idx):
      start = idx * self.batch_size
      end   = min(start + self.batch_size, self.dat_length)
      x_batch, y_batch = self.read_batch(start, end)
      self.pathused    = [self.imgpaths[ids] for ids in range(start, end)]
      mask_crypt, bboxes, p_i = batch_process_mask(y_batch, max_crypts=self.num_bbox, crypt_class=self.crypt_class)
      if self.crypt_class:
         # in order:                            clone, partial, fufi, crypt
         return [x_batch, bboxes], [mask_crypt, p_i[:,:,0], p_i[:,:,1], p_i[:,:,2], p_i[:,:,3]]
      else:
          # in order:                           clone, partial, fufi
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
      u01 = np.random.uniform()
      if u01<0.5 or not self.fufis:
         # Pick the clone
         img, mask = self.all_svs_opened[imgpath].fetch_clone(clone_ind)         
      else:
         # get a random fufi
         img, mask = self.all_svs_opened[imgpath].fetch_fufi(prop_displ = 0.45)
                     
      if self.dilate_masks==True:
        n_dil = int(5/self.um_per_pixel) # if mpp is one or less than 1
        for i in range(mask.shape[2]):
            mask[:,:,i] = cv2.morphologyEx(mask[:,:,i].copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)

      if self.aug==True:
          composition = A.Compose([
              A.HorizontalFlip(), A.VerticalFlip(), A.Rotate(border_mode = cv2.BORDER_CONSTANT),
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
