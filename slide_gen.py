#!/usr/bin/env python3
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from MiscFunctions import plot_img
from just_read_svs import svs_file
st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def bbox_y1_x1_y2_x2(cnti):
    bb_cv2 = cv2.boundingRect(cnti)
    # x,y,w,h -> y1, x1, y2, x2
    return np.array([bb_cv2[1], bb_cv2[0], bb_cv2[1] + bb_cv2[3], bb_cv2[0]+ bb_cv2[2]])

def breaks(max_x, dx, min_overlap):
    if max_x < dx:
        return(np.zeros(1))
    # x =n*dx - o(n-1)
#    l = n(dx-o) + dx    |      |o|   |o|    dx  \
    num_tile  = np.ceil((max_x-min_overlap)/(dx-min_overlap)) + 1
    overlap   = (num_tile*dx - max_x)/(num_tile-1)
    x_vals    = (dx-overlap)*np.arange(0, num_tile)
    x_vals[0] = 0
    return x_vals
    
# Use xy, thumbnail coords for boxes then turn to image prop when reading
class slide_tile_gen(Sequence):
   def __init__(self, file_name, mpp = 1.1, tile_size = 1024, max_len = 10, batch_size = 8, min_micrn_overlap = 80, norm = True, segmnt_and_tile = False):   
      self.mpp = mpp
      self.svs = svs_file(file_name, tile_size, mpp)
      self.img_thmb, self.scale_thmb, self.thmb_dims = self.svs.fetch_thumbnail()
      self.thmb_mpp = self.scale_thmb * self.svs.mpp0
      self.norm = norm
      self.indx_zero = 0
      self.min_micrn_overlap = min_micrn_overlap
      self.tile_size = tile_size 
      self.batch_size = batch_size
      self.norm_mean  = np.array([0.485, 0.456, 0.406])
      self.norm_std   = np.array([0.229, 0.224, 0.225])
      self.full_img_size_um = np.asarray(self.svs.full_dims) * self.svs.mpp0
      self.tilesize_um = self.tile_size * self.mpp
      self.thmb_dx, self.thmb_min_ovlp = self.get_thmb_dim_info()
      self.max_len = max_len
      # y1, x1, y2, x2
      self.bbox = [np.array([0, 0, self.thmb_dims[0], self.thmb_dims[1]])] # full image
      self.thmb_mask = []

      self.tile_array    = self.make_tiles()
      self.num_tiles     = self.tile_array.shape[0]
      self.total_batches = np.ceil(self.num_tiles/self.batch_size).astype(np.int32)
      self.batches_left  = self.total_batches

   def __len__(self):
      return np.min([self.max_len, self.batches_left])

   def set_start_indx(self, strt_indx):
      self.indx_zero = strt_indx
      self.batches_left = np.ceil((self.num_tiles-self.indx_zero)/self.batch_size).astype(np.int32)

   def __getitem__(self, idx):
      start = self.indx_zero  + idx * self.batch_size
      end   = min(start + self.batch_size, self.num_tiles)
      x_batch = self.read_batch(start, end)         
      return x_batch

   def get_thmb_dim_info(self):
       thmb_min_ovlp = self.thmb_dims[1]*self.min_micrn_overlap/self.full_img_size_um[0]
       thmb_dx       = self.thmb_dims[1]*self.tilesize_um/self.full_img_size_um[0]
       return thmb_dx, thmb_min_ovlp    

   def get_img_thmb_tilerect(self):
       img_box = self.img_thmb.copy()
       ij_vals_all = np.around(self.tile_array / self.thmb_mpp)
       for ii in range(self.num_tiles):
           if ii%self.batch_size==0:
               col = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
           ij_vals = (ij_vals_all[ii]).astype(np.int32)           
           cv2.rectangle(img_box,(ij_vals[1], ij_vals[0]),
                        (int(ij_vals[1] + self.thmb_dx),
                         int(ij_vals[0] + self.thmb_dx)),col,1)
       plot_img(img_box)

   def make_tiles(self):   
      xbreaks_um = breaks(self.full_img_size_um[0], self.tilesize_um, self.min_micrn_overlap)
      ybreaks_um = breaks(self.full_img_size_um[1], self.tilesize_um, self.min_micrn_overlap)
      out_vals = np.meshgrid(xbreaks_um, ybreaks_um, sparse=False, indexing='ij')
      ij_vals_all_um = np.array([out_vals[1].flatten(), out_vals[0].flatten()]).T
      return ij_vals_all_um       

   def read_batch(self, start, end):
      x_batch  = [] 
      ij_vals_input = self.tile_array / self.mpp
      max_xy = np.asarray(self.full_img_size_um / (self.mpp))
      for ids in range(start, end):
         img = self.get_image(ij_vals_input[ids,::-1]/max_xy)
         x_batch.append(img)
      x_batch = np.array(x_batch)
      return x_batch
      
   def get_image(self, p_xy0):
       # Pick random clone, crypt or fufi
       img = self.svs.fetch_subimg_prop(p_xy0)                                   
       ## convert to floating point space, normalize and mask non-used clones
       img = img.astype(np.float32) / 255
       if self.norm:
          img = (img - self.norm_mean) / self.norm_std      
       return img    


