#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 05:02:40 2021

@author: edward
"""

import cv2
import os
import numpy as np
import pandas as pd
import openslide as osl
from MiscFunctions import getROI_img, rescale_contours, add_offset, mkdir_p, read_cnt_text_file, plot_img
import pyvips
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

## Correct width and height for cropping so that it never overshoots the 
## size of the image
def correct_xy(max_vals, xy_vals, wh_vals):
    if xy_vals[0]<0: xy_vals[0] = 0
    if xy_vals[1]<0: xy_vals[1] = 0    

    final_x  = xy_vals[0] + wh_vals[0]
    final_y  = xy_vals[1] + wh_vals[1]

    if final_x > max_vals[0] : xy_vals[0] = max_vals[0] - wh_vals[0]
    if final_y > max_vals[1] : xy_vals[1] = max_vals[1] - wh_vals[1]
    
    return xy_vals

def build_image_tile(list_to_plot, nrow = 1):   
    num_images = len(list_to_plot)
    num_cols   = int(num_images/nrow)
    if num_images%nrow != 0:
        raise(UserWarning, "If more than one row make sure there are enough images!")
    if isinstance(list_to_plot, tuple) == 0: 
        vis = list_to_plot
    else:
        last_val = num_cols 
        vis      = np.concatenate(list_to_plot[0:last_val], axis=1)
        for row_i in range(1, nrow):
            first_val = last_val
            last_val  = first_val + num_cols
            vis_aux   = np.concatenate(list_to_plot[first_val:last_val], axis=1)
            vis       = np.concatenate((vis, vis_aux), axis=0)
    return vis

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def pad_image(img, size):
   if (img.shape[0]>=img.shape[1]):
      img = image_resize(img, height = size)
      if (img.shape[1]!=size):
         pad = np.zeros((size, (size-img.shape[1])//2, 3), dtype=np.uint8)
         img = np.hstack([pad, img, pad])
         img = cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
   else:
      img = image_resize(img, width = size)
      if (img.shape[0]!=size):
         pad = np.zeros(((size-img.shape[0])//2, size, 3), dtype=np.uint8)
         img = np.vstack([pad, img, pad])
         img = cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
   return img

# https://github.com/libvips/pyvips/issues/100    
class svs_file:
    def __init__(self, file_name, tilesize, um_per_pixel):
        slide = osl.OpenSlide(file_name)
        self.mpp0 = float(slide.properties['openslide.mpp-x'])
        self.tilesize = tilesize
        self.dwnsmpl_lvl, self.tile_size_read, self.resize_param, self.scale_abs = self.get_size_match_info(slide, um_per_pixel, tilesize)
        self.maxdims = slide.level_dimensions[self.dwnsmpl_lvl]
        self.scale   = slide.level_downsamples[self.dwnsmpl_lvl]
        self.full_dims = slide.level_dimensions[0]
        slide.close()
        self.vips_img = pyvips.Image.openslideload(file_name, level = self.dwnsmpl_lvl)
        self.file_name = file_name
        
    def fetch_param(self):
        return self.dwnsmpl_lvl, self.scale_abs, self.scale, self.maxdims
    
    # xy0 must be with respect to level used for reading

    def fetch_subimg(self, xy0):
        wh            = (self.tile_size_read, self.tile_size_read)
        max_vals      = (self.vips_img.width, self.vips_img.height)
        xy0 = correct_xy(max_vals, xy0, wh) ## Correct over_under shoot

        area          = self.vips_img.crop(xy0[0], xy0[1], wh[0], wh[1])
        area          = area.resize(self.resize_param)
        new_img       = np.ndarray(buffer=area.write_to_memory(),
                            dtype=np.uint8,
                            shape=[area.height, area.width, area.bands])
        new_img       = cv2.cvtColor(new_img[:,:,0:3], cv2.COLOR_RGB2BGR)
        return new_img        

    def fetch_subimg_prop(self, pxy0):
        xy0 = [int(np.rint(self.vips_img.width*pxy0[0])), int(np.rint(self.vips_img.height*pxy0[1]))]
        return self.fetch_subimg(xy0)

    def fetch_subimg_rect(self, xy0, wh):
        max_vals      = (self.vips_img.width, self.vips_img.height)
        xy0 = correct_xy(max_vals, xy0, wh) ## Correct over_under shoot

        area          = self.vips_img.crop(xy0[0], xy0[1], wh[0], wh[1])
        area          = area.resize(self.resize_param)
        new_img       = np.ndarray(buffer=area.write_to_memory(),
                            dtype=np.uint8,
                            shape=[area.height, area.width, area.bands])
        new_img       = cv2.cvtColor(new_img[:,:,0:3], cv2.COLOR_RGB2BGR)
        return new_img
        
    def fetch_subimg_rect_prop(self, pxy0, pwh):
        xy0 = [int(np.rint(self.vips_img.width*pxy0[0])), int(np.rint(self.vips_img.height*pxy0[1]))]
        wh = [int(np.rint(self.vips_img.width*pwh[0])), int(np.rint(self.vips_img.height*pwh[1]))]
        return self.fetch_subimg_rect(xy0, wh)
        
    def get_size_match_info(self, slide, mpp_fin, tilesize):
        mpp0           = float(slide.properties['openslide.mpp-x'])
        dsls           = slide.level_downsamples
        mpp_levels     = np.array(dsls)*mpp0
        zoom_level     = np.where(mpp_fin - mpp_levels>0)[0][-1]
        tile_size_read = np.ceil(tilesize*mpp_fin/mpp_levels[zoom_level])
        resize_param   = tilesize/tile_size_read
        scale_abs      = mpp_fin/mpp0
        return zoom_level, tile_size_read, resize_param, scale_abs

    def fetch_thumbnail(self, size = 1024):
        thmb     = pyvips.Image.thumbnail(self.file_name, size)
        new_img  = np.ndarray(buffer=thmb.write_to_memory(),
                                    dtype=np.uint8,
                                    shape=[thmb.height, thmb.width, thmb.bands])
        img_out  = cv2.cvtColor(new_img[:,:,0:3], cv2.COLOR_RGB2BGR)
#        pad_size = img_out.shape[0]
        raw_size = img_out.shape[:2]
        img_pad = np.zeros([size, size, 3], dtype = np.uint8)
        img_pad[0:img_out.shape[0],0:img_out.shape[1],:] = img_out
        scale_thmb = self.full_dims[0]/img_out.shape[1]
        return img_pad, scale_thmb, raw_size
     
    def load_events(self, output_folder):
         self.raw_data = pd.read_csv(output_folder + '/raw_crypt_output.csv')
         self.x1_prop = np.asarray(self.raw_data['bbox_x1']) / self.full_dims[0]
         self.y1_prop = np.asarray(self.raw_data['bbox_y1']) / self.full_dims[1]
         x2_prop      = np.asarray(self.raw_data['bbox_x2']) / self.full_dims[0]
         y2_prop      = np.asarray(self.raw_data['bbox_y2']) / self.full_dims[1]
         self.w_prop = x2_prop-self.x1_prop
         self.h_prop = y2_prop-self.y1_prop
         self.buf_x = 25 / self.full_dims[0]
         self.buf_y = 25 / self.full_dims[1]
    
    def initialize_event_tile_params(self, wh = 80, bfs = 2):
         self.wh = wh
         self.buf_edge_lr = np.zeros((self.wh, bfs, 3), dtype=np.uint8)
         self.buf_edge_tb = np.zeros((bfs, self.wh+2*bfs, 3), dtype=np.uint8)
         self.blacksquare = np.zeros((self.wh+2*bfs, self.wh+2*bfs, 3), dtype=np.uint8)
    
    def set_event_type(self, event):
        self.event_type = event
    
    def build_event_tiles(self, threshold, max_display = 56, take_as_bottom = 8):
          # update/store threshold
         if self.event_type=='p_clone': self.clone_threshold = threshold
         if self.event_type=='p_partial': self.partial_threshold = threshold
         if self.event_type=='p_fufi': self.fufi_threshold = threshold
         if self.event_type=='p_crypt': self.crypt_threshold = threshold
         idxs = np.where(self.raw_data[self.event_type] >= threshold)[0]
         if len(idxs)>0:
            probs = np.asarray(self.raw_data[self.event_type].iloc[idxs])
            allordered = np.argsort(probs)[::-1]
            num_in = len(probs)
            if num_in <= take_as_bottom:
               to_plot = idxs[allordered]
               to_plot_ps = np.around(probs[allordered], 2)
            else:
               unsampled_n = np.minimum(5, num_in)
               allordered = np.argsort(probs)[::-1]
               bottom_n = idxs[allordered[-unsampled_n:]]
               top_n = idxs[allordered[:-unsampled_n]]
               pick_n = np.minimum(len(top_n), max_display)
               sampled = np.sort(np.random.choice(range(len(top_n)), size=pick_n, replace=False))
               to_plot = np.hstack([top_n[np.sort(sampled)], bottom_n])
               to_plot_ps = np.around(np.hstack([probs[allordered[:-unsampled_n]][np.sort(sampled)], probs[allordered[-unsampled_n:]]]), 2)
            imli_br = []
            imli_cr = []
            nrows_cr = []
            text_col = (50,50,200)
            per_row = int(np.ceil(np.sqrt(len(to_plot))))
            for i,c in enumerate(to_plot):
               pxy0 = (np.maximum(0, self.x1_prop[c] - self.buf_x), 
                       np.maximum(0, self.y1_prop[c] - self.buf_y))
               pwh = (np.minimum(1, self.w_prop[c] + 2*self.buf_x),
                      np.minimum(1, self.h_prop[c] + 2*self.buf_y))
               img = self.fetch_subimg_rect_prop(pxy0, pwh)               
               img = pad_image(img, self.wh)
               cv2.putText(img, str(to_plot_ps[i]),(img.shape[1]-40,img.shape[0]-10), 0, 0.5, text_col)
               img = np.hstack([self.buf_edge_lr, img, self.buf_edge_lr])
               img = np.vstack([self.buf_edge_tb, img, self.buf_edge_tb])
               imli_br.append(img)
            if (len(imli_br) <= per_row): nrows = 1
            elif (len(imli_br) % per_row == 0):
               nrows = len(imli_br) // per_row
            else:
               nrows = len(imli_br) // per_row + 1
               leftover = nrows * per_row - len(imli_br)
               for k in range(leftover): imli_br.append(self.blacksquare)
            outimg = build_image_tile(tuple(imli_br), nrow = nrows)
         else: outimg = self.blacksquare
         return outimg
       
    def plot_sampled_events(self):
         self.min_sld_val = 0.05
         self.max_sld_val = 1
         retval = ord('a')
         list_to_plot = self.build_event_tiles(self.max_sld_val/2.)
         fig, ax = plt.subplots()
         plt.subplots_adjust(bottom=0.25)
         tiles = plt.imshow(cv2.cvtColor(list_to_plot, cv2.COLOR_RGB2BGR))
         slider_bkd_color = 'lightgoldenrodyellow'
         ax_thresh = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=slider_bkd_color)
         slider = Slider(ax_thresh, "Threshold", self.min_sld_val, self.max_sld_val, valinit=0.5, valstep=0.025, color="green")
         
         def update(val):
              threshold = slider.val
              list_to_plot = self.build_event_tiles(threshold)
              tiles.set_data(cv2.cvtColor(list_to_plot, cv2.COLOR_RGB2BGR))
              tiles.autoscale()
              fig.canvas.draw_idle()
         
         slider.on_changed(update)
         plt.title(self.event_type)
         plt.draw()
         plt.pause(1)
         while retval != '':
            retval = input("<Press Enter when happy with threshold (with this window in focus)>")
         plt.close(fig)
