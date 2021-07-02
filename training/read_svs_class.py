#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:04:16 2021

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

def load_contours_preferentially(imgpath, cnt_type='crypt', datapath=None, save_new=False):
   pathend = imgpath.split('/')[-1] 
   imname = pathend.split('.')[0]
   fsl = os.path.abspath(imgpath[:-len(pathend)]) + '/Analysed_slides/Analysed_' + imname + '/'
   if datapath is None:
      datapath = fsl # use Analysed_imname path if no datapath specified
   ## prefer to load numpy binary 
   if os.path.exists(datapath + cnt_type + '_contours.npy') and not save_new: 
      cnts = np.load(datapath + cnt_type + '_contours.npy', allow_pickle=True)
      return cnts
   ## else load txt file and save binary for next time
   elif os.path.exists(fsl + cnt_type + '_contours.txt'):
      cnts = np.asarray(read_cnt_text_file(fsl + cnt_type + '_contours.txt'))
      np.save(datapath + cnt_type + '_contours.npy', cnts)
      return cnts
   else:
      print("Contours of type ` %s ` not found in any format for image %s" % (cnt_type, imgpath))
      return np.empty(0)

def load_data_preferentially(imgpath, datapath=None, save_new=False):
   imgpath = os.path.abspath(imgpath)
   pathend = imgpath.split('/')[-1] 
   imname = pathend.split('.')[0]
   fsl = os.path.abspath(imgpath[:-len(pathend)]) + '/Analysed_slides/Analysed_' + imname + '/'
   if datapath is None:
      datapath = fsl # use Analysed_imname path if no datapath specified
   ## prefer to load numpy binary 
   if os.path.exists(datapath + 'crypt_network_data.npy') and not save_new: 
      netdat = np.load(datapath + 'crypt_network_data.npy')
      netdat = np.hstack([netdat[:,:4], netdat[:,6:7]]) # <x><y><fufi><mutant><area>
   ## else load txt file and save binary for next time
   elif os.path.exists(fsl + 'crypt_network_data.txt'):
      netdat = np.asarray(pd.read_csv(fsl + '/crypt_network_data.txt', sep='\t'))
      netdat = np.hstack([netdat[:,:4], netdat[:,6:7]]) # <x><y><fufi><mutant><area>
      np.save(datapath + 'crypt_network_data.npy', netdat)
   else:
      print("Crypt network data not found for image %s" % imgpath)
      return np.empty(0), 'unknown'
   # once loaded, add mark info if known, infer from filepath if not
   try:
      set_info = pd.read_csv(imgpath[:-(len(imname)+4)] + '/slide_info.csv')
      set_info = set_info.astype({'Image ID':'str'})
      mark = set_info['mark'].iloc[np.where(set_info['Image ID']==imname)[0][0]]
   except:
      mark = 'unknown'
   return netdat, mark

   
# https://github.com/libvips/pyvips/issues/100    
class svs_file_w_labels:
    def __init__(self, file_name, tilesize, um_per_pixel, curated_info = None):
        slide = osl.OpenSlide(file_name)
        self.mpp0 = float(slide.properties['openslide.mpp-x'])
        self.tilesize = tilesize
        self.dwnsmpl_lvl, self.tile_size_read, self.resize_param, self.scale_abs = self.get_size_match_info(slide, um_per_pixel, tilesize)
        self.maxdims = slide.level_dimensions[self.dwnsmpl_lvl]
        self.scale   = slide.level_downsamples[self.dwnsmpl_lvl]
        slide.close()
        self.vips_img = pyvips.Image.openslideload(file_name, level = self.dwnsmpl_lvl)
        self.sld_dat, self.mark = load_data_preferentially(file_name)

        if curated_info is not None:
            self.sld_dat = np.array(pd.read_csv(curated_info, index_col = 0))
                
        self.clone_num = np.sum(self.sld_dat[:,3]==1)
        self.partial_num = np.sum(self.sld_dat[:,3]==2)
        self.crypt_num = np.sum(self.sld_dat[:,3]>-1)
        self.fufi_num  = np.sum(self.sld_dat[:,2]>0)

        num_crypts = len(self.sld_dat)
        index_all = np.arange(num_crypts).reshape([num_crypts,1])
        self.sld_dat = np.hstack([self.sld_dat, index_all])
        self.file_name = file_name
        
    def fetch_param(self):
        return self.dwnsmpl_lvl, self.scale_abs, self.scale, self.maxdims
    
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

    def fetch_crypt(self, crypt_num, contour = False, ret_info = False):
        # xy0 must be with respect to level used for reading
        xy0     = self.sld_dat[crypt_num,:2]/self.scale - self.tile_size_read/2
        xy0     = xy0.astype(int)
        img_out = self.fetch_subimg(xy0)
        # From here on use final scale (image has been resized)
        cnts_cr = load_contours_preferentially(self.file_name, cnt_type = 'crypt')
        cnts_cr = cnts_cr/self.scale_abs
        # rescale xy0
        xy0 = xy0*self.scale/self.scale_abs
        cnts_cr = [(cnti - xy0).astype(np.int32) for cnti in cnts_cr]
        if contour:
            img_out = cv2.drawContours(img_out, cnts_cr, -1, (255, 0, 0), 2)
            img_out = cv2.drawContours(img_out, [cnts_cr[crypt_num]], -1, (0, 0, 255), 2)
        if ret_info:
            scld_slide = self.sld_dat.copy()
            scld_slide[:,0:2] = scld_slide[:,0:2]/self.scale_abs
            #Find indexes of crypts and retrun crypts and info
            these_inds = np.where(np.logical_and(
               np.logical_and(scld_slide[:,0]>(xy0[0]),
                              scld_slide[:,0]<(xy0[0] + self.tilesize)),
               np.logical_and(scld_slide[:,1]>(xy0[1]),
                              scld_slide[:,1]<(xy0[1] + self.tilesize))))[0]    
            cnts_out = [cnts_cr[ii] for ii in these_inds]
            scld_slide[:,0:2] = scld_slide[:,0:2]-xy0
            return img_out, cnts_out, scld_slide[these_inds]        
        return img_out    

    def fetch_img_mask(self, crypt_num_or_xy, prop_displ = 0):        
        if type(crypt_num_or_xy) is not int:
            xy0 = crypt_num_or_xy
        else:           
            crypt_num = crypt_num_or_xy
            # xy0 must be with respect to level used for reading
            xy0     = self.sld_dat[crypt_num,:2]/self.scale - self.tile_size_read/2
        # print(crypt_num_or_xy)
        # random shift
        if prop_displ > 0:
            shiftsize = int(prop_displ * self.tile_size_read)    
            shift_by  = np.random.randint(low=-shiftsize, high=shiftsize, size=(2))
            xy0 = xy0 + shift_by

        xy0     = xy0.astype(int)
        img_out = self.fetch_subimg(xy0)
        # From here on use final scale (image has been resized)
        cnts_cr = load_contours_preferentially(self.file_name, cnt_type = 'crypt')
        cnts_cr = cnts_cr/self.scale_abs
        # rescale xy0
        xy0 = xy0*self.scale/self.scale_abs
        cnts_cr = [(cnti - xy0).astype(np.int32) for cnti in cnts_cr]
        
        # Crypt, fufi,         
        mask_out = np.zeros((self.tilesize, self.tilesize, 4), dtype = np.uint8)        
        
        scld_slide = self.sld_dat.copy()
        scld_slide[:,0:2] = scld_slide[:,0:2]/self.scale_abs
        #Find indexes of crypts and retrun crypts and info
        these_inds = np.where(np.logical_and(
           np.logical_and(scld_slide[:,0]>(xy0[0] - 0.1*self.tilesize),
                          scld_slide[:,0]<(xy0[0] + 1.1*self.tilesize)),
           np.logical_and(scld_slide[:,1]>(xy0[1] - 0.1*self.tilesize),
                          scld_slide[:,1]<(xy0[1] + 1.1*self.tilesize))))[0]    
                
        cnts_out = [cnts_cr[ii] for ii in these_inds]
        
        scld_slide = scld_slide[these_inds,:]
        scld_slide[:,0:2] = scld_slide[:,0:2]-xy0
        
        for ii in range(len(scld_slide)):
            cyrpt_i = scld_slide[ii]
            # Deleted crypt skip to next or draw crypt
            if cyrpt_i[3] == -1:
                continue
            else:
                # Crypt
                mask_out[:,:,0] = cv2.drawContours(mask_out[:,:,0].copy(), cnts_out, ii, 255, -1)
            ## Using mask order Clone, Partial, Fufi to align with the c_p_f_r fraction parameter
            # Clone
            if cyrpt_i[3] > 0:
                mask_out[:,:,1] = cv2.drawContours(mask_out[:,:,1].copy(), cnts_out, ii, 255, -1)
            # Partial
            if cyrpt_i[3] > 1:
                mask_out[:,:,2] = cv2.drawContours(mask_out[:,:,2].copy(), cnts_out, ii, 255, -1)            
            # Fufi
            if cyrpt_i[2] > 0:
                mask_out[:,:,3] = cv2.drawContours(mask_out[:,:,3].copy(), cnts_out, ii, 255, -1)
        return img_out, mask_out       

    def fetch_clone(self, clone_id = None, prop_displ = 0):
        indx_cln = np.where(self.sld_dat[:,3]>0)[0]
        num_vals = len(indx_cln)
        if num_vals>0:
            if clone_id is None:
                clone_id  = np.random.choice(num_vals)
            crypt_pick = int(indx_cln[clone_id])               
        else:
            crypt_pick = np.random.choice(len(self.sld_dat[:,3]))
        img_out, mask_out  = self.fetch_img_mask(crypt_pick, prop_displ)
        return img_out, mask_out

    def fetch_partial(self, clone_id = None, prop_displ = 0):
        indx_cln = np.where(self.sld_dat[:,3]>1)[0]
        num_vals = len(indx_cln)
        if num_vals>0:
            if clone_id is None:
                clone_id  = np.random.choice(num_vals)
            crypt_pick = int(indx_cln[clone_id])
            img_out, mask_out  = self.fetch_img_mask(crypt_pick, prop_displ)
        else:
            img_out, mask_out = self.fetch_clone(prop_displ=prop_displ)        
        return img_out, mask_out         

    def fetch_fufi(self, fufi_id = None, prop_displ = 0):
        indx_ffi = np.where(self.sld_dat[:,2]>0)[0]
        num_vals = len(indx_ffi)
        if num_vals>0:
            if fufi_id is None:
                fufi_id  = np.random.choice(num_vals)
            crypt_pick = int(indx_ffi[fufi_id])
        else:
            crypt_pick = np.random.choice(len(self.sld_dat[:,2]))
        img_out, mask_out  = self.fetch_img_mask(crypt_pick, prop_displ)
        return img_out, mask_out 

    def fetch_rndmtile(self):
        max_val = np.min((self.vips_img.width, self.vips_img.height))
        xx = np.random.randint(low=0, high=max_val-self.tile_size_read)
        yy = np.random.randint(low=0, high=max_val-self.tile_size_read)
        img_out, mask_out  = self.fetch_img_mask(np.array([xx,yy]), prop_displ = 0)
        return img_out, mask_out 

    def get_size_match_info(self, slide, mpp_fin, tilesize):
        mpp0           = float(slide.properties['openslide.mpp-x'])
        dsls           = slide.level_downsamples
        mpp_levels     = np.array(dsls)*mpp0
        zoom_level     = np.where(mpp_fin - mpp_levels>0)[0][-1]
        tile_size_read = np.ceil(tilesize*mpp_fin/mpp_levels[zoom_level])
        resize_param   = tilesize/tile_size_read
        scale_abs      = mpp_fin/mpp0
        return zoom_level, tile_size_read, resize_param, scale_abs

# TODO - generator stuff
# Use correct fufi contours

## Curation check 


# Get curated info
# Partials look at 
# folder_out    = "/home/edward/WIMM/Decryptics_train/decryptics_code/manual_curation_files/"
# already_curated = pd.read_csv(folder_out + 'curated_files_summary.txt', names = ["file_name", "slide_crtd"])

# for j in range(len(already_curated)):
#     if already_curated['slide_crtd'][j] != "cancel":
#         uu = np.array(pd.read_csv(already_curated['slide_crtd'][j], index_col = 0))
#         if np.any(uu[:,3]==-1):
#             print("gotcha")
#             break

        
# sld_i = svs_file_w_labels(already_curated['file_name'][j], 1024, 1, curated_info = already_curated['slide_crtd'][j])
# # # uu = pd.read_csv(already_curated['slide_crtd'][0])
# # # indx_rmv = np.where( uu[:,3]<0)[0]
# # indx_rmv = np.where( uu[:,2]>0)[0]
# # img_cnt = sld_i.fetch_crypt(indx_rmv[0], contour = True)
# # # plot_img(img, hold_plot=False, nameWindow="kk")

# img, mask_i = sld_i.fetch_rndmtile()
# # img = sld_i.fetch_clone()
# plot_img((img, mask_i[:,:,0:3]))


# Test removed crypt
# Test removed clone
# Test partial

# qq1 = load_contours_preferentially(already_curated['file_name'][j], cnt_type = 'crypt')
# qq2 = load_contours_preferentially(already_curated['file_name'][j], cnt_type = 'fufi')



# resolution thing 
# file_i = '/home/edward/WIMM/Decryptics_train/train/KM16/KM16S_446554.svs'
# sld_i           = svs_file2(file_i, 1024, 1)
# sld_dat_i       = sld_i.sld_dat
# cl_inds         = np.where(sld_dat_i[:,3]>0)[0]
# print([file_i, len(cl_inds), sld_i.mark])
# sld_i.fetch_param()
# img = sld_i.fetch_crypt(cl_inds[30], contour = True)
# plot_img(img)
# print(img.shape)



## Timgin tests
# # cv subsample # 54.2 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_subsample_cv = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) # 63 ms 
# plot_img(uu_subsample_cv)

# # resize # 54.8 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_resize = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) 
# plot_img(uu_resize)

# plot_img([uu_subsample_cv, uu_resize])

# # subsample # 53.4 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_subsample = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4)
# plot_img(uu_subsample)

# # shrink # 57 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_shrink = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) 
# plot_img(uu_shrink)

# # reduce # 55.2 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_reduce = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) 
# plot_img(uu_reduce)

# # no resize # 62.2 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_noresize = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) 
# plot_img(uu_noresize)





