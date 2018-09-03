#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:49:03 2018

@author: doran
"""

from GUI_ChooseROI_class import getROI_svs
from MiscFunctions import getROI_img_vips, plot_img, read_cnt_text_file
import os, glob
import cv2
import numpy as np
import matplotlib.pylab as plt
from pympler.tracker import SummaryTracker

# First need to create training data. This will be of the form:
#   -- correctly sized tiles of pre-analysed .svs images;
#   -- correctly sized tiles of binary mask created by contours.
# Each sample is then a tile of the original image and its binary mask.
# To do this: use a version of the segement tile image function (with
# no overlapping?) to get indices of tiles. To retain equal sizes, throw
# away the final column and row which will be the wrong size.
# Generate a binary mask for the entire .svs image then tile in same way
# using the same indices. Then save image tile and mask tile together.

def find_tiles(dwnsamp_mask, num_x, num_y, maskthresh):
   kept_tiles = []
   for xx in range(num_x):
      for yy in range(num_y):
         tile_mask = dwnsamp_mask[(tb+yy*tile_y):(tb+tile_y+yy*tile_y), (lb+xx*tile_x):(lb+tile_x+xx*tile_x)]
         if (np.sum(tile_mask) > maskthresh):
            kept_tiles.append((xx, yy))
         del tile_mask
   return kept_tiles

if __name__=="__main__":
   base_path = "/home/doran/Work/images/"
   training_dat = []
   folder_im = []
   folder_cnt = []

   # mPAS .svs slides:
#   batch_ID1 = "/mPAS_WIMM/"
#   folder_im += [base_path + batch_ID1 +"/raw_images/"]
#   folder_cnt += [base_path + batch_ID1 + "/Analysed_slides/Analysed_"]
#   training_dat += [base_path + batch_ID1 +"/raw_images/575845.svs"]
#   batch_ID1b = "/mPAS_subset_test/"
#   folder_im += [base_path + batch_ID1b +"/raw_images/"]
#   folder_cnt += [base_path + batch_ID1b + "/Analysed_slides/Analysed_"]
#   training_dat += [base_path + batch_ID1b +"/raw_images/575833.svs"]

#   # MAOA .svs slides:
#   batch_ID2 = "/MAOA_slides/"
#   folder_im += [base_path + batch_ID2 +"/raw_images/"]
#   folder_cnt += [base_path + batch_ID2 + "/Analysed_slides/Analysed_"]
#   training_dat += [base_path + batch_ID2 +"/raw_images/540796.svs"]
#   batch_ID2b = "/MAOA_March2018/"
#   folder_im += [base_path + batch_ID2b +"/raw_images/"]
#   folder_cnt += [base_path + batch_ID2b + "/Analysed_slides/Analysed_"]
#   training_dat += [base_path + batch_ID2b +"/raw_images/586574.svs"]

   # mPAS jpg test images
   #batch_ID3 = "/mPAS_Clone_test_images/"
   #folder_im += [base_path + batch_ID3 +"/raw_images/"]
   #folder_cnt += [base_path + batch_ID3 + "/Analysed_slides/"]
   #training_dat += glob.glob(folder_im3 + "*.jpg")

#   # KDM6A .svs slides:
#   batch_ID4 = "/KDM6A_March2018/"
#   folder_im += [base_path + batch_ID4 +"/raw_images/"]
#   folder_cnt += [base_path + batch_ID4 + "/Analysed_slides/Analysed_"]
#   training_dat += [base_path + batch_ID4 +"/raw_images/642708.svs"]

#   # Possible slides for NONO, STAG, etc.
#   batch_ID5 = "/NONO_March2018/"
#   folder_im += [base_path + batch_ID5 +"/raw_images/"]
#   folder_cnt += [base_path + batch_ID5 + "/Analysed_slides/Analysed_"]
#   training_dat += [base_path + batch_ID5 +"/raw_images/627193.svs", base_path + batch_ID5 +"/raw_images/627187.svs"]

#   batch_ID6 = "/STAG_March2018/"
#   folder_im += [base_path + batch_ID6 +"/raw_images/"]
#   folder_cnt += [base_path + batch_ID6 + "/Analysed_slides/Analysed_"]
#   training_dat += [base_path + batch_ID6 +"/raw_images/601160.svs", base_path + batch_ID6 +"/raw_images/601177.svs"]

   ###############################################################

   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/input/"
   imgout = dnnfolder + "/train/"
   maskout = dnnfolder + "/train_masks/"
   try:
       os.mkdir(imgout)
   except:
       pass
   try:
       os.mkdir(maskout)
   except:
       pass

   # set chosen data and hierarchy
   #training_dat = mpas_training_dat_svs + maoa_training_dat_svs + maoa_training_dat_svsb + 
   #folder_cnt = folder_cnt4
   #folder_im = folder_im4
   n_slides = len(training_dat) # shift = 0
   tile_x = 4096
   tile_y = 4096
   maskthresh = 255 * 1000 # throw away masks with fewer than 1000 white pixels

   #tracker = SummaryTracker()

   for i in range(n_slides):
       # Read on whole .svs slide
       filename = training_dat[i]
       if (filename[-4:]==".svs"):
           obj_svs  = getROI_svs(filename , get_roi_plot = False)
           img      = getROI_img_vips(filename, (0,0),  obj_svs.dims_slides[0])
           xshape = obj_svs.dims_slides[0][1] # cols
           yshape = obj_svs.dims_slides[0][0] # rows
           xshape_unzoom = obj_svs.dims_slides[1][1] # cols
           yshape_unzoom = obj_svs.dims_slides[1][0] # rows
       else:
           img = cv2.imread(filename, cv2.IMREAD_COLOR)
           xshape = img.shape[1] # cols
           yshape = img.shape[0] # rows
       
       lb = (xshape_unzoom % tile_x) // 2 # left buffer
       num_x = xshape_unzoom//tile_x
       tb = (yshape_unzoom % tile_y) // 2 # top buffer
       num_y = yshape_unzoom//tile_y
       scaleval = int(xshape/xshape_unzoom)

       # Read in slide contours
       im_number = filename[:-4] # remove extension
       im_number = im_number[len(folder_im[i]):] # remove full path
       contours = read_cnt_text_file(folder_cnt[i] + im_number + "/crypt_contours.txt")
       # Draw contours to make mask
       big_mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
       for j in range(len(contours)):
           cv2.drawContours(big_mask, [contours[j]], 0, 255, -1)
           
       # Downsample to create unzoomed img/mask 
       dwnsamp_mask = cv2.pyrDown(big_mask)
       dwnsamp_mask = cv2.pyrDown(dwnsamp_mask)
       
       kept_in

       # Downsample to create unzoomed img/mask
       dwnsamp_img = cv2.pyrDown(img)
       dwnsamp_img = cv2.pyrDown(dwnsamp_img)

       del img
       del big_mask
           
       # Tile image and mask; save output
       for xx in range(num_x):
           print("Doing x %d of %d" % (xx, num_x-1))
           for yy in range(num_y):
               outfile_img = imgout + "/img_" + im_number + "_x" + str(xx) + "_y" + str(yy) + ".png"
               outfile_mask = maskout + "/mask_" + im_number + "_x" + str(xx) + "_y" + str(yy) + ".png"
               tile_img = dwnsamp_img[(tb+yy*tile_y):(tb+tile_y+yy*tile_y), (lb+xx*tile_x):(lb+tile_x+xx*tile_x), :3]
               tile_mask = dwnsamp_mask[(tb+yy*tile_y):(tb+tile_y+yy*tile_y), (lb+xx*tile_x):(lb+tile_x+xx*tile_x)]
               if (np.sum(tile_mask) > maskthresh):
                   cv2.imwrite(outfile_img, tile_img)
                   cv2.imwrite(outfile_mask, tile_mask)
               del tile_img, tile_mask
       del dwnsamp_img
       del dwnsamp_mask    
       #tracker.print_diff()
       
