#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:49:03 2018

@author: doran
"""

from GUI_ChooseROI_class import getROI_svs
from MiscFunctions import getROI_img_osl, plot_img, read_cnt_text_file
import os, glob
import cv2
import numpy as np
import matplotlib.pylab as plt
from pympler.tracker import SummaryTracker
from DNN.DNN_slice_data import get_tile_indices

# First need to create training data. This will be of the form:
#   -- correctly sized tiles of pre-analysed .svs images;
#   -- correctly sized tiles of binary mask created by contours.
# Each sample is then a tile of the original image and its binary mask.
# To do this: use a version of the segement tile image function (with
# no overlapping?) to get indices of tiles. To retain equal sizes, throw
# away the final column and row which will be the wrong size.
# Generate a binary mask for the entire .svs image then tile in same way
# using the same indices. Then save image tile and mask tile together.

#def find_tiles(dwnsamp_mask, num_x, num_y, maskthresh, shape_dims):
#   kept_tiles = []
#   tile_x = shape_dims[0]
#   tile_y = shape_dims[1]
#   lb = shape_dims[2]
#   tb = shape_dims[3]
#   for xx in range(num_x):
#      for yy in range(num_y):
#         tile_mask = dwnsamp_mask[(tb+yy*tile_y):(tb+(yy+1)*tile_y), (lb+xx*tile_x):(lb+(xx+1)*tile_x)]
#         if (np.sum(tile_mask) > maskthresh):
#            kept_tiles.append((xx, yy))
#         del tile_mask
#   return kept_tiles

#def save_tiles(source_img, tiles, imnumber, imgout_path, name, shape_dims):
#   if not (name=="img" or name=="mask" or name=="premask"): return 0
#   tile_x = shape_dims[0]
#   tile_y = shape_dims[1]
#   lb = shape_dims[2]
#   tb = shape_dims[3]
#   for xy in tiles:
#      xx = xy[0]; yy = xy[1]
#      outfile_img = imgout_path + "/" + name + "_" + im_number + "_x" + str(xx) + "_y" + str(yy) + ".png"
#      tile_img = source_img[(tb+yy*tile_y):(tb+(yy+1)*tile_y), (lb+xx*tile_x):(lb+(xx+1)*tile_x), :3]
#      cv2.imwrite(outfile_img, tile_img)
#      del tile_img

def find_tiles(dwnsamp_mask, all_indx, maskthresh):
   kept_tiles = []
   x_tiles = len(all_indx)
   y_tiles = len(all_indx[0])
   for i in range(x_tiles):
      for j in range(y_tiles):
         xy_vals = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
         wh_vals = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
         tile_mask = dwnsamp_mask[xy_vals[1]:(xy_vals[1]+wh_vals[1]) , xy_vals[0]:(xy_vals[0]+wh_vals[0])]
         if (np.sum(tile_mask) > maskthresh):
            kept_tiles.append((i, j))
         del tile_mask
   return kept_tiles

def save_tiles(source_img, all_indx, kept_tiles, imnumber, imgout_path, name):
   if not (name=="img" or name=="mask" or name=="premask"): return 0
   for xy in kept_tiles:
      i = xy[0]; j = xy[1]
      outfile_img = imgout_path + "/" + name + "_" + im_number + "_x" + str(i) + "_y" + str(j) + ".png"
      xy_vals = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
      wh_vals = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
      tile_img = source_img[xy_vals[1]:(xy_vals[1]+wh_vals[1]) , xy_vals[0]:(xy_vals[0]+wh_vals[0]) , :3]
      cv2.imwrite(outfile_img, tile_img)
      del tile_img


if __name__=="__main__":
   base_path = "/home/doran/Work/images/"
   training_dat = []
   folder_im = []
   folder_cnt = []

   batch_ID = "/Mouse_Aug2018_HE/"
   slidelist = ["668514", "668525", "668530", "676377", "676378"]
   for slide in slidelist:
      folder_im += [base_path + batch_ID]
      folder_cnt += [base_path + batch_ID + "/Analysed_slides/Analysed_"]
      training_dat += [base_path + batch_ID + slide + ".svs"]

#   batch_ID = "/mPAS_WIMM/"
#   slidelist = ["618446", "618451"]
#   for slide in slidelist:
#      folder_im += [base_path + batch_ID +"/raw_images/"]
#      folder_cnt += [base_path + batch_ID + "/Analysed_slides/Analysed_"]
#      training_dat += [base_path + batch_ID +"/raw_images/" + slide + ".svs"]
   
#   batch_ID = "/mPAS_WIMM/"
#   slidelist = ["618445", "578367", "620694", "643868", "643870"] # 618446, 618451
#   for slide in slidelist:
#      folder_im += [base_path + batch_ID +"/raw_images/"]
#      folder_cnt += [base_path + batch_ID + "/Analysed_slides/Analysed_"]
#      training_dat += [base_path + batch_ID +"/raw_images/" + slide + ".svs"]
#      
#   batch_ID = "/MAOA_March2018/"
#   slidelist = ["586576", "586575", "586572", "586574", "586577"]
#   for slide in slidelist:
#      folder_im += [base_path + batch_ID +"/raw_images/"]
#      folder_cnt += [base_path + batch_ID + "/Analysed_slides/Analysed_"]
#      training_dat += [base_path + batch_ID +"/raw_images/" + slide + ".svs"]

#   batch_ID = "/MAOA_slides/"
#   slidelist = ["540797", "540793"]
#   for slide in slidelist:
#      folder_im += [base_path + batch_ID +"/raw_images/"]
#      folder_cnt += [base_path + batch_ID + "/Analysed_slides/Analysed_"]
#      training_dat += [base_path + batch_ID +"/raw_images/" + slide + ".svs"]

#   batch_ID = "/KDM6A_March2018/"
#   slidelist = ["642739", "642719", "642708", "642728", "642709"]
#   for slide in slidelist:
#      folder_im += [base_path + batch_ID +"/raw_images/"]
#      folder_cnt += [base_path + batch_ID + "/Analysed_slides/Analysed_"]
#      training_dat += [base_path + batch_ID +"/raw_images/" + slide + ".svs"]      
#      
#   batch_ID = "/NONO_March2018/"
#   slidelist = ["627229", "627187", "627246", "627193", "627212"]
#   for slide in slidelist:
#      folder_im += [base_path + batch_ID +"/raw_images/"]
#      folder_cnt += [base_path + batch_ID + "/Analysed_slides/Analysed_"]
#      training_dat += [base_path + batch_ID +"/raw_images/" + slide + ".svs"]  
#      
#   batch_ID = "/STAG_March2018/"
#   slidelist = ["601178", "601177", "601166", "601160", "601163"]
#   for slide in slidelist:
#      folder_im += [base_path + batch_ID +"/raw_images/"]
#      folder_cnt += [base_path + batch_ID + "/Analysed_slides/Analysed_"]
#      training_dat += [base_path + batch_ID +"/raw_images/" + slide + ".svs"]      
   
   ###############################################################

   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/input/mouse/"
   imgout = dnnfolder + "/train/"
   maskout = dnnfolder + "/pre-mask/"
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
   tile_x = 256
   tile_y = 256
   maskthresh = 255 * 20 # throw away masks with fewer than 1000 white pixels

   dilate = True
   st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

   for i in range(2,n_slides):
      tracker = SummaryTracker()
      print("Slide %d" % i)
      # Read on whole .svs slide
      filename = training_dat[i]
      if (filename[-4:]==".svs"):
         obj_svs  = getROI_svs(filename, get_roi_plot = False)
         img      = getROI_img_osl(filename, (0,0),  obj_svs.dims_slides[0])
         xshape = obj_svs.dims_slides[0][0] # cols
         yshape = obj_svs.dims_slides[0][1] # rows
         xshape_unzoom = obj_svs.dims_slides[1][0] # cols
         yshape_unzoom = obj_svs.dims_slides[1][1] # rows
      else:
         img = cv2.imread(filename, cv2.IMREAD_COLOR)
         xshape = img.shape[1] # cols
         yshape = img.shape[0] # rows

      scaleval = int(xshape/xshape_unzoom)

#      lb = (xshape_unzoom % tile_x) // 2 # left buffer
#      num_x = xshape_unzoom//tile_x
#      tb = (yshape_unzoom % tile_y) // 2 # top buffer
#      num_y = yshape_unzoom//tile_y
#      shape_dims = (tile_x, tile_y, lb, tb)

      all_indx = get_tile_indices((xshape,yshape), overlap = 50, SIZE = (tile_x, tile_y))

      # Read in slide contours
      im_number = filename[:-4] # remove extension
      im_number = im_number[len(folder_im[i]):] # remove full path
      contours = read_cnt_text_file(folder_cnt[i] + im_number + "/crypt_contours.txt")
      # Draw contours to make mask
      big_mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
      for j in range(len(contours)):
         cv2.drawContours(big_mask, [contours[j]], 0, 255, -1)

      # Dilate to grow crypt contours?
      if (dilate==True): big_mask = cv2.morphologyEx(big_mask, cv2.MORPH_DILATE,  st_3, iterations=4)
        
      # Downsample to create unzoomed mask 
      dwnsamp_mask = cv2.pyrDown(big_mask)
      dwnsamp_mask = cv2.pyrDown(dwnsamp_mask)
      del big_mask
      
      # Determine which tiles are to be kept
      #kept_tiles = find_tiles(dwnsamp_mask, num_x, num_y, maskthresh, shape_dims)
      kept_tiles = find_tiles(dwnsamp_mask, all_indx, maskthresh)
      
      # Downsample to create unzoomed img
      dwnsamp_img = cv2.pyrDown(img)
      dwnsamp_img = cv2.pyrDown(dwnsamp_img)

      # Save img output
      #save_tiles(dwnsamp_img, kept_tiles, im_number, imgout, "img", shape_dims)
      save_tiles(dwnsamp_img, all_indx, kept_tiles, im_number, imgout, "img")

      # Reduce img below 255 and draw on contours
      img[img<21] -= 20
      for j in range(len(contours)):
         cv2.drawContours(img, [contours[j]], 0, (255,255,255), -1)
      
      # Downsample the contoured image and save kept tiles as pre-mask
      dwnsamp_img = cv2.pyrDown(img)
      dwnsamp_img = cv2.pyrDown(dwnsamp_img)
      del img
      #save_tiles(dwnsamp_img, kept_tiles, im_number, maskout, "premask", shape_dims)
      save_tiles(dwnsamp_img, all_indx, kept_tiles, im_number, maskout, "premask")
      tracker.print_diff()

