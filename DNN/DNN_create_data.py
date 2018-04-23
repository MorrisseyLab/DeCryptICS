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


# First need to create training data. This will be of the form:
#   -- 1000x1000 tiles of pre-analysed .svs images;
#   -- 1000x1000 tiles of binary mask created by contours.
# Each sample is then a tile of the original image and its binary mask.
# To do this: use a version of the segement tile image function (with
# no overlapping?) to get indices of tiles. To retain equal sizes, throw
# away the final column and row which will be the wrong size.
# Generate a binary mask for the entire .svs image then tile in same way
# using the same indices. Then save image tile and mask tile together.

base_path = "/home/doran/Work/images/"

# mPAS .svs slides:
batch_ID1 = "/mPAS_WIMM/"
folder_im1 = base_path + batch_ID1 +"/raw_images/"
folder_cnt1 = base_path + batch_ID1 + "/Analysed_slides/Analysed_"
mpas_training_dat_svs = [folder_im1 + "575845.svs"]

# MAOA .svs slides:
batch_ID2 = "/MAOA_slides/"
folder_im2 = base_path + batch_ID2 +"/raw_images/"
folder_cnt2 = base_path + batch_ID2 + "/Analysed_slides/Analysed_"
maoa_training_dat_svs = [folder_im2 + "540796.svs"]
batch_ID2b = "/MAOA_March2018/"
folder_im2b = base_path + batch_ID2b +"/raw_images/"
folder_cnt2b = base_path + batch_ID2b + "/Analysed_slides/Analysed_"
maoa_training_dat_svsb = [folder_im2b + "586574.svs"]

# mPAS jpg test images
batch_ID3 = "/mPAS_Clone_test_images/"
folder_im3 = base_path + batch_ID3 +"/raw_images/"
folder_cnt3 = base_path + batch_ID3 + "/Analysed_slides/"
mpas_training_dat_jpg = glob.glob(folder_im3 + "*.jpg")

# KDM6A .svs slides:
batch_ID4 = "/KDM6A_March2018/"
folder_im4 = base_path + batch_ID4 +"/raw_images/"
folder_cnt4 = base_path + batch_ID4 + "/Analysed_slides/Analysed_"
kdm6a_training_dat_svs = [folder_im4 + "642708.svs"]

# Possible slides for NONO, STAG, etc.
batch_ID5 = "/NONO_March2018/"
folder_im5 = base_path + batch_ID5 +"/raw_images/"
folder_cnt5 = base_path + batch_ID5 + "/Analysed_slides/Analysed_"
nono_training_dat_svs = [folder_im5 + "627193.svs", folder_im5 + "627187.svs"]
batch_ID6 = "/STAG_March2018/"
folder_im6 = base_path + batch_ID6 +"/raw_images/"
folder_cnt6 = base_path + batch_ID6 + "/Analysed_slides/Analysed_"
stag_training_dat_svs = [folder_im6 + "601160.svs", folder_im6 + "601177.svs"]

###############################################################

dnnfolder = "/home/doran/Work/py_code/DNN_ImageMasking/input/"
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
training_dat = kdm6a_training_dat_svs
folder_cnt = folder_cnt4
folder_im = folder_im4
n_slides = len(training_dat) # shift = 0
tile_x = 1024
tile_y = 1024
maskthresh = 255 * 1000 # throw away masks with fewer than 1000 white pixels
# There is an error here where the last few tiles are empty
for i in range(n_slides):
    # Read on whole .svs slide
    filename = training_dat[i]
    if (filename[-4:]==".svs"):
        obj_svs  = getROI_svs(filename , get_roi_plot = False)
        img      = getROI_img_vips(filename, (0,0),  obj_svs.dims_slides[0])
        xshape = obj_svs.dims_slides[0][1] # cols
        yshape = obj_svs.dims_slides[0][0] # rows
    else:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        xshape = img.shape[1] # cols
        yshape = img.shape[0] # rows
    
    lb = (xshape % tile_x) // 2 # left buffer
    num_x = xshape//tile_x
    tb = (yshape % tile_y) // 2 # yop buffer
    num_y = yshape//tile_y

    # Read in slide contours
    im_number = filename[:-4] # remove extension
    im_number = im_number[len(folder_im):] # remove full path
    contours = read_cnt_text_file(folder_cnt + im_number + "/crypt_contours.txt")
    # Draw contours to make mask
    big_mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    for j in range(len(contours)):
        cv2.drawContours(big_mask, [contours[j]], 0, 255, -1)
    # Tile image and mask; save output
    for xx in range(num_x):
        print("Doing x %d of %d" % (xx, num_x))
        for yy in range(num_y):
            outfile_img = imgout + "/img_" + im_number + "_x" + str(xx) + "_y" + str(yy) + ".png"
            outfile_mask = maskout + "/mask_" + im_number + "_x" + str(xx) + "_y" + str(yy) + ".png"
            tile_img = img[(tb+yy*tile_y):(tb+tile_y+yy*tile_y), (lb+xx*tile_x):(lb+tile_x+xx*tile_x), :3]
            tile_mask = big_mask[(tb+yy*tile_y):(tb+tile_y+yy*tile_y), (lb+xx*tile_x):(lb+tile_x+xx*tile_x)]
            if (np.sum(tile_mask) > maskthresh):
                cv2.imwrite(outfile_img, tile_img)
                cv2.imwrite(outfile_mask, tile_mask)
            del tile_img, tile_mask
    del img
    del big_mask
    
    
    
