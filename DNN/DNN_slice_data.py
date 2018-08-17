#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:06:10 2018

@author: doran
"""
import cv2, glob, os, errno
import numpy as np
import pyvips

def get_tile_indices(maxvals, overlap = 200, SIZE = (1024, 1024)):
    all_indx = []
    width = SIZE[0]
    height = SIZE[1]
    x_max = maxvals[0] # x -> cols
    y_max = maxvals[1] # y -> rows
    num_tiles_x = x_max // (width-overlap)
    endpoint_x  = num_tiles_x*(width-overlap) + overlap    
    overhang_x  = x_max - endpoint_x
    if (overhang_x>0): num_tiles_x += 1
    
    num_tiles_y = y_max // (height-overlap)
    endpoint_y  = num_tiles_y*(height-overlap) + overlap    
    overhang_y  = y_max - endpoint_y
    if (overhang_y>0): num_tiles_y += 1   
     
    for i in range(num_tiles_x):
        x0 = i*(width - overlap)
        if (i == (num_tiles_x-1)): x0 = x_max - width
        all_indx.append([])
        for j in range(num_tiles_y):
            y0 = j*(height - overlap)
            if (j == (num_tiles_y-1)): y0 = y_max - height
            all_indx[i].append((x0, y0, width, height))
    return all_indx
        
if __name__=="__main__":
   ## Slicing pre-mask and training images into smaller chunks
   ##########################################################################
   newsize = 256
   oldsize = 2048 # 4096
    
   # load pre-masks
   dnnpath = "/home/doran/Work/py_code/zoomed_out_DeCryptICS/DNN/input"
   outpath = dnnpath + "/pre-mask/"
   inpath = outpath + "mpas_extra_2048/" #"all_stains/"
   pmfiles = glob.glob(inpath + "*.png")
   
   names = [path.split('/')[-1][8:] for path in pmfiles]
   imnames = ['img_'+n for n in names]
   outpath_i = dnnpath + "/train/"
   inpath_i = outpath_i + "mpas_extra_2048/" #"all_stains/"
   imfiles = [inpath_i+n for n in imnames]
   
   # run pre-masks
   for path in pmfiles:
      img = cv2.imread(path, cv2.IMREAD_COLOR)
      name = path.split('/')[-1].split('.')[0]
      all_indx = get_tile_indices((oldsize,oldsize), overlap = 50, SIZE = (newsize, newsize))
      x_tiles = len(all_indx)
      y_tiles = len(all_indx[0])
      for i in range(x_tiles):
         for j in range(y_tiles):
            xy_vals = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
            wh_vals = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
            img1    = img[xy_vals[1]:(xy_vals[1]+wh_vals[1]) , xy_vals[0]:(xy_vals[0]+wh_vals[0])]
            outfile = name + '_tile' + str(i) + '_' + str(j) + '.png'
            cv2.imwrite(outpath + outfile, img1)
            
   # run training images
   for path in imfiles:
      img = cv2.imread(path, cv2.IMREAD_COLOR)
      name = path.split('/')[-1].split('.')[0]
      all_indx = get_tile_indices((oldsize,oldsize), overlap = 50, SIZE = (newsize, newsize))
      x_tiles = len(all_indx)
      y_tiles = len(all_indx[0])
      for i in range(x_tiles):
         for j in range(y_tiles):
            xy_vals = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
            wh_vals = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
            img1    = img[xy_vals[1]:(xy_vals[1]+wh_vals[1]) , xy_vals[0]:(xy_vals[0]+wh_vals[0])]
            outfile = name + '_tile' + str(i) + '_' + str(j) + '.png'
            cv2.imwrite(outpath_i + outfile, img1)


   # Pyvips SLOW IMPLEMENTATION
#   # load pre-masks
#   dnnpath = "/home/doran/Work/py_code/zoomed_out_DeCryptICS/DNN/input"
#   outpath = dnnpath + "/pre-mask/"
#   inpath = outpath + "all_stains/"
#   imfiles = glob.glob(inpath + "*.png")
#   # run processing and save
#   for path in imfiles:
#      img = pyvips.Image.new_from_file(path)
#      name = path.split('/')[-1].split('.')[0]
#      n = int(img.width/float(newsize))
#      for x in range(n):
#         for y in range(n):
#            img1 = img.crop(x*newsize, y*newsize , newsize , newsize)
#            outfile = name + '_tile' + str(x) + str(y) + '.png'
#            img1.write_to_file(outpath + outfile)
#            
#   # load training images
#   dnnpath = "/home/doran/Work/py_code/zoomed_out_DeCryptICS/DNN/input"
#   outpath = dnnpath + "/train/"
#   inpath = outpath + "all_stains/"
#   imfiles = glob.glob(inpath + "*.png")

#   # run processing and save
#   for path in imfiles:
#      img = pyvips.Image.new_from_file(path)
#      name = path.split('/')[-1].split('.')[0]
#      n = int(img.width/float(newsize))
#      for x in range(n):
#         for y in range(n):
#            img1 = img.crop(x*newsize, y*newsize , newsize , newsize)
#            outfile = name + '_tile' + str(x) + str(y) + '.png'
#            img1.write_to_file(outpath + outfile)
#            
#            
            
            

