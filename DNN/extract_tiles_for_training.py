#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 7 11:45:42 2018

@author: doran
"""
import cv2, os, sys
import openslide as osl
import pandas as pd
import numpy as np
from experimental_DeCryptICS.MiscFunctions import read_cnt_text_file, getROI_img_osl, mkdir_p, plot_img

TILE_SIZE = 256

if __name__=="__main__":
   level = 1
   # Output paths
   basefolder   = '/home/doran/Work/py_code/experimental_DeCryptICS/DNN/'
   img_outpath  = basefolder + 'input/train/'
   mask_outpath = basefolder + 'input/train_masks/'
   mkdir_p(img_outpath)
   mkdir_p(mask_outpath)
   
   # Load csv with centroid X,Y, FP/TP, slide path, mark type, fufi/clone
   tilelist = pd.read_csv(sys.argv[1])
   tilelist = np.asarray(tilelist)
   oldcnts  = 'Nope'
   oldslide = 'Nope'

   # Go through csv centroid list
   for i in range(tilelist.shape[0]):
      slide_path = tilelist[i,0] 
      slide_num = slide_path.split('/')[-1].split('.')[0]
      cntpath   = '/' + os.path.join(*slide_path.split('/')[:-1]) + "/Analysed_slides/Analysed_" + slide_num + '/'
      if (tilelist[i,5][0].upper()=='F'): 
         cntpath = cntpath + 'fufi_contours.txt'
         suffix = "fufi"
      elif (tilelist[i,5][0].upper()=='C'): 
         cntpath = cntpath + 'clone_contours.txt'
         suffix = "clone"
      elif (tilelist[i,5][0].upper()=='l'): 
         cntpath = cntpath + 'crypt_contours.txt'
         suffix = "crypt"
      else:
         print("Incorrect identifier in clone_fufi column, breaking...")
         break
         
      mark = tilelist[i,1]
      truefalse = tilelist[i,4]
      
      # Load contours (fufi or clone) for specific slide
      if (oldcnts != cntpath):
         contours = read_cnt_text_file(cntpath)
      oldcnts = cntpath
      
      # Extract ROI from svs
      if (oldslide != slide_path):
         vim = osl.OpenSlide(slide_path)
         max_vals = vim.level_dimensions[level]
      oldslide = slide_path
      
      x , y = int(tilelist[i,2]/tilelist[i,6]/vim.level_downsamples[level]) , int(tilelist[i,3]/tilelist[i,6]/vim.level_downsamples[level])
      if x<TILE_SIZE/2:
         xl = 0
      elif x>(max_vals[0]-TILE_SIZE/2):
         xl = int(max_vals[0]-TILE_SIZE)
      else:
         xl = int(x-TILE_SIZE/2)
      if y<TILE_SIZE/2:
         yl = int(0)
      elif y>(max_vals[1]-TILE_SIZE/2):
         yl = int(max_vals[1]-TILE_SIZE)
      else:
         yl = int(y-TILE_SIZE/2)
      img = getROI_img_osl(slide_path, (xl, yl), (TILE_SIZE, TILE_SIZE), level = level)
      mask = np.zeros(img.shape[:2], dtype=np.uint8)      

      tilename = slide_num + '_' + mark + '_x' + str(xl) + '_y' + str(yl) + '_' + str(TILE_SIZE) + '_' + truefalse.upper() + '_' + suffix
      # if FP save tile and black negative mask      
      if (truefalse.upper()=='F'):
         cv2.imwrite(img_outpath + "/img_"+tilename+".png", img)
         cv2.imwrite(mask_outpath + "/mask_"+tilename+".png", mask)
      
      # if TP find contour in list by centroid intersection, draw mask and save
      elif (truefalse.upper()=='T'):
         # Draw only chosen contour
         if (tilelist[i,7][0].upper()=='O'):
            inside_cnt = None
            for cnt_i in contours:
               if (cv2.pointPolygonTest(cnt_i, (int(x*vim.level_downsamples[level]), int(y*vim.level_downsamples[level])), False) == True):
                  inside_cnt = cnt_i
                  break
            # Rescale and translate to ROI
            if (not type(inside_cnt)==type(None)):
               inside_cnt = (inside_cnt/vim.level_downsamples[level]).astype(np.int32)
               cntROI = inside_cnt - (xl, yl)
               cv2.drawContours(mask, [cntROI], 0, 255, -1) ## Get mask
               cv2.imwrite(img_outpath + "/img_"+tilename+".png", img)
               cv2.imwrite(mask_outpath + "/mask_"+tilename+".png", mask)
         # Draw all contours, then trim off sides
         if (tilelist[i,7][0].upper()=='A'):
            mask = np.zeros((max_vals[1],max_vals[0]), dtype=np.uint8)
            for cnt_i in contours:
               # Rescale and translate to ROI
               cnt_j = (cnt_i/vim.level_downsamples[level]).astype(np.int32)
               cntROI = cnt_j - (xl, yl)
               cv2.drawContours(mask, [cntROI], 0, 255, -1) ## Get mask
            # trim mask
            mask = mask[:TILE_SIZE, :TILE_SIZE]
            cv2.imwrite(img_outpath + "/img_"+tilename+".png", img)
            cv2.imwrite(mask_outpath + "/mask_"+tilename+".png", mask)
      
      
