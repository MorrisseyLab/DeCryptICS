#!/usr/bin/env python3
import time
import math
import cv2
import os
import numpy as np
import pandas as pd
import openslide as osl
from MiscFunctions import getROI_img_osl, rescale_contours, add_offset, mkdir_p, read_cnt_text_file, plot_img

st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
st_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
st_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
st_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

## get the pixel coordinates of each crypt from a slide
imgpath = "/home/doran/Work/images/agne/DFC_ST_070220_A.svs"

pathend = imgpath.split('/')[-1] 
imname = pathend.split('.')[0]
fsl = os.path.abspath(imgpath[:-len(pathend)]) + '/Analysed_slides/Analysed_' + imname + '/'
cnts = np.asarray(read_cnt_text_file(fsl + 'crypt_contours.txt'))

cnt_pixels = np.zeros((len(cnts)), dtype=object)
for i in range(len(cnts)):
   bb_m = np.asarray(cv2.boundingRect(cnts[i]))
   XY = np.meshgrid(range(bb_m[0], bb_m[0]+bb_m[2]), range(bb_m[1], bb_m[1]+bb_m[3]))
   thiscnt = np.zeros((bb_m[2]*bb_m[3], 2), dtype=np.int32)
   j = 0
   for xx in range(XY[0].shape[1]):
      for yy in range(XY[1].shape[0]):
         if (cv2.pointPolygonTest(cnts[i], (XY[0][yy,xx], XY[1][yy,xx]), False)==0 or
             cv2.pointPolygonTest(cnts[i], (XY[0][yy,xx], XY[1][yy,xx]), False)==1   ):
            thiscnt[j, 0] = XY[0][yy,xx]
            thiscnt[j, 1] = XY[1][yy,xx]
            j += 1
   cnt_pixels[i] = thiscnt[:j, :]

np.save(fsl + 'crypt_pixels.npy', cnt_pixels)
