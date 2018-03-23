#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:52:01 2018

@author: doran
"""
from SegmentTiled_MPAS import read_cnt_text_file, write_cnt_text_file
from devel_knn_prune import remove_tiling_overlaps_knn
import glob
import cv2

if __name__=="__main__":
    folder_in  = "/home/doran/Work/images/mPAS_WIMM/raw_images/"
#    imfiles = glob.glob(folder_in + "*.svs")
#    imfiles = [name[:-4] for name in imfiles] 
#    imfiles = [name[len(folder_in):] for name in imfiles
    imfiles = ["620677"]
    folder_out = "/home/doran/Work/images/mPAS_WIMM/Analysed_slides/"
    file_in = folder_out + "Analysed_" + imfiles[0] + "/crypt_contours.txt"
    
    crypt_contours = read_cnt_text_file(file_in)
    crypt_contours, num_removed = remove_tiling_overlaps_knn(crypt_contours)
    
    new_cnts = []
    
    #Finding vertices in input image
    for i in crypt_contours:
        approx = cv2.approxPolyDP(i, 0.01*cv2.arcLength(i,True), True)
        if (len(approx)<6):
            new_cnts.append(i)
        else:
            new_cnts.append(approx)
    crypt_contours = new_cnts
    write_cnt_text_file(crypt_contours, file_in+"temp.txt")