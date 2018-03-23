# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:57:09 2016

@author: edward
"""
from joblib        import Parallel, delayed
from MiscFunctions import plot_img
import numpy as np, cv2
from cv2 import morphologyEx 
from cnt_Feature_Functions import *

def correct_indx_two_images(indx_img, indx_sub_img, delta_x, max_val):
    if indx_img==max_val or indx_img==0: 
        delta_x     = 0
    return indx_img + delta_x, indx_sub_img + delta_x

def correct_indx_over_under_shoot(indx, max_val):
    if indx>max_val: indx = max_val
    if indx<0:       indx = 0
    return int(indx)

## Choose    
def getImageSplitIndex(img_in, st_size, iterations, n_cores_even = 4, plot_split = False):    
    ## Multiply morpholodgy disc size times number of iterations times a value
    # that should be two but chose larger value
    overlap_region = st_size*iterations*5
    # Get tiling indexes with overlap_region both in x and y
    x_dim   = img_in.shape[0]
    y_dim   = img_in.shape[1]
    
#    n_cores_even = 4
    delta_x = np.ceil((x_dim + 1.*(n_cores_even-1)*overlap_region)/n_cores_even)
    delta_y = np.ceil((y_dim + (2-1)*overlap_region)/2.)
    y_end   = 0
    y_last  = 0
    all_indices = []
    for y_i in range(2):
        y_start = correct_indx_over_under_shoot(y_last - overlap_region, y_dim)
        y_end   = correct_indx_over_under_shoot(y_start + delta_y, y_dim)
        x_last  = 0
        for x_i in range(n_cores_even):
            x_start = correct_indx_over_under_shoot(x_last - overlap_region, x_dim)
            x_end   = correct_indx_over_under_shoot(x_start + delta_x, x_dim)
            all_indices.append((x_start, x_end, y_start, y_end))   
#            print "Delta x is " + str(x_end - x_start)
#            print "Delta y is " + str(y_end - y_start)
            x_last  = x_end
        y_last = y_end

    if(plot_split):
        img_plot = img_in.copy()
        for rect_i in all_indices: cv2.rectangle(img_plot, (rect_i[3], rect_i[1]), (rect_i[2], rect_i[0]), 125, 20)
        plot_img(img_plot, hold_plot = True)        
    return all_indices, overlap_region

def cv2_morph(img_in, indx_i, morph_type, st_n, iterations):
    img_out = morphologyEx(img_in[indx_i[0]:indx_i[1], indx_i[2]:indx_i[3]], morph_type, st_n, iterations = iterations)
    return img_out


def multicore_morph(img_in, morph_type, st_n, iterations, n_cores_even = 4): 
#    n_cores_even = 4 ## Number of cores (will be *2)
    st_size                  = st_n.shape[0]
    indx_all, overlap_region = getImageSplitIndex(img_in, st_size, iterations, n_cores_even = n_cores_even, plot_split = False)
    results  = Parallel(n_jobs=n_cores_even*2)(delayed(cv2_morph)(img_in, indx_i,morph_type, st_n, iterations) for indx_i in indx_all)
    img_out  = np.zeros(img_in.shape, dtype= np.uint8)
    for indx_i,img_i in zip(indx_all, results):
        overlap_exclude = int(overlap_region/4)
        indx_1, indx_1_sub = correct_indx_two_images(indx_i[0], 0, overlap_exclude, img_in.shape[0])
        indx_2, indx_2_sub = correct_indx_two_images(indx_i[1], img_i.shape[0], -1*overlap_exclude, img_in.shape[0])
        indx_3, indx_3_sub = correct_indx_two_images(indx_i[2], 0, overlap_exclude, img_in.shape[1])
        indx_4, indx_4_sub = correct_indx_two_images(indx_i[3], img_i.shape[1], -1*overlap_exclude, img_in.shape[1])
        img_out[indx_1:indx_2, indx_3:indx_4] = img_i[indx_1_sub:indx_2_sub, indx_3_sub:indx_4_sub]
    return img_out


def cv2_morph_bg(img_in, indx_i):
    img_out = morphologyEx(img_in[indx_i[0]:indx_i[1], indx_i[2]:indx_i[3]], cv2.MORPH_CLOSE, st_3, iterations = 160)
    img_out = morphologyEx(img_out,   cv2.MORPH_OPEN,   st_3, iterations = 10) 
    img_out = morphologyEx(img_out, cv2.MORPH_DILATE,   st_3, iterations = 10) 
    return img_out


def getForeground_mc(img_in, n_cores_even = 4):
    # D.K. test:
    # img_in = nucl_thresh_aux
    # n_cores_even = 4
    indx_all, overlap_region = getImageSplitIndex(img_in, 3, 160, plot_split = False, n_cores_even = n_cores_even)
    results  = Parallel(n_jobs=n_cores_even*2)(delayed(cv2_morph_bg)(img_in, indx_i) for indx_i in indx_all)
    img_out  = np.zeros(img_in.shape, dtype= np.uint8)
    for indx_i,img_i in zip(indx_all, results):
        overlap_exclude = int(overlap_region/4)
        indx_1, indx_1_sub = correct_indx_two_images(indx_i[0], 0, overlap_exclude, img_in.shape[0])
        indx_2, indx_2_sub = correct_indx_two_images(indx_i[1], img_i.shape[0], -1*overlap_exclude, img_in.shape[0])
        indx_3, indx_3_sub = correct_indx_two_images(indx_i[2], 0, overlap_exclude, img_in.shape[1])
        indx_4, indx_4_sub = correct_indx_two_images(indx_i[3], img_i.shape[1], -1*overlap_exclude, img_in.shape[1])
        img_out[indx_1:indx_2, indx_3:indx_4] = img_i[indx_1_sub:indx_2_sub, indx_3_sub:indx_4_sub]
    foreground = filterSmallArea(img_out, 5e5)  
    return foreground
    
