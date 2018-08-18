#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:30:18 2018

@author: doran
"""
import cv2
import numpy as np
from deconv_mat                        import *
from MiscFunctions                     import col_deconvol, col_deconvol_and_blur, col_deconvol_and_blur2
from cnt_Feature_Functions             import contour_MajorMinorAxis, st_3, plotCnt
from classContourFeat                  import getAllFeatures
from func_FindAndFilterLumens          import mergeAllContours, GetContAndFilter_TwoBlur
from multicore_morphology              import getForeground_mc
from automaticThresh_func              import calculate_thresholds
from knn_prune                         import prune_contours_knn, drop_broken_runs, prune_attributes, prune_minoraxes 
from Segment_clone_from_crypt          import find_clone_statistics

## If thresh_cut is None, thresholds will be calculated from the image
def Segment_crypts(img, thresh_cut, deconv_mat, nbins=20, find_clones = False):
    
    ## Avoid problems with multicore
    cv2.setNumThreads(0)       
          
    ## Colour Deconvolve to split channles into nuclear and clone stain
    ## Blur and threshold image
    ####################################################################

    ## If thesh_cut is a list then unpack, otherwise calculate theresholds
    if thresh_cut is None:
        thresh_cut_nucl, thresh_cut_nucl_blur, th_clone = calculate_thresholds(img, deconv_mat)
    else:
        thresh_cut_nucl, thresh_cut_nucl_blur, th_clone = thresh_cut 

    ## Colour deconvolution
    smallBlur_img_nuc, blurred_img_nuc, blurred_img_clone = col_deconvol_and_blur(img, deconv_mat, (11, 11), (37, 37), (27, 27))

    # Threshold
    _, nuclei_ch_raw = cv2.threshold( smallBlur_img_nuc,      thresh_cut_nucl, 255, cv2.THRESH_BINARY)
    _, nucl_thresh   = cv2.threshold(   blurred_img_nuc, thresh_cut_nucl_blur, 255, cv2.THRESH_BINARY)

    ## Clone segment 
    ###########################################
    clone_thresh = get_mPAS_Stains2(blurred_img_clone, smallBlur_img_nuc, th_clone)
    _, clone_ch_raw = cv2.threshold(blurred_img_clone, th_clone, 255, cv2.THRESH_BINARY)
          
    ## Clean up, filter nuclei and get foreground from nuclei
    ###########################################
    nuclei_ch_raw   = cv2.morphologyEx(nuclei_ch_raw, cv2.MORPH_OPEN,  st_3, iterations=1)
    nucl_thresh_aux = cv2.morphologyEx(  nucl_thresh, cv2.MORPH_OPEN,  st_3, iterations=1)
    foreground = getForeground_mc(nucl_thresh_aux)    
    backgrd    = 255 - foreground
    
    del foreground, nucl_thresh_aux
    
    ## Segment crypts lumen
    ###########################################
    qq_both = GetContAndFilter_TwoBlur([thresh_cut_nucl_blur, thresh_cut_nucl], [blurred_img_nuc, smallBlur_img_nuc], [-0.2,-0.1, 0, 0.1], 
                                       backgrd, clone_thresh, nuclei_ch_raw, smallBlur_img_nuc, n_cores = 8)

    ## Prune using various metrics
    ###########################################
    num_contours_kept = 0
    for run in qq_both:
        num_contours_kept += len(run[0])    
    if (not num_contours_kept==0):
        med_minor_axis = [np.median([contour_MajorMinorAxis(contours_i)[1] for contours_i in qq_i[0]]) for qq_i in qq_both if qq_i[0]!=[]]
        qq_new = drop_broken_runs(qq_both, med_minor_axis, nuclei_ch_raw)
        if (not len(qq_new)==0):
            crypt_cnt  = mergeAllContours(qq_new, img.shape[0:2])    
            features = getAllFeatures(crypt_cnt, nuclei_ch_raw, backgrd, smallBlur_img_nuc)

            # Prune on individual attributes
            minor_ax_thresh_individualcnts = max(med_minor_axis)*5./12.
            prune_cnts = prune_minoraxes(crypt_cnt, features, minor_ax_thresh_individualcnts)
            features = getAllFeatures(prune_cnts, nuclei_ch_raw, backgrd, smallBlur_img_nuc)

            # Combining the above prunings:
            prune_cnts = prune_attributes(prune_cnts, features)
            features = getAllFeatures(prune_cnts, nuclei_ch_raw, backgrd, smallBlur_img_nuc)

            # If broken, check for lots of small nonsense contours
            if (np.median(features.allSizes)<800):
                prune_cnts = []
                print("Broken: too many small nonsense contours found!")
            
            # Prune on knn attributes
            if (len(prune_cnts)>6):
                prune_cnts = prune_contours_knn(prune_cnts, features, nuclei_ch_raw, backgrd, smallBlur_img_nuc, stddevmult = 3.6, nn = 7)
                features = getAllFeatures(prune_cnts, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
        if (len(qq_new)==0):
            num_contours_kept = 0
    if (num_contours_kept==0): 
        prune_cnts = []
        
    crypt_cnt = prune_cnts
    
    clone_channel_features = 0
    if (find_clones==True):
       ## Find clone channel features
       ###########################################
       img_nuc, img_clone = col_deconvol_and_blur2(img, deconv_mat, (11, 11), (13, 13))
       ## DO THIS IN THE HIGH RES IMAGE RATHER THAN ZOOMED OUT IMAGE! ADD SCALE FACTOR TO ALL CONTOURS
       # (maybe have to move outside to SegementTiled.py and reload svs slide with adjusted contours?
       clone_channel_features = find_clone_statistics(crypt_cnt, img_nuc, img_clone, nbins=20)
    
    return crypt_cnt, clone_channel_features
    
