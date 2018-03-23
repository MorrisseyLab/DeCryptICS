# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:18:27 2015

@author: edward
"""

import cv2, time, pdb
import numpy as np
from MiscFunctions          import plot_img, col_deconvol, col_deconvol_32
from cnt_Feature_Functions  import *


def unpack_or_calculate_thresholds(img, deconv_mat_MPAS, thresh_cut, smallBlur_img_nuc, blurred_img_nuc):
    ## If thresholds aren't provided calculate below with Otsu
    if thresh_cut is not None:
        thresh_cut = np.array(thresh_cut, dtype=np.int)
    else:
        img_deconv32  = col_deconvol_32(img, deconv_mat_MPAS)
        thresh_stringent, thresh_lax  = get_mPAS_Thresholds(img_deconv32)
        thresh_cut_nucl, _ = cv2.threshold( smallBlur_img_nuc, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh_cut_nucl_blur, _ = cv2.threshold( blurred_img_nuc, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh_cut = np.array([thresh_cut_nucl, thresh_cut_nucl_blur, thresh_stringent, thresh_lax], dtype=np.int)
    return thresh_cut

def correctBelowZero(thresh_i):
    if(thresh_i<1): thresh_i = 1
    return thresh_i

def get_mPAS_Thresholds(blurred_img_mPAS_32):
    mult_stringent = 4
    mult_lax       = 2.5
    
    perc_vals        = np.percentile(blurred_img_mPAS_32, [4, 50])
    thresh_stringent = 255*(perc_vals[1] + mult_stringent*(perc_vals[1] - perc_vals[0]))
    thresh_lax       = 255*(perc_vals[1] + mult_lax*(perc_vals[1] - perc_vals[0]))
    
    thresh_stringent = correctBelowZero(thresh_stringent)
    thresh_lax       = correctBelowZero(thresh_lax)
    return thresh_stringent, thresh_lax

def get_mPAS_ThreshAndStains(img, img_deconv_mpas, deconv_mat_mpas_purp):
    criteria_hist  = 30
    mult_stringent = 4
    mult_lax       = 2.5

    #img_deconv_mpas = chooseColDeconv(img) 
    img_deconv_mpas_32 = col_deconvol_32(img, deconv_mat_mpas_purp) 
    #plot_img(img, hold_plot=True)
    
    blurred_mPAS    = cv2.GaussianBlur(img_deconv_mpas[:,:,1], (25, 25), 0)
    blurred_mPAS_32 = cv2.GaussianBlur(img_deconv_mpas_32[:,:,1], (25, 25), 0)
    #plot_img(blurred_mPAS, hold_plot=True)
    
#    #thresh_mPAS  = np.mean(blurred_mPAS) + 3*np.std(blurred_mPAS) # 3
#    thresh_mPAS    = np.mean(blurred_mPAS) + 3*np.std(blurred_mPAS)
    perc_vals        = np.percentile(blurred_mPAS_32, [4, 50])
    thresh_stringent = 255*(perc_vals[1] + mult_stringent*(perc_vals[1] - perc_vals[0]))
    thresh_lax       = 255*(perc_vals[1] + mult_lax*(perc_vals[1] - perc_vals[0]))
    
    thresh_stringent = correctBelowZero(thresh_stringent)
    thresh_lax       = correctBelowZero(thresh_lax)
        
#    _, mpas_thresh_raw  = cv2.threshold(blurred_mPAS, thresh_mPAS, 255, cv2.THRESH_BINARY)
    _, mpas_thresh = cv2.threshold(blurred_mPAS,    thresh_stringent, 255, cv2.THRESH_BINARY)
    
    if thresh_stringent > criteria_hist: # Very clean and string signal <30      
        _, mpas_thresh_lax = cv2.threshold(blurred_mPAS,    thresh_lax, 255, cv2.THRESH_BINARY)
        mpas_thresh = cv2.morphologyEx( mpas_thresh, cv2.MORPH_OPEN,  st_3, iterations=2)
        mpas_thresh = filterContains(mpas_thresh_lax, mpas_thresh, 0)

#    plot_img((mpas_thresh_raw, mpas_thresh_raw2, mpas_thresh_raw3, mpas_thresh_final), 2, hold_plot=True)
    
    mpas_thresh = cv2.morphologyEx( mpas_thresh,  cv2.MORPH_CLOSE,  st_5, iterations=1)
    mpas_thresh = cv2.morphologyEx( mpas_thresh,   cv2.MORPH_OPEN,  st_5, iterations=3)
    
    mpas_thresh          = filterSmallArea_outer(mpas_thresh, 200)
    mPas_putative_cnt, _ = cv2.findContours(mpas_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [-2:]
    #allMean3Channel      = [contour_mean_Area(i, img_deconv_mpas[:,:,1]) for i in mPas_putative_cnt]
#    allMeannucl          = [contour_mean_Area(i, img_deconv_mpas[:,:,0]) for i in mPas_putative_cnt]
    #kk                   = np.column_stack((allMean3Channel, allMeannucl))
    #plotCntAndFeat(img, mPas_putative_cnt, kk, kk[:,1] < 0.5)
    mpas_thresh = filterStains(mpas_thresh, img_deconv_mpas[:,:,0], 0.5)
#    plot_img(cv2.merge((mpas_thresh, mpas_thresh2, mpas_thresh2)), hold_plot = True)
    return mpas_thresh


#def get_mPAS_Stains(img, img_deconv_mpas, thresh_stringent, thresh_lax):
#    criteria_hist  = 30

#    blurred_mPAS   = cv2.GaussianBlur(img_deconv_mpas[:,:,1], (25, 25), 0)
#    _, mpas_thresh = cv2.threshold(blurred_mPAS, thresh_stringent, 255, cv2.THRESH_BINARY)
#    
#    if thresh_stringent > criteria_hist: # Very clean and strong signal if thresh_stringent <30      
#        _, mpas_thresh_lax = cv2.threshold(blurred_mPAS,    thresh_lax, 255, cv2.THRESH_BINARY)
#        mpas_thresh = cv2.morphologyEx( mpas_thresh, cv2.MORPH_OPEN,  st_3, iterations=2)
#        mpas_thresh = filterContains(mpas_thresh_lax, mpas_thresh, 0)

##    plot_img((mpas_thresh_raw, mpas_thresh_raw2, mpas_thresh_raw3, mpas_thresh_final), 2, hold_plot=True)
#    
#    mpas_thresh = cv2.morphologyEx( mpas_thresh,  cv2.MORPH_CLOSE,  st_5, iterations=1)
#    mpas_thresh = cv2.morphologyEx( mpas_thresh,   cv2.MORPH_OPEN,  st_5, iterations=3)
#    
#    mpas_thresh          = filterSmallArea_outer(mpas_thresh, 200)
#    mPas_putative_cnt, _ = cv2.findContours(mpas_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [-2:]
#    ## Filter stains with more than x% overlap with other channel
#    mpas_thresh          = filterStains(mpas_thresh, img_deconv_mpas[:,:,0], 0.5)
#    return mpas_thresh
 
def get_mPAS_Stains2(blurred_img_clone, smallBlur_img_nuc, th_clone):

    _, mpas_thresh = cv2.threshold(blurred_img_clone, th_clone, 255, cv2.THRESH_BINARY)
    
    mpas_thresh = cv2.morphologyEx( mpas_thresh,  cv2.MORPH_CLOSE,  st_5, iterations=1)    
    mpas_thresh = cv2.morphologyEx( mpas_thresh,   cv2.MORPH_OPEN,  st_5, iterations=3)    

#    plot_img(mpas_thresh, hold_plot=True, nameWindow = 'After open')
    # D.K. filtering out areas smaller than 200 pixels (a chosen value)
    mpas_thresh          = filterSmallArea_outer(mpas_thresh, 200)
#    plot_img(mpas_thresh, hold_plot=True, nameWindow = 'After filter small area')
    # D.K. The below variable mPas_putative_cnt is not used
    mPas_putative_cnt, _ = cv2.findContours(mpas_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [-2:]
    ## Filter stains with more than x% overlap with other channel   
    mpas_thresh          = filterStains(mpas_thresh, smallBlur_img_nuc, 0.5)
#    plot_img(mpas_thresh, hold_plot=True, nameWindow = 'After filter high nuclear overlap')
    return mpas_thresh


