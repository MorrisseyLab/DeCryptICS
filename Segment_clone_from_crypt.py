# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:18:27 2015

@author: edward
"""
import cv2
import numpy as np
from MiscFunctions          import plot_img
from cnt_Feature_Functions  import *
from classContourFeat      import getAllFeatures
import matplotlib.pylab as plt
from knn_prune import tukey_lower_thresholdval, tukey_upper_thresholdval

def retrieve_clone_nuclear_features(crypt_cnt, nuclei_ch_raw, clone_ch_raw, backgrd, smallBlur_img_nuc):
    nuc_feat = getAllFeatures(crypt_cnt, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
    clone_feat = getAllFeatures(crypt_cnt, clone_ch_raw, backgrd, smallBlur_img_nuc)
    frac_halo = clone_feat.allHalo/nuc_feat.allHalo # divide by zero risk
    frac_halogap = (1+clone_feat.allHaloGap)/(1+nuc_feat.allHaloGap)
    clone_content = clone_feat.allMeanNucl
    return frac_halo, frac_halogap, clone_content

def find_clones(crypt_cnt, clone_channel_feats, clone_marker_type, numIQR=2):                        
    frac_halo = clone_channel_feats[0]
    frac_halogap = clone_channel_feats[1]
    clone_content = clone_channel_feats[2]
    if (clone_marker_type=="P"):
        inds1 = np.where(frac_halo > tukey_upper_thresholdval(frac_halo, numIQR = numIQR))[0]
        inds2 = np.where(frac_halogap < tukey_lower_thresholdval(frac_halogap, numIQR = numIQR))[0]
        inds = np.unique(np.hstack([inds1,inds2]))
        full_partial_statistic = (frac_halo[inds] + (2 - frac_halogap[inds]))/2.
    if (clone_marker_type=="N"):
        inds1 = np.where(frac_halo < tukey_lower_thresholdval(frac_halo, numIQR = numIQR))[0]
        inds2 = np.where(frac_halogap > tukey_upper_thresholdval(frac_halogap, numIQR = numIQR))[0]
        inds = np.unique(np.hstack([inds1,inds2]))
        full_partial_statistic = ((1 - frac_halo[inds]) + (frac_halogap[inds]-1))/2.
    if (clone_marker_type=="PNN"):
        inds = np.where(clone_content > tukey_upper_thresholdval(clone_content, numIQR = numIQR))[0]
        inds = np.asarray([ii for ii in inds_cc if clone_content[ii] > 5e-3]) # flush out junk contours
        full_partial_statistic = clone_content[inds]
    if (clone_marker_type=="NNN"):
        inds = np.where(clone_content < tukey_lower_thresholdval(clone_content, numIQR = numIQR))[0]
        full_partial_statistic = 1 - clone_content[inds]
    clone_cnt = list(np.asarray(crypt_cnt)[inds])
    return clone_cnt, full_partial_statistic

def plot_contour_roi(cnt_i, img1):
    expand_box    = 50
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
    plt.imshow(img_ROI)
