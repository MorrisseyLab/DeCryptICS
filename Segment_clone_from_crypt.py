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

def find_clones_negspace(crypt_cnt, nuclei_ch_raw, clone_ch_raw, backgrd, smallBlur_img_nuc):
    nuc_feat = getAllFeatures(crypt_cnt, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
    clone_feat = getAllFeatures(crypt_cnt, clone_ch_raw, backgrd, smallBlur_img_nuc)
    # Compare nuclear halo and max halo hap between the nuclei and clone channel.
    # Should see approximate equality in the wild types, and a big drop-off in the 
    # clone channel for the clonal crypts.  Max halo gap should give us an indication
    # of number of single-cell clones?
    frac_halo = clone_feat.allHalo/nuc_feat.allHalo
    frac_halogap = (1+clone_feat.allHaloGap)/(1+nuc_feat.allHaloGap)
    inds = np.where(frac_halo<0.95)[0]
    inds1 = np.where(frac_halogap>1.05)[0]
    indsu = np.unique(np.hstack([inds,inds1]))
    clone_cnt = list(np.asarray(crypt_cnt)[indsu])
    return clone_cnt

def find_clones_posspace(crypt_cnt, clone_ch_raw, backgrd, smallBlur_img_nuc):
    clone_feat = getAllFeatures(crypt_cnt, clone_ch_raw, backgrd, smallBlur_img_nuc)
    clone_content = clone_feat.allMeanNucl
    inds = np.where(clone_content>1e-3)[0]
    clone_cnt = list(np.asarray(crypt_cnt)[inds])
    return clone_cnt

def plot_contour_roi(cnt_i, img1):
    expand_box    = 50
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
    plt.imshow(img_ROI)

#def Halo_RGB_space(cnt_i, img1):
#    # Expand box
#    expand_box    = 50
#    roi           = cv2.boundingRect(cnt_i)            
#    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
#    roi[roi <1]   = 0
#    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
#    cnt_roi       = cnt_i - Start_ij_ROI # chnage coords to start from x,y
#    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :] 
#    max_dilations      = 20
#    rgb_vectors = []
#    for i in range(1, max_dilations+1):        
#        halo_cnt = extractHaloContour(cnt_roi, img_ROI, i)
#        halo_cnt = halo_cnt + 0*Start_ij_ROI
#        h_cnt = np.zeros([halo_cnt.shape[1], 1, halo_cnt.shape[3]], dtype=np.int32)
#        for ii in range(halo_cnt.shape[1]):
#            h_cnt[ii, 0, :] = halo_cnt[0, ii, 0, :]
#        col_mat = extract_3_1D_vecs(h_cnt, img_ROI)
#        rgb_vectors.append(col_mat)
#    return rgb_vectors
#
#def extract_3_1D_vecs(cnt_i, img_ROI):
#    numpixels = cnt_i.shape[0]
#    colmat = np.zeros([1, numpixels, 3])
#    k = 0
#    for xy_i in cnt_i[:,0,:]:        
#        x = xy_i[0]
#        y = xy_i[1]
#        colmat[0, k, 0] = img_ROI[y,x,0]
#        colmat[0, k, 1] = img_ROI[y,x,1]
#        colmat[0, k, 2] = img_ROI[y,x,2]
#        k += 1 
#    return colmat

#########################################

#    cv2.transform(, deconv_mat)
#    
#    # wildtype
#    cnt_i = cnts[4]
#    img1 = big_img
#    expand_box    = 50
#    roi           = cv2.boundingRect(cnt_i)            
#    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
#    roi[roi <1]   = 0
#    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
#    cnt_roi       = cnt_i - Start_ij_ROI # chnage coords to start from x,y
#    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
#    plt.imshow(img_ROI)
#    
#    # clone partial
#    cnt_j = cnts[10]
#    img1 = big_img
#    expand_box    = 50
#    roi           = cv2.boundingRect(cnt_j)
#    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
#    roi[roi <1]   = 0
#    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
#    cnt_roi       = cnt_i - Start_ij_ROI # chnage coords to start from x,y
#    img_ROIj       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
#    plt.imshow(img_ROIj)
#    
#    # clone full
#    cnt_k = cnts[23]
#    img1 = big_img
#    expand_box    = 50
#    roi           = cv2.boundingRect(cnt_k)
#    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
#    roi[roi <1]   = 0
#    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
#    cnt_roi       = cnt_i - Start_ij_ROI # chnage coords to start from x,y
#    img_ROIk       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
#    plt.imshow(img_ROIk)