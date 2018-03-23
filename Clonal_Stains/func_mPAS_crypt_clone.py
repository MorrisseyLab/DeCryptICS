# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:55:23 2015

@author: edward
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from cnt_Feature_Functions import *

def joinClonesInSameCrypt(mPAS_clone_info):
    crypts_all           = np.unique(mPAS_clone_info[:, 4])
    mPAS_clone_info_coll = np.zeros((len(crypts_all), 6))
    for ii in range(len(crypts_all)):
        crypt_i      = crypts_all[ii]
        indx_i       = mPAS_clone_info[:,4] == crypt_i
        info_crypt_i = mPAS_clone_info[indx_i,:]
        mPAS_clone_info_coll[ii,:] = [         info_crypt_i[0,0], # Clust num
                                               info_crypt_i[0,1], # Frac clonal (crypt wise value, not clone specific)
                                       np.sum(info_crypt_i[:,2]), # area of mpas
                                               info_crypt_i[0,3], # area of crypt
                                               info_crypt_i[0,4],
                                        np.max(info_crypt_i[:,5])] # indx of crypt
    return mPAS_clone_info_coll

## Features are [cluster_num, frac_clonal, area_mPAS, area_crypt, crypt_indx, intens_mpas] 
## Return crypt wise info (frac clonal, cluster [ie clone patch] number and crypt id)
## Use eroded crypts to get more accurate clonal fraction 
def getCloneFeatures(mpas_cnt, crypts_erode_cnt, crypt_cnt, mPAS_cluster_cnt, mpas_thresh, vec_intens_mpas):
    mPAS_clone_info = np.zeros((len(mpas_cnt) , 6))
    mPAS_clone_info[:,5] = vec_intens_mpas
    for ii in range(len(mpas_cnt)):
        ## Find crypt 
        crypt_eroded_ii_cnt , _  = find_Contour_inside(mpas_cnt[ii], crypts_erode_cnt, ret_index = True)
        crypt_ii_cnt, crypt_indx = find_Contour_inside(mpas_cnt[ii], crypt_cnt, ret_index = True)
        # If no mPAS is found
        if type(crypt_indx) is list: crypt_indx = -1            
        frac_clonal = contour_mean_Area(crypt_eroded_ii_cnt, mpas_thresh) if len(crypt_eroded_ii_cnt) > 0 else 0 # Make sure there is something            
        m_ij        = cv2.moments(mpas_cnt[ii])
        pos_xy      = (int(m_ij['m10']/m_ij['m00']), int(m_ij['m01']/m_ij['m00']))
        clust_val   = [i for i in range(len(mPAS_cluster_cnt)) if cv2.pointPolygonTest(mPAS_cluster_cnt[i], pos_xy, False)!=-1]
        if len(clust_val) == 0: clust_val = [-1]
        area_mPAS_i = contour_Area(mpas_cnt[ii]) 
        area_crypt  = contour_Area(crypt_ii_cnt) if len(crypt_ii_cnt) > 0 else 0
        mPAS_clone_info[ii, 0:5] = [clust_val[0] , frac_clonal, area_mPAS_i, area_crypt, crypt_indx]
    ## Collapse clones in same crypt
    mPAS_clone_info = joinClonesInSameCrypt(mPAS_clone_info)    
    return mPAS_clone_info

## Features are
## crypt_cnt   0-Area, 1-MA, 2-ma, 3-eccentricity, 4-centroid_x, 5-centroid_y, 6-indx in contour list
def getCryptFeatures(crypt_cnt):
    crypt_features = np.zeros((len(crypt_cnt),6))
    for i in range(len(crypt_cnt)):
        cnt_i             = crypt_cnt[i]
        pos_xy            = contour_xy(cnt_i)
        try:
            _, axis_ellip, _  = cv2.fitEllipse(cnt_i)
            MA = max(axis_ellip)
            ma = min(axis_ellip)
            eccentricity_i = np.sqrt(1.-((1.*ma)/MA)**2)
        except:
            MA = -1
            ma = -1
            eccentricity_i = -1
            
        crypt_features[i, 0:6] = [contour_Area(cnt_i), MA, ma, eccentricity_i, pos_xy[0], pos_xy[1]]
    return crypt_features
    

