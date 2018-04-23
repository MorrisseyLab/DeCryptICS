#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:40:37 2018

@author: edward
""" 

from sklearn.neighbors import NearestNeighbors
import numpy as np
from cnt_Feature_Functions import contour_xy, plotCntAndFeat, filterList
from cnt_Feature_Functions    import filterSmallArea, st_3, st_5, drawAllCont
from cnt_Feature_Functions    import getPercentileInts, filterSmallArea_outer, contour_Area
from cnt_Feature_Functions    import joinContoursIfClose, contour_MajorMinorAxis, contour_MajorMinorAxis, contour_mean_Area
from classContourFeat         import getAllFeatures
import cv2

def tukey_lower_thresholdval(vec, numIQR = 1.5):
    lower_quartile = np.percentile(vec, q=25)
    IQR = np.percentile(vec, q=75) - lower_quartile
    val = lower_quartile - numIQR*IQR
    return val

def tukey_upper_thresholdval(vec, numIQR = 1.5):
    upper_quartile = np.percentile(vec, q=75)
    IQR = upper_quartile - np.percentile(vec, q=25)
    val = upper_quartile + numIQR*IQR
    return val

def tukey_outliers_above(vec, numIQR = 1.5):
    upper_quartile = np.percentile(vec, q=75)
    IQR = upper_quartile - np.percentile(vec, q=25)
    inds = np.where(vec < upper_quartile + numIQR*IQR)[0]
    return inds

def tukey_outliers_below(vec, numIQR = 1.5):
    lower_quartile = np.percentile(vec, q=25)
    IQR = np.percentile(vec, q=75) - lower_quartile
    inds = np.where(vec > lower_quartile - numIQR*IQR)[0]
    return inds

def outlier_calc(var_vec, indx_val, indx_vec_val, type_compare, stddevmult = 1.6):
    val     = var_vec[indx_val]
    vec_val = var_vec[indx_vec_val]
    if type_compare == "small":
        bool_outlier = val < np.mean(vec_val) - stddevmult*np.std(vec_val)
    elif type_compare == "large":
        bool_outlier = val > np.mean(vec_val) + stddevmult*np.std(vec_val) 
    elif type_compare == "both":
        bool_outlier = val > np.mean(vec_val) + stddevmult*np.std(vec_val) or val < np.mean(vec_val) - stddevmult*np.std(vec_val)
    return(bool_outlier)

def remove_tiling_overlaps_knn(contours, nn = 2):
    if len(contours)==0: return contours, np.array([])
    # check moments arent zero or something
    all_xy = [contour_xy(cnt_i) for cnt_i in contours if not cv2.moments(cnt_i)['m00']==0]
    all_xy = np.array(all_xy)  
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(all_xy)
    distances, indices = nbrs.kneighbors(all_xy)
    throw_inds = []
    for i in range(indices.shape[0]):
        cnt = contours[i]
        compare_cnt_num = indices[i,1]
        inside_bool = cv2.pointPolygonTest(cnt, (all_xy[compare_cnt_num,0], all_xy[compare_cnt_num,1]), False)
        if (not inside_bool==-1):            
                home_area = contour_Area(cnt)
                away_area = contour_Area(contours[compare_cnt_num])
                if (home_area < away_area):
                    compare_cnt_num = i
                if (compare_cnt_num not in throw_inds):
                    throw_inds.append(compare_cnt_num)            
    fixed_contour_list = [contours[i] for i in range(len(contours)) if i not in throw_inds]
    keep_inds = np.asarray( [i for i in range(len(contours)) if i not in throw_inds] )
    return fixed_contour_list, keep_inds

def prune_contours_knn(crypt_cnt, features, nuclei_ch_raw, backgrd, smallBlur_img_nuc, stddevmult = 1.6, nn = 7):
    all_xy = [contour_xy(cnt_i) for cnt_i in crypt_cnt]
    all_xy = np.array(all_xy)  
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(all_xy)
    distances, indices = nbrs.kneighbors(all_xy)
    crypt_features = np.zeros((len(crypt_cnt), 7), dtype = np.bool)
    ## index 0 is itself
    for inx_cnt_i in range(all_xy.shape[0]):
        curr_indx  = indices[inx_cnt_i, 0]
        knn_i_indx = indices[inx_cnt_i, 1:]
        crypt_features[curr_indx, 0] = outlier_calc( features.allEcc,      curr_indx, knn_i_indx, "large", stddevmult) #large eccentricity
        crypt_features[curr_indx, 1] = outlier_calc( features.allSolid,    curr_indx, knn_i_indx, "small", stddevmult) #small solidity
        crypt_features[curr_indx, 2] = outlier_calc( features.allSizes,    curr_indx, knn_i_indx, "small", stddevmult) #small size
        crypt_features[curr_indx, 3] = outlier_calc( features.allSizes,    curr_indx, knn_i_indx, "large", stddevmult) #large size
        crypt_features[curr_indx, 4] = outlier_calc( features.allMeanNucl, curr_indx, knn_i_indx, "large", stddevmult) #large meanNuclContent
        crypt_features[curr_indx, 5] = outlier_calc( features.allHalo,     curr_indx, knn_i_indx, "small", stddevmult) #small halo
        crypt_features[curr_indx, 6] = outlier_calc( features.allHaloGap,  curr_indx, knn_i_indx, "large", stddevmult) #large halo gap
    
    # small size
#    inds = np.where(crypt_features[:,2]==False)[0]
#    crypt_cnt = np.asarray(crypt_cnt)[inds]
#    crypt_features = crypt_features[inds,:]
    # large size and large eccentricity
    inds = np.where(np.bitwise_and(crypt_features[:,3], crypt_features[:, 0])==False)[0]
    crypt_cnt = np.asarray(crypt_cnt)[inds]
    crypt_features = crypt_features[inds,:]
    # small solidity / large eccentricity
    inds = np.where(np.bitwise_and(crypt_features[:,1], crypt_features[:, 0])==False)[0]
    crypt_cnt = crypt_cnt[inds]
    crypt_features = crypt_features[inds,:]
    # small halo / large halo gap
#    inds = np.where(np.bitwise_and(crypt_features[:, 5], crypt_features[:, 6])==False)[0]
#    crypt_cnt = crypt_cnt[inds]
#    crypt_features = crypt_features[inds,:]
    crypt_cnt = list(crypt_cnt)
    return crypt_cnt

def drop_broken_runs(qq_both, med_minor_axis, nuclei_ch_raw):
    # first throw on minor axis
    minor_axis_hard_threshold = 5.*max(med_minor_axis)/8.
    indx_to_keep = np.where(np.asarray(med_minor_axis)>minor_axis_hard_threshold)[0]
    indx_to_throw = np.where(np.asarray(med_minor_axis)<minor_axis_hard_threshold)[0]
    for i in indx_to_throw:
        print("Throwing all of run %d" % i)
    qq_new = [qq_both[j] for j in indx_to_keep]
    # Now do mean nuclear content threshold
    meannucl_vec     = [np.median([contour_mean_Area(contours_i, nuclei_ch_raw) for contours_i in qq_i[0]]) for qq_i in qq_new if qq_i[0]!=[]]
    if (len(meannucl_vec)==0):
        return qq_new
    meannucl_vec_aux = [np.median([contour_mean_Area(contours_i, nuclei_ch_raw) for contours_i in qq_i[0]]) for qq_i in qq_both if qq_i[0]!=[]]
    eps = 0.03 # to account for situation where minimum is zero
    meannucl_hard_threshold = 11.*(min(meannucl_vec)+eps)/8.
    indx_to_keep = np.where(np.asarray(meannucl_vec)<meannucl_hard_threshold)[0]
    indx_to_throw = np.where(np.asarray(meannucl_vec)>meannucl_hard_threshold)[0]
    for i in indx_to_throw:
        oldindx = np.where(meannucl_vec_aux==meannucl_vec[i])[0][0]
        print("Throwing all of run %d" % oldindx)
    qq_new = [qq_new[j] for j in indx_to_keep]
    return qq_new

def drop_minoraxis_from_each_run(qq_both, nuclei_ch_raw, backgrd, smallBlur_img_nuc):
    for i in range(len(qq_both)):
        cnts = qq_both[i][0]
        features = getAllFeatures(cnts, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
        ma_inds = features.allMinorAxis > 50.
        cnts = filterList(cnts, ma_inds)
        qq_both[i] = [cnts, qq_both[i][1], qq_both[i][2], qq_both[i][3], qq_both[i][4], qq_both[i][5] ]        
    return qq_both

def prune_minoraxes(crypt_cnt, features, minor_ax_thresh):
    # prune small Minor Axes
    ma_inds = np.where(features.allMinorAxis > minor_ax_thresh)[0]
    crypt_cnt = list(np.asarray(crypt_cnt)[ma_inds])
    return crypt_cnt

def prune_halos_and_gaps(crypt_cnt, features):
    h_inds = tukey_outliers_below(features.allHalo, numIQR=2.)
    hg_inds = tukey_outliers_above(features.allHaloGap, numIQR=2.)
    comb_inds = np.unique(np.hstack([h_inds, hg_inds])) # above or below at least one limit
    #crypt_cnt = list(np.asarray(crypt_cnt)[comb_inds])
    return comb_inds

def prune_extreme_sizes(crypt_cnt, features):
    s_inds = tukey_outliers_below(features.allSizes, numIQR=3.)
    l_inds = tukey_outliers_above(features.allSizes, numIQR=3.)
    comb_inds = np.intersect1d(s_inds, l_inds) # between both limits
    #crypt_cnt = list(np.asarray(crypt_cnt)[comb_inds])
    return comb_inds
    
def prune_individual_attributes(crypt_cnt, features):
    # Make this leniant -- then do knn pruning
    # prune small solidities < 0.55
    s_inds = np.where(features.allSolid > 0.4)[0]
    # prune large halo gaps > 0.15
    hg_inds = np.where(features.allHaloGap < 0.5)[0]
    # prune giant eccentricities
    e_inds = np.where(features.allEcc < 0.965)[0]    
    
    comb_inds = np.intersect1d(s_inds, hg_inds)
    comb_inds = np.intersect1d(comb_inds, e_inds)
    #crypt_cnt = list(np.asarray(crypt_cnt)[comb_inds])
    return comb_inds

def prune_attributes(crypt_cnt, features):
    if (features.allHalo.shape[0]==0):
        return crypt_cnt
    else:
        inds1 = prune_halos_and_gaps(crypt_cnt, features)
        inds2 = prune_extreme_sizes(crypt_cnt, features)
        inds3 = prune_individual_attributes(crypt_cnt, features)
        comb_inds = np.intersect1d(inds1, inds2)
        comb_inds = np.intersect1d(comb_inds, inds3)
        crypt_cnt = list(np.asarray(crypt_cnt)[comb_inds])
    return crypt_cnt
    
def prune_combination_attributes(crypt_cnt, features):
    comb_inds = np.where(np.bitwise_and(features.allSolid > 0.7, 
                            np.bitwise_and(features.allMinorAxis > 55, 
                               np.bitwise_or(features.allHalo > 0.55, 
                                 features.allHaloGap < 0.15))))[0]
    #comb_inds = np.unique(h_inds)
    crypt_cnt = list(np.asarray(crypt_cnt)[comb_inds])
    return crypt_cnt
