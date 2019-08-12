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

def check_length(contours):
   if (type(contours)==list): return contours
   if (type(contours)==np.ndarray): return [contours] # fix single contours un-listing themselves (might not be needed)
   else: return 99 # error

def remove_tiling_overlaps_knn(contours, nn=4):
   ## sanity check
   numconts = len(contours)
   if numconts==0: return contours, np.array([])
   if numconts==1: return contours, np.array([0])
   nn = min(numconts,  nn)
   
   # check moments arent zero or something
   throw_inds = [i for i in range(len(contours)) if cv2.moments(contours[i])['m00']==0]
   contours = [contours[i] for i in range(len(contours)) if not i in throw_inds]    
   distances, indices, all_xy, _ = nn2(contours, contours, nn)
   inside_compare = np.zeros(indices.shape)
   for i in range(indices.shape[0]):
      for j in range(1,indices.shape[1]):
         ii = indices[i,0]
         jj = indices[i,j]
         inside_compare[i,j] = cv2.pointPolygonTest(contours[ii], (all_xy[jj,0], all_xy[jj,1]), False)
   insides = np.asarray(np.where(inside_compare[:,1:]>=0)).T
   insides[:,1] = insides[:,1] + 1 # shift as ignored `self' column in above test
   for pair in insides:
      ii = indices[pair[0],0]
      jj = indices[pair[0],pair[1]]
      if (ii not in throw_inds and jj not in throw_inds):
         aA = contour_Area(contours[ii])
         aB = contour_Area(contours[jj])
         if (aA < aB):
            throw_inds.append(ii)
         else:
            throw_inds.append(jj)
   keep_inds = np.asarray( [i for i in range(len(contours)) if i not in throw_inds] )   
   fixed_contour_list = [contours[i] for i in range(len(contours)) if i not in throw_inds]
   return check_length(fixed_contour_list), keep_inds

def nn2(dat_to, dat_from, nn=4):
   ## sanity check
   dat_from = check_length(dat_from)
   dat_to = check_length(dat_to)
   nn = min(len(dat_to),  nn)
   ## Construct knn
   all_xy_crypt    = np.array([contour_xy(cnt_i) for cnt_i in dat_to if not cv2.moments(cnt_i)['m00']==0])
   all_xy_target   = np.array([contour_xy(cnt_i) for cnt_i in dat_from if not cv2.moments(cnt_i)['m00']==0])
   nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(all_xy_crypt)
   distances, indices = nbrs.kneighbors(all_xy_target) # shape (len(dat_from) x nn), hence nn<=len(dat_to)
   return distances, indices, all_xy_crypt, all_xy_target

def inside_comparison_cryptin(indices, dat_to, xy_from):
   if len(indices.shape)==1: indices = indices.reshape(indices.shape[0],1)      
   inside_compare = np.zeros(indices.shape)
   for i in range(indices.shape[0]):
      for j in range(indices.shape[1]):
         jj = indices[i,j] # crypt index
         inside_compare[i,j] = cv2.pointPolygonTest(dat_to[i], (xy_from[jj,0], xy_from[jj,1]), False)
   return inside_compare

def inside_comparison_incrypt(indices, dat_to, xy_from):
   inside_compare = np.zeros(indices.shape)
   for i in range(indices.shape[0]):
      for j in range(indices.shape[1]):
         jj = indices[i,j] # crypt index
         inside_compare[i,j] = cv2.pointPolygonTest(dat_to[jj], (xy_from[i,0], xy_from[i,1]), False)
   return inside_compare
  
def crypt_indexing_fufi(contours, target_overlay, nn=4, crypt_dict={}):
   # empty check
   if (len(target_overlay)==0):
      return contours, [], crypt_dict
   target_overlay = check_length(target_overlay)
   contours = check_length(contours)
   ## form knn with fufis
   distances, indices, all_xy_crypt, all_xy_target = nn2(contours, target_overlay, nn)
   inside_compare = inside_comparison_cryptin(indices, target_overlay, all_xy_crypt)
   # join crypts found inside same fufi; cull fufis that don't contain any crypts;
   # extend fufis that contain the second closest crypt but not the first
   cryptinds_fufis = []
   cryptcnt_joined = []
   fufis_to_throw = []
   if (indices.shape[1]>1):
      for i in range(indices.shape[0]):
         if (inside_compare[i,0]>=0 and inside_compare[i,1]>=0):
            # when two crypts are inside a fufi
            c = 0
            cryptinds_f = []         
            while (inside_compare[i,c]>=0): # check for more crypts
               cryptinds_f.append(indices[i,c])
               c += 1
               if (c >= inside_compare.shape[1]): break 
            cont = np.vstack([np.array(contours[i]) for i in cryptinds_f])
            hull = cv2.convexHull(cont)
            cryptcnt_joined.append(hull)
            cryptinds_fufis += cryptinds_f # note crypt indices to be overwritten
         if (inside_compare[i,0]<0 and inside_compare[i,1]>=0):
            # when second crypt is inside a fufi but first is not
            cont = np.vstack([np.array(target_overlay[i]), np.array(contours[indices[i,1]])])
            hull = cv2.convexHull(cont)
            target_overlay[i] = hull # overwrite fufi contour
         if (inside_compare[i,0]<0 and inside_compare[i,1]<0):
            fufis_to_throw.append(i)
   # remove crypts that should be joined
   fixed_contour_list = [contours[i] for i in range(len(contours)) if i not in cryptinds_fufis]
   # add joined crypts
   fixed_contour_list += cryptcnt_joined
   # throw bad fufis
   fixed_fufi_list = [target_overlay[i] for i in range(len(target_overlay)) if i not in fufis_to_throw]
   crypt_dict["fufi_label"] = np.zeros(len(fixed_contour_list))
   if (len(fixed_fufi_list)>0):
      # re-define labelling for crypt indexing
      distances, indices, all_xy_crypt, all_xy_target = nn2(fixed_contour_list, fixed_fufi_list, nn)
      inside_compare = inside_comparison_cryptin(indices, fixed_fufi_list, all_xy_crypt)   
#      crypt_dict["crypt_xy"] = all_xy_crypt
      for jj in range(indices.shape[0]):
         if (inside_compare[jj,0]>=0): crypt_dict["fufi_label"][indices[jj,0]] = 1
#   else:
#      crypt_dict["crypt_xy"] = np.array([contour_xy(cnt_i) for cnt_i in fixed_contour_list if not cv2.moments(cnt_i)['m00']==0])
   return check_length(fixed_contour_list), check_length(fixed_fufi_list), crypt_dict

def join_clones_in_fufi(contours, target_overlay, nn=4):
   ## sanity check
   numclones = len(contours)
   target_overlay = check_length(target_overlay)
   contours = check_length(contours)
   if (numclones<2):
      return(contours)
   else:
      nn = min(numclones,  nn)
   ## form knn with fufis
   distances, indices, all_xy_clone, all_xy_target = nn2(contours, target_overlay, nn)
   inside_compare = inside_comparison_cryptin(indices, target_overlay, all_xy_clone)
   # join clones found inside same fufi
   cloneinds_fufis = []
   clonecnt_joined = []
   for i in range(indices.shape[0]):
      if (nn > 2):
         if (inside_compare[i,0]>=0 and inside_compare[i,1]>=0):
            # when two clone are inside the same fufi
            c = 0
            cloneinds_f = []         
            while (inside_compare[i,c]>=0): # check for more clones
               cloneinds_f.append(indices[i,c])
               c += 1
               if (c >= inside_compare.shape[1]): break 
            cont = np.vstack([np.array(contours[i]) for i in cloneinds_f])
            hull = cv2.convexHull(cont)
            clonecnt_joined.append(hull)
            cloneinds_fufis += cloneinds_f # note clone indices to be overwritten
   # remove crypts that should be joined
   fixed_contour_list = [contours[i] for i in range(len(contours)) if i not in cloneinds_fufis]
   # add joined crypts
   fixed_contour_list += clonecnt_joined
   return check_length(fixed_contour_list)
      
def crypt_indexing_clone(crypt_contours, target_overlay, nn=1, crypt_dict={}):
   crypt_contours = check_length(crypt_contours)
   num_repeats = 1
   while num_repeats>0:
      ## sanity check
      target_overlay = check_length(target_overlay)      
      ## label crypts as clones
      clone_inds = []
      if len(target_overlay)>0:
         distances, indices, all_xy_crypt, all_xy_target = nn2(crypt_contours, target_overlay, nn)
         inside_compare1 = inside_comparison_incrypt(indices, crypt_contours, all_xy_target)
         inside_compare2 = inside_comparison_cryptin(indices, target_overlay, all_xy_crypt)
         for i in range(indices.shape[0]):
            if (inside_compare1[i,0]>=0 or inside_compare2[i,0]>=0):
               clone_inds.append(indices[i,0])
      # account for the fact that two clones can reside in the same crypt
      u_inds, u_counts = np.unique(clone_inds, return_counts=True)
      repeats = np.where(u_counts>1)[0]
      num_repeats = len(repeats)
      rmv_cnts = []
      add_cnts = []
      for r in repeats:
         rinds = np.where(indices[:,0]==u_inds[r])[0]
         cr_area = contour_Area(crypt_contours[u_inds[r]])
         cl_area = [cr_area]
         for rc in rinds:
            cl_area.append(contour_Area(target_overlay[rc]))
         maxarea = np.argmax(cl_area)
         for rc in rinds: rmv_cnts.append(rc)
         if maxarea==0:
            # take crypt contour
            add_cnts.append(crypt_contours[u_inds[r]])
         else:
            # take one of clone contours
            add_cnts.append(target_overlay[rinds[maxarea-1]])
      # get rid of unneeded contours
      keep_inds = np.setdiff1d(np.array(range(len(target_overlay))), rmv_cnts)
      target_overlay = [target_overlay[i] for i in keep_inds]
      # add remaining contours
      for cnt in add_cnts: target_overlay.append(cnt)         
   crypt_dict["clone_label"] = np.zeros(len(crypt_contours))
   for ind in clone_inds: crypt_dict["clone_label"][ind] = 1
   return crypt_dict

def get_crypt_patchsizes_and_ids(patch_indices, crypt_dict):
   # gives mutant patches of size > 1 a unique ID.  Single mutants have ID = 0
   crypt_dict["patch_size"] = np.zeros(len(crypt_dict["clone_label"]))
   crypt_dict["patch_id"] = np.zeros(len(crypt_dict["clone_label"]))
   #for patch in patch_indices:
   for i in range(1,len(patch_indices)+1):
      patch = patch_indices[i-1]
      for index in patch:
         crypt_dict["patch_size"][index] = len(patch)
         crypt_dict["patch_id"][index] = i
   return crypt_dict

## Outlier calculations
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

## knn pruning functions      
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
