# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:18:27 2015

@author: edward
"""
import cv2
import numpy as np
import matplotlib.pylab as plt
import itertools
from deconv_mat            import *
from MiscFunctions         import *
from cnt_Feature_Functions import *
from classContourFeat      import getAllFeatures
from knn_prune             import tukey_lower_thresholdval, tukey_upper_thresholdval

def find_clone_statistics(crypt_cnt, img_nuc, img_clone, nbins = 20):
   # for each contour do no_threshold_signal_collating()
   numcnts = len(crypt_cnt)
   xy_coords = np.zeros([numcnts, 2], dtype=np.int32)
   halo_n = np.zeros([numcnts])
   halo_c = np.zeros([numcnts])
   out_av_sig_nucl  = np.zeros([numcnts, nbins], dtype=np.float32)
   out_av_sig_clone = np.zeros([numcnts, nbins], dtype=np.float32)
   in_av_sig_nucl   = np.zeros([numcnts, nbins], dtype=np.float32)
   in_av_sig_clone  = np.zeros([numcnts, nbins], dtype=np.float32)
   for i in range(numcnts):
      contour = crypt_cnt[i]
      X = no_threshold_signal_collating(contour, img_nuc, img_clone, nbins)
      halo_n[i]              = X[0]
      halo_c[i]              = X[1]
      out_av_sig_nucl[i, :]  = X[2]
      out_av_sig_clone[i, :] = X[3]
      xy_coords[i,0]         = X[4][0]
      xy_coords[i,1]         = X[4][1]
      in_av_sig_nucl[i, :]   = X[5]
      in_av_sig_clone[i, :]  = X[6]
   clone_feature_list = (halo_n, halo_c, out_av_sig_nucl, out_av_sig_clone, xy_coords, in_av_sig_nucl, in_av_sig_clone)
   return clone_feature_list

def determine_clones(clone_feature_list, clonal_mark_type):
   halo_n = clone_feature_list[0]
   halo_c = clone_feature_list[1]
   out_av_sig_nucl = clone_feature_list[2]
   out_av_sig_clone = clone_feature_list[3]
   xy_coords = clone_feature_list[4]
   in_av_sig_nucl = clone_feature_list[5]
   in_av_sig_clone = clone_feature_list[6]
   # Calculate outliers of each bin
   out_outlier_nucl_vecs  = outlier_vec_calculator(out_av_sig_nucl)
   out_outlier_clone_vecs = outlier_vec_calculator(out_av_sig_clone)
   in_outlier_nucl_vecs   = outlier_vec_calculator(in_av_sig_nucl)
   in_outlier_clone_vecs  = outlier_vec_calculator(in_av_sig_clone)
   if (clonal_mark_type=="P" or clonal_mark_type=="PNN" or clonal_mark_type=="BP"):
      out_outlier_clone_vec = out_outlier_clone_vecs[1] # big outlier
      out_outlier_nucl_vec  = out_outlier_nucl_vecs[1] # big outlier
      in_outlier_clone_vec = in_outlier_clone_vecs[1] # big outlier
      in_outlier_nucl_vec  = in_outlier_nucl_vecs[1] # big outlier
   if (clonal_mark_type=="N" or clonal_mark_type=="NNN" or clonal_mark_type=="BN"):
      out_outlier_clone_vec = out_outlier_clone_vecs[0] # small outlier
      out_outlier_nucl_vec  = out_outlier_nucl_vecs[0] # small outlier
      in_outlier_clone_vec = in_outlier_clone_vecs[0] # small outlier
      in_outlier_nucl_vec  = in_outlier_nucl_vecs[0] # small outlier      
   # Determine signal width for clones
   clone_signal_width = []
   if (clonal_mark_type=="BN" or clonal_mark_type=="BP"):
      clone_signal_width_in = []
      clone_signal_width_out = []
   for i in range(out_av_sig_nucl.shape[0]):
      if (clonal_mark_type=="P" or clonal_mark_type=="N"):
         clone_signal_width.append(signal_width(out_av_sig_nucl[i, :], out_outlier_nucl_vec, out_av_sig_clone[i,:], out_outlier_clone_vec, clonal_mark_type))
      if (clonal_mark_type=="PNN" or clonal_mark_type=="NNN"):
         clone_signal_width.append(signal_width(in_av_sig_nucl[i, :], in_outlier_nucl_vec, in_av_sig_clone[i,:], in_outlier_clone_vec, clonal_mark_type))
      if (clonal_mark_type=="BN" or clonal_mark_type=="BP"):
         clone_signal_width_out.append(signal_width(out_av_sig_nucl[i, :], out_outlier_nucl_vec, out_av_sig_clone[i,:], out_outlier_clone_vec, clonal_mark_type))
         clone_signal_width_in.append(signal_width(in_av_sig_nucl[i, :]  , in_outlier_nucl_vec , in_av_sig_clone[i,:] , in_outlier_clone_vec , clonal_mark_type))
         clone_signal_width = np.minimum(np.asarray(clone_signal_width_out) , np.asarray(clone_signal_width_in))
   inds = np.where(np.asarray(clone_signal_width)>0.1)[0] # throw away weak signals
   return clone_signal_width
   
def no_threshold_signal_collating(cnt_i, img_nuc, img_clone, nbins = 20):
   # Find max halo contour in nuclear channel
   nucl_halo, clone_halo, output_cnt = max_halocnt_nucl_clone(cnt_i, img_nuc, img_clone)
   # Bin length of contour into ~20 bins; average pixel intensity in each bin in both nuclear and clonal channel
   av_sig_nucl = bin_intensities_flattened(output_cnt, img_nuc, nbins)
   av_sig_clone = bin_intensities_flattened(output_cnt, img_clone, nbins)
   # Save along with (x,y) coordinate of centre of crypt   
   M = cv2.moments(output_cnt)
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   centre_xy = (cX, cY)
   # Repeat for a contour that is slightly eroded to look for inside signal
   inner_cnt = extractInnerRingContour(cnt_i, img_nuc, 1)
   inav_sig_nucl = bin_intensities_flattened(inner_cnt, img_nuc, nbins)
   inav_sig_clone = bin_intensities_flattened(inner_cnt, img_clone, nbins)
   return nucl_halo, clone_halo, av_sig_nucl, av_sig_clone, centre_xy, inav_sig_nucl, inav_sig_clone

def max_halocnt_nucl_clone(cnt_i, img_nuc, img_clone):
    # Max and min halo size to calculate
    start_diff = 1 # min diff to check 
    end_diff   = 8 # end_diff -1 max diff to check
    # Expand box
    expand_box    = 100
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_roi       = cnt_i - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img_nuc[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    img_ROI_c     = img_clone[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)    
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1) ## Get mask
    max_morphs    = 15
    sum_morphs    = np.zeros(max_morphs+1)
    areas_morphs  = np.zeros(max_morphs+1)
    # Area and sum pre-dilations
    areas_morphs[0] = cv2.countNonZero(mask_fill1)
    sum_morphs[0]   = cv2.mean(img_ROI, mask_fill1)[0]/255. * areas_morphs[0]
    for i in range(1, max_morphs+1):
      mask_fill1    = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = 1)
      areas_morphs[i]  = cv2.countNonZero(mask_fill1)
      sum_morphs[i]    = cv2.mean(img_ROI, mask_fill1)[0]/255. * areas_morphs[i]
    z_ind = np.where(areas_morphs==0)[0]
    if (z_ind.shape[0]>0):
       areas_morphs = areas_morphs[:z_ind[0]]
       sum_morphs = sum_morphs[:z_ind[0]]
    # Finding maximum halo
    num_diffs  = end_diff-start_diff
    max_each = np.zeros(num_diffs)
    indices = []
    for diff_size, ii in zip(range(start_diff,end_diff), range(num_diffs)):
      if (not (len(sum_morphs)-diff_size) <= 0):
        indx_1    = range(diff_size,len(sum_morphs))
        indx_2    = range(0,len(sum_morphs)-diff_size)
        halo_mean = (sum_morphs[indx_1] - sum_morphs[indx_2])/(areas_morphs[indx_1] - areas_morphs[indx_2] + 1e-5)
        max_each[ii] = np.max(halo_mean)
        maxindx = np.where(halo_mean==max_each[ii])[0][0]        
        morph_pair = (indx_1[maxindx], indx_2[maxindx])
        indices.append(morph_pair)
    nucl_halo = np.max(max_each)
    maxindx_global = np.where(max_each==nucl_halo)[0][0]
    morph_pair_m = indices[maxindx_global]
    clone_halo = extractCloneSignal(cnt_roi, img_ROI_c, morph_pair_m)
    maxmiddlecontour = int( np.ceil( (morph_pair_m[0]+morph_pair_m[1])/2. ) )
    mid_halo_cnt = extractRingContour(cnt_roi, img_ROI, maxmiddlecontour, Start_ij_ROI)
    output_cnt = mid_halo_cnt.astype(np.int32)
    return nucl_halo, clone_halo, output_cnt
 
def extractCloneSignal(cnt_roi, img_ROI_c, morph_pair):
    mask_fill1    = np.zeros(img_ROI_c.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1)
    sum_morphs    = np.zeros(2)
    areas_morphs  = np.zeros(2)
    for i in range(2):
       mask_fill1      = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = morph_pair[i])
       areas_morphs[i] = cv2.countNonZero(mask_fill1)
       sum_morphs[i]   = cv2.mean(img_ROI_c, mask_fill1)[0]/255. * areas_morphs[i]
    clone_halo_mean = (sum_morphs[0] - sum_morphs[1])/(areas_morphs[0] - areas_morphs[1] + 1e-5)
    return clone_halo_mean  

def extractRingContour(cnt_roi, img_ROI, num_morphs, Start_ij_ROI):
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1)
    mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_3, iterations = num_morphs)
    halo_cnt, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    num_hal = len(halo_cnt)
    if (num_hal==0):
        return (np.asarray(cnt_roi) + Start_ij_ROI).astype(np.int32)
    elif (num_hal==1):
        return ((np.asarray(halo_cnt) + Start_ij_ROI).astype(np.int32))[0]
    else:        
        areas = []
        for i in range(num_hal):
            areas.append(contour_Area(halo_cnt[i]))
        maxarea = np.where(areas==np.max(areas))[0][0]
        return ((np.asarray(halo_cnt[maxarea]) + Start_ij_ROI).astype(np.int32))

def extractInnerRingContour(cnt_i, img, num_morphs):
    expand_box    = 100
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_roi       = cnt_i - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1)
    mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_CLOSE, st_5, iterations = num_morphs) # get rid of inner black blobs
    mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_ERODE, st_5, iterations = num_morphs)
    halo_cnt, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    num_hal = len(halo_cnt)
    if (num_hal==0):        
        return cnt_i.astype(np.int32)
    elif (num_hal==1):
        return ((np.asarray(halo_cnt) + Start_ij_ROI).astype(np.int32))[0]
    else:        
        areas = []
        for i in range(num_hal):
            areas.append(contour_Area(halo_cnt[i]))
        maxarea = np.where(areas==np.max(areas))[0][0]
        return ((np.asarray(halo_cnt[maxarea]) + Start_ij_ROI).astype(np.int32))

def bin_intensities_flattened(output_cnt, img1, nbins = 20):
   roi           = cv2.boundingRect(output_cnt)
   Start_ij_ROI  = roi[0:2] # get x,y of bounding box
   cnt_j       = output_cnt - Start_ij_ROI # change coords to start from x,y
   img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] # note here the use of y coord first!
   flat_cnt = []
   for xy_i in cnt_j[:,0,:]:
      flat_cnt.append(img_ROI[xy_i[1],xy_i[0]])
   numpixels = len(flat_cnt)
   overhang = numpixels % nbins
   normal_bin_width = numpixels // nbins
   #(nbins - overhang) * normal_bin_width + overhang*(normal_bin_width + 1) == numpixles
   av_sig = np.zeros(nbins)
   cw = normal_bin_width
   for i in range(nbins-overhang):
      av_sig[i] = np.mean(flat_cnt[i*cw : (i+1)*cw])
   done = (nbins-overhang)*normal_bin_width
   cw = normal_bin_width + 1
   for i in range(overhang):
      av_sig[i + nbins-overhang] = np.mean(flat_cnt[done + i*cw : done + (i+1)*cw])
   return av_sig
   
def outlier_vec_calculator(av_sig_mat, numIQR = 1.25):
   big_outlier = np.zeros(av_sig_mat.shape[1])
   small_outlier = np.zeros(av_sig_mat.shape[1])
   for j in range(av_sig_mat.shape[1]):
      small_outlier[j] = tukey_lower_thresholdval(av_sig_mat[:, j], numIQR) #np.maximum(tukey_lower_thresholdval(av_sig_mat[:, j], numIQR), 20)
      big_outlier[j] = tukey_upper_thresholdval(av_sig_mat[:, j], numIQR) #np.minimum(tukey_upper_thresholdval(av_sig_mat[:, j], numIQR), 235)
   return small_outlier, big_outlier
   
def signal_width(av_sig_nucl_vec, outlier_nucl_vec, av_sig_clone_vec, outlier_clone_vec, clonal_mark_type):
   if (clonal_mark_type=="N" or clonal_mark_type=="NNN" or clonal_mark_type=="BN"):
      n_tf = av_sig_nucl_vec < outlier_nucl_vec
      c_tf = av_sig_clone_vec < outlier_clone_vec
   if (clonal_mark_type=="P" or clonal_mark_type=="PNN" or clonal_mark_type=="BP"):
      n_tf = av_sig_nucl_vec > outlier_nucl_vec
      c_tf = av_sig_clone_vec > outlier_clone_vec   
   clone_trues = np.bitwise_and(c_tf, np.bitwise_not(n_tf))
   wedges = [ list(x[1]) for x in itertools.groupby(clone_trues, lambda x: x == True) if x[0] ]   
   if (len(wedges)==0):
      return 0
   if (len(wedges)>1):
      # join loop
      if (clone_trues[0]==True and clone_trues[-1]==True):
         wedges[0] = wedges[-1]+wedges[0]
         wedges = wedges[:-1]
   maxwedge = 0
   for i in range(len(wedges)):
      wedge = wedges[i]
      if (len(wedge) > maxwedge):
         maxwedge, ind = len(wedge), i
   normed_wedge = maxwedge/len(clone_trues)
   return normed_wedge   


'''
def retrieve_clone_nuclear_features(crypt_cnt, nuclei_ch_raw, clone_ch_raw, backgrd, smallBlur_img_nuc):
    nuc_feat = getAllFeatures(crypt_cnt, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
    clone_feat = getAllFeatures(crypt_cnt, clone_ch_raw, backgrd, smallBlur_img_nuc)
    frac_halo = clone_feat.allHalo/(nuc_feat.allHalo+0.01)
    frac_halogap = (1.+clone_feat.allHaloGap)/(1.+nuc_feat.allHaloGap)
    clone_content = clone_feat.allMeanNucl
    return frac_halo, frac_halogap, clone_content

def find_clones(crypt_cnt, clone_channel_feats, clone_marker_type, numIQR=2):                        
    frac_halo = clone_channel_feats[0]
    frac_halogap = clone_channel_feats[1]
    clone_content = clone_channel_feats[2]
    if (clone_marker_type=="P"):
        inds1 = np.where(frac_halo > tukey_upper_thresholdval(frac_halo, numIQR = numIQR))[0]
        inds2 = np.where(frac_halogap < tukey_lower_thresholdval(frac_halogap, numIQR = numIQR))[0]
        inds = np.intersect1d(inds1,inds2)
        full_partial_statistic = (frac_halo[inds] + (2 - frac_halogap[inds]))/2.
    if (clone_marker_type=="N"):
        inds1 = np.where(frac_halo < tukey_lower_thresholdval(frac_halo, numIQR = numIQR))[0]
        inds2 = np.where(frac_halogap > tukey_upper_thresholdval(frac_halogap, numIQR = numIQR))[0]
        inds = np.intersect1d(inds1,inds2)
        full_partial_statistic = ((1 - frac_halo[inds]) + (frac_halogap[inds]-1))/2.
    if (clone_marker_type=="PNN"):
        inds = np.where(clone_content > tukey_upper_thresholdval(clone_content, numIQR = numIQR))[0]
        inds = np.asarray([ii for ii in inds if clone_content[ii] > 1e-2]) # flush out junk contours
        full_partial_statistic = clone_content[inds]
    if (clone_marker_type=="NNN"):
        inds = np.where(clone_content < tukey_lower_thresholdval(clone_content, numIQR = numIQR))[0]
        full_partial_statistic = 1 - clone_content[inds]
    clone_cnt = list(np.asarray(crypt_cnt)[inds])
    return clone_cnt, full_partial_statistic
'''
'''
def retrieve_clone_nuclear_features(crypt_cnt, img, thresh, clonal_mark_type):
    if (clonal_mark_type=="P"): deconv_mat = deconv_mat_KDM6A
    if (clonal_mark_type=="N"): deconv_mat = deconv_mat_KDM6A
    if (clonal_mark_type=="PNN"): deconv_mat = deconv_mat_MPAS
    if (clonal_mark_type=="NNN"): deconv_mat = deconv_mat_MAOA
    nuclear_channel, _ , clone_channel = col_deconvol_and_blur(img, deconv_mat, (11,11), (11,11), (11,11))
    _, clone_ch_thresh = cv2.threshold(clone_channel, thresh[1], 255, cv2.THRESH_BINARY)
    _, nuc_ch_thresh = cv2.threshold(clone_channel, thresh[0], 255, cv2.THRESH_BINARY)
    
    clone_channel_feats = find_clone_features(crypt_cnt, clone_ch_thresh, clonal_mark_type)
    nuclear_channel_feats = find_clone_features(crypt_cnt, nuc_ch_thresh, clonal_mark_type)
    return clone_channel_feats

def find_clonesnew(crypt_cnt, clone_channel_feats, nuclear_channel_feats, clonal_mark_type, numIQR = 3):
   halo_signal = clone_channel_feats[0]
   wedge_signal = clone_channel_feats[1]
   halo_signal_n = nuclear_channel_feats[0]
   wedge_signal_n = nuclear_channel_feats[1]
   if (clonal_mark_type=="P"):
      inds1 = np.where(halo_signal > tukey_upper_thresholdval(halo_signal, numIQR = numIQR))[0]
      inds2 = np.where(wedge_signal > tukey_upper_thresholdval(wedge_signal, numIQR = numIQR))[0]
      inds = np.intersect1d(inds1,inds2)
      full_partial_statistic = np.minimum(np.maximum(wedge_signal[inds] , halo_signal[inds]), np.ones([inds.shape[0]]))
   if (clonal_mark_type=="N"):
      inds1 = np.where(halo_signal < tukey_lower_thresholdval(halo_signal, numIQR = numIQR))[0]
      inds2 = np.where(wedge_signal > tukey_upper_thresholdval(wedge_signal, numIQR = numIQR))[0]
      inds = np.intersect1d(inds1,inds2)
      full_partial_statistic = np.minimum(np.maximum(wedge_signal[inds] , (1-halo_signal[inds])), np.ones([inds.shape[0]]))
   if (clonal_mark_type=="PNN"):
      inds1 = np.where(halo_signal > tukey_upper_thresholdval(halo_signal, numIQR = numIQR))[0]
      inds2 = np.where(wedge_signal > tukey_upper_thresholdval(wedge_signal, numIQR = numIQR))[0]
      inds2 = np.asarray([ii for ii in inds2 if wedge_signal[ii] > 3e-2])
      inds = np.intersect1d(inds1,inds2)
      full_partial_statistic = np.minimum(np.maximum(wedge_signal[inds] , halo_signal[inds]), np.ones([inds.shape[0]]))
   if (clonal_mark_type=="NNN"):
      inds1 = np.where(halo_signal < tukey_lower_thresholdval(halo_signal, numIQR = numIQR))[0]
      inds2 = np.where(wedge_signal > tukey_upper_thresholdval(wedge_signal, numIQR = numIQR))[0]
      inds2 = np.asarray([ii for ii in inds2 if wedge_signal[ii] > 3e-2])
      inds = np.intersect1d(inds1,inds2)
      full_partial_statistic = np.minimum(np.maximum(wedge_signal[inds] + (1-halo_signal[inds])), np.ones([inds.shape[0]]))
   clone_cnt = list(np.asarray(crypt_cnt)[inds])
   return clone_cnt, full_partial_statistic

def find_clone_features(crypt_cnt, img1, clonal_mark_type):
   numcnts = len(crypt_cnt)
   halo_signal = np.zeros([numcnts])
   wedge_signal = np.zeros([numcnts])
   for i in range(numcnts):
      halo_signal[i], output_cnt = calc_max_halo_signal(crypt_cnt[i], img1, clonal_mark_type)
      wedge_signal[i] = calc_max_clone_wedge(output_cnt, img1, clonal_mark_type)
   return halo_signal, wedge_signal

def calc_max_halo_signal(cnt_i, img1, clonal_mark_type):
    # Max and min halo size to calculate
    start_diff = 1 # min diff to check 
    end_diff   = 8 # end_diff -1 max diff to check
    # Expand box
    expand_box    = 100
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_roi       = cnt_i - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1) ## Get mask     
    max_morphs    = 15
    img_plot      = img_ROI.copy()
    sum_morphs    = np.zeros(max_morphs+1)
    areas_morphs  = np.zeros(max_morphs+1)
    # Area and sum pre-dilations
    areas_morphs[0] = cv2.countNonZero(mask_fill1)
    sum_morphs[0]   = cv2.mean(img_ROI, mask_fill1)[0]/255. * areas_morphs[0]
    for i in range(1, max_morphs+1):
      if (clonal_mark_type=="PNN" or clonal_mark_type=="NNN"):
         mask_fill1    = cv2.morphologyEx(mask_fill1, cv2.MORPH_ERODE, st_5, iterations = 1)
      if (clonal_mark_type=="P" or clonal_mark_type=="N"):
         mask_fill1    = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = 1)
      areas_morphs[i]  = cv2.countNonZero(mask_fill1)
      sum_morphs[i]    = cv2.mean(img_ROI, mask_fill1)[0]/255. * areas_morphs[i]
    z_ind = np.where(areas_morphs==0)[0]
    if (z_ind.shape[0]>0):
       areas_morphs = areas_morphs[:z_ind[0]]
       sum_morphs = sum_morphs[:z_ind[0]]
    num_diffs  = end_diff-start_diff
    max_each = np.zeros(num_diffs)
    indices = []
    for diff_size, ii in zip(range(start_diff,end_diff), range(num_diffs)):
      if (not (len(sum_morphs)-diff_size) <= 0):
        indx_1    = range(diff_size,len(sum_morphs))
        indx_2    = range(0,len(sum_morphs)-diff_size)
        if (clonal_mark_type=="PNN" or clonal_mark_type=="NNN"):
           halo_mean = (sum_morphs[indx_2] - sum_morphs[indx_1])/(areas_morphs[indx_2] - areas_morphs[indx_1] + 1e-5)
        if (clonal_mark_type=="P" or clonal_mark_type=="N"):
           halo_mean = (sum_morphs[indx_1] - sum_morphs[indx_2])/(areas_morphs[indx_1] - areas_morphs[indx_2] + 1e-5)
        max_each[ii] = np.max(halo_mean)
        maxindx = np.where(halo_mean==max_each[ii])[0][0]        
        middle_contour_number = (indx_1[maxindx]+indx_2[maxindx])/2.
        indices.append(middle_contour_number)         
    maxhalo = np.max(max_each)
    maxindx_global = np.where(max_each==maxhalo)[0][0]
    maxmiddlecontour = int(np.ceil(indices[maxindx_global]))
    mid_halo_cnt = extractRingContour(cnt_i, img1, maxmiddlecontour, clonal_mark_type)
    if (len(mid_halo_cnt.shape)==len(cnt_i.shape)):
       output_cnt = mid_halo_cnt.astype(np.int32)
    else:
      ind_pair = np.where(np.asarray(mid_halo_cnt.shape) == 2)[-1][0]
      ind_numpnts = np.where(np.asarray(mid_halo_cnt.shape) > 2)[0][0]
      output_cnt = np.zeros([mid_halo_cnt.shape[ind_numpnts], 1, mid_halo_cnt.shape[ind_pair]], dtype=np.int32)
      for ii in range(mid_halo_cnt.shape[ind_numpnts]):
         if (ind_numpnts==1 and ind_pair==3):
            output_cnt[ii, 0, :] = mid_halo_cnt[0, ii, 0, :]
         if (ind_numpnts==0 and ind_pair==3):
            output_cnt[ii, 0, :] = mid_halo_cnt[ii, 0, 0, :]
         if (ind_numpnts==0 and ind_pair==2):
            output_cnt[ii, 0, :] = mid_halo_cnt[ii, 0, :, 0]
    return maxhalo, output_cnt
    
def calc_max_clone_wedge(output_cnt, img1, clonal_mark_type, noise_lim=255./2.):
   roi           = cv2.boundingRect(output_cnt)
   Start_ij_ROI  = roi[0:2] # get x,y of bounding box
   cnt_j       = output_cnt - Start_ij_ROI # change coords to start from x,y
   img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] # note here the use of y coord first!
   flat_cnt = []
   for xy_i in cnt_j[:,0,:]:
      flat_cnt.append(img_ROI[xy_i[1],xy_i[0]])
   if (clonal_mark_type=="P" or clonal_mark_type=="PNN"):
      wedges = [ list(x[1]) for x in itertools.groupby(flat_cnt, lambda x: x < noise_lim+0.01) if not x[0] ]
   if (clonal_mark_type=="N" or clonal_mark_type=="NNN"):
      wedges = [ list(x[1]) for x in itertools.groupby(flat_cnt, lambda x: x > noise_lim) if not x[0] ]
   if (len(wedges)==0):
      return 0
   if (len(wedges)>1):
      # join loop
      if (wedges[-1][-1] == flat_cnt[-1]):
         wedges[0] = wedges[-1]+wedges[0]
         wedges = wedges[:-1]
   maxwedge = 0
   for i in range(len(wedges)):
      wedge = wedges[i]
      if (len(wedge) > maxwedge):
         maxwedge, ind = len(wedge), i
   normed_wedge = maxwedge/len(flat_cnt)
   return normed_wedge
   
def plot_contour_roi(cnt_i, img1):
    expand_box    = 50
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
    plt.imshow(img_ROI)
    
def extractRingContour(cnt_i, img1, num_morphs, clonal_mark_type):
    expand_box    = 100
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_roi       = cnt_i - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1)
    if (clonal_mark_type=="PNN" or clonal_mark_type=="NNN"):
       mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_ERODE, st_5, iterations = num_morphs)
    if (clonal_mark_type=="P" or clonal_mark_type=="N"):
       mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = num_morphs)
    halo_cnt, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    numcnts = len(halo_cnt)
    if (numcnts==0):
        return cnt_i
    elif (numcnts==1):
        return np.asarray(halo_cnt) + Start_ij_ROI
    else:        
        areas = []
        for i in range(numcnts):
            areas.append(contour_Area(halo_cnt[i]))
        maxarea = np.where(areas==np.max(areas))[0][0]
        return np.asarray(halo_cnt[maxarea]) + Start_ij_ROI      

def retrieve_clone_nuclear_features(crypt_cnt, nuclei_ch_raw, clone_ch_raw, backgrd, smallBlur_img_nuc):
    nuc_feat = getAllFeatures(crypt_cnt, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
    clone_feat = getAllFeatures(crypt_cnt, clone_ch_raw, backgrd, smallBlur_img_nuc)
    frac_halo = clone_feat.allHalo/(nuc_feat.allHalo+0.01)
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
        inds = np.asarray([ii for ii in inds if clone_content[ii] > 3e-2]) # flush out junk contours
        full_partial_statistic = clone_content[inds]
    if (clone_marker_type=="NNN"):
        inds = np.where(clone_content < tukey_lower_thresholdval(clone_content, numIQR = numIQR))[0]
        full_partial_statistic = 1 - clone_content[inds]
    clone_cnt = list(np.asarray(crypt_cnt)[inds])
    return clone_cnt, full_partial_statistic
'''
