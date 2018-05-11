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
from clone_analysis_funcs  import *

'''
## bounding clone finding by coordinate
highylim = np.where(clone_feature_list[4][:,1]<36000)
highxlim = np.where(clone_feature_list[4][:,0]<16000)
lowylim = np.where(clone_feature_list[4][:,1]>25000)
lowxlim = np.where(clone_feature_list[4][:,0]>10000)
x_inds = np.intersect1d(lowxlim[0], highxlim[0])
y_inds = np.intersect1d(lowylim[0], highylim[0])
z_inds = np.intersect1d(x_inds, y_inds)
'''

def find_clone_statistics(crypt_cnt, img_nuc, img_clone, nbins = 20):
   # for each contour do no_threshold_signal_collating()
   numcnts = len(crypt_cnt)
   xy_coords = np.zeros([numcnts, 2], dtype=np.int32)
   halo_n = np.zeros([numcnts])
   halo_c = np.zeros([numcnts])
   content_n = np.zeros([numcnts])
   content_c = np.zeros([numcnts])
   out_av_sig_nucl  = np.zeros([numcnts, nbins], dtype=np.float32)
   out_av_sig_clone = np.zeros([numcnts, nbins], dtype=np.float32)
   in_av_sig_nucl   = np.zeros([numcnts, nbins], dtype=np.float32)
   in_av_sig_clone  = np.zeros([numcnts, nbins], dtype=np.float32)
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
      content_n[i]           = X[7]
      content_c[i]           = X[8]
   clone_feature_list = (halo_n, halo_c, out_av_sig_nucl, out_av_sig_clone, xy_coords, in_av_sig_nucl, in_av_sig_clone, content_n, content_c)
   return clone_feature_list

def combine_feature_lists(clone_feature_list, numcnts, nbins = 20):
   xy_coords = np.zeros([numcnts, 2], dtype=np.int32)
   halo_n = np.zeros([numcnts])
   halo_c = np.zeros([numcnts])
   content_n = np.zeros([numcnts])
   content_c = np.zeros([numcnts])
   out_av_sig_nucl  = np.zeros([numcnts, nbins], dtype=np.float32)
   out_av_sig_clone = np.zeros([numcnts, nbins], dtype=np.float32)
   in_av_sig_nucl   = np.zeros([numcnts, nbins], dtype=np.float32)
   in_av_sig_clone  = np.zeros([numcnts, nbins], dtype=np.float32)
   in_av_sig_clone  = np.zeros([numcnts, nbins], dtype=np.float32)
   cumnum = 0
   for i in range(len(clone_feature_list)):
      curr_num = clone_feature_list[i][0].shape[0]
      halo_n[cumnum:(cumnum+curr_num)]              = clone_feature_list[i][0]
      halo_c[cumnum:(cumnum+curr_num)]              = clone_feature_list[i][1]
      out_av_sig_nucl[cumnum:(cumnum+curr_num), :]  = clone_feature_list[i][2]
      out_av_sig_clone[cumnum:(cumnum+curr_num), :] = clone_feature_list[i][3]
      xy_coords[cumnum:(cumnum+curr_num),0]         = clone_feature_list[i][4][:,0]
      xy_coords[cumnum:(cumnum+curr_num),1]         = clone_feature_list[i][4][:,1]
      in_av_sig_nucl[cumnum:(cumnum+curr_num), :]   = clone_feature_list[i][5]
      in_av_sig_clone[cumnum:(cumnum+curr_num), :]  = clone_feature_list[i][6]
      content_n[cumnum:(cumnum+curr_num)]           = clone_feature_list[i][7]
      content_c[cumnum:(cumnum+curr_num)]           = clone_feature_list[i][8]
      cumnum = cumnum + curr_num
   clone_features = (halo_n, halo_c, out_av_sig_nucl, out_av_sig_clone, xy_coords, in_av_sig_nucl, in_av_sig_clone, content_n, content_c)
   return clone_features

def remove_thrown_indices_clone_features(clone_features, kept_indices):
   halo_n           = clone_features[0][kept_indices]
   halo_c           = clone_features[1][kept_indices]
   out_av_sig_nucl  = clone_features[2][kept_indices, :]
   out_av_sig_clone = clone_features[3][kept_indices, :]
   xy_coords = np.zeros([kept_indices.shape[0], 2], dtype=np.int32)
   for i in range(kept_indices.shape[0]):
      xy_coords[i,0]         = clone_features[4][kept_indices[i], 0]
      xy_coords[i,1]         = clone_features[4][kept_indices[i], 1]
   in_av_sig_nucl   = clone_features[5][kept_indices, :]
   in_av_sig_clone  = clone_features[6][kept_indices, :]
   content_n        = clone_features[7][kept_indices]
   content_c        = clone_features[8][kept_indices]   
   clone_features = (halo_n, halo_c, out_av_sig_nucl, out_av_sig_clone, xy_coords, in_av_sig_nucl, in_av_sig_clone, content_n, content_c)
   return clone_features
   
def add_xy_offset_to_clone_features(clone_features, xy_offset):
   for i in range(clone_features[0].shape[0]):
      clone_features[4][i, 0] = clone_features[4][i, 0] + xy_offset[0]
      clone_features[4][i, 1] = clone_features[4][i, 1] + xy_offset[1]
   return clone_features


def determine_clones(clone_feature_list, clonal_mark_type):
   halo_n           = clone_feature_list[0]
   halo_c           = clone_feature_list[1]
   out_av_sig_nucl  = clone_feature_list[2]
   out_av_sig_clone = clone_feature_list[3]
   xy_coords        = clone_feature_list[4]
   in_av_sig_nucl   = clone_feature_list[5]
   in_av_sig_clone  = clone_feature_list[6]
   content_n        = clone_feature_list[7]
   content_c        = clone_feature_list[8]
   out_frac_nc = (1+out_av_sig_clone) / (1+out_av_sig_nucl)
   in_frac_nc = (1+in_av_sig_clone) / (1+in_av_sig_nucl)
   # Define matrix for wedge finding
   out_wedge = out_frac_nc # out_frac_nc or out_av_sig_clone
   in_wedge = in_frac_nc # in_frac_nc or in_av_sig_clone
   # Calculate outliers of each bin
   out_frac_outlier_vecs = outlier_vec_calculator(out_frac_nc)
   in_frac_outlier_vecs = outlier_vec_calculator(in_frac_nc)
   if (clonal_mark_type=="P" or clonal_mark_type=="PNN" or clonal_mark_type=="BP"):
      out_outlier_vec = out_wedge[1]
      in_outlier_vec = in_wedge[1]
      clone_outlier_val = outlier_level_calc_above(out_av_sig_clone, numIQR=0.5)
   if (clonal_mark_type=="N" or clonal_mark_type=="NNN" or clonal_mark_type=="BN"): 
      out_outlier_vec = out_wedge[0]
      in_outlier_vec = in_wedge[0]
      clone_outlier_val = outlier_level_calc(out_av_sig_clone, numIQR=0.5)
   # Find nuclear dropout outlier val
   nucl_outlier_val = outlier_level_calc(out_av_sig_nucl)
   '''
   # Determine signal width for clones
   frac_signal_width = np.zeros(out_av_sig_nucl.shape[0])
   frac_signal_total = np.zeros(out_av_sig_nucl.shape[0])
   if (clonal_mark_type=="BN" or clonal_mark_type=="BP"):
      frac_signal_width_in = np.zeros(out_av_sig_nucl.shape[0])
      frac_signal_total_in = np.zeros(out_av_sig_nucl.shape[0])
      frac_signal_width_out = np.zeros(out_av_sig_nucl.shape[0])
      frac_signal_total_out = np.zeros(out_av_sig_nucl.shape[0])

   for i in range(out_av_sig_nucl.shape[0]):
      if (clonal_mark_type=="P" or clonal_mark_type=="N"):
         frac_signal_width[i], frac_signal_total[i] = signal_width(out_wedge[i, :], out_outlier_vec, out_av_sig_nucl[i, :], nucl_outlier_val, clonal_mark_type)
      if (clonal_mark_type=="PNN" or clonal_mark_type=="NNN"):
         frac_signal_width[i], frac_signal_total[i] = signal_width(in_wedge[i, :], in_outlier_vec, out_av_sig_nucl[i, :], nucl_outlier_val, clonal_mark_type)
      if (clonal_mark_type=="BP"):
         frac_signal_width_out[i], frac_signal_total_out[i] = signal_width(out_wedge[i, :], out_outlier_vec, out_av_sig_nucl[i, :], nucl_outlier_val, "P")
         frac_signal_width_in[i], frac_signal_total_in[i] = signal_width(in_wedge[i, :], in_outlier_vec, out_av_sig_nucl[i, :], nucl_outlier_val, "PNN")
      if (clonal_mark_type=="BN"):
         frac_signal_width_out[i], frac_signal_total_out[i] = signal_width(out_wedge[i, :], out_outlier_vec, out_av_sig_nucl[i, :], nucl_outlier_val, "N")
         frac_signal_width_in[i], frac_signal_total_in[i] = signal_width(in_wedge[i, :], in_outlier_vec, out_av_sig_nucl[i, :], nucl_outlier_val, "NNN")
   if (clonal_mark_type=="BN" or clonal_mark_type=="BP"):
      # this needs work
      frac_signal_width = np.minimum(frac_signal_width_out , frac_signal_width_in)         

   # Combination of three tests: halo fractions/clone content, total signal below numIQR, signal width above 1/20th? (either we lose single cells or pick up FPs with last?
   #inds_sigtot = np.where( frac_signal_total > tukey_upper_thresholdval(frac_signal_total, numIQR=1.))[0]
   #inds_sigwid = np.where( frac_signal_width > 0.095)[0]
   #inds_sigwid = np.setdiff1d(inds_sigwid, np.hstack([inds_sigtot, inds_halo])) # and something to separate the clones from the noise?
   # Now look for extreme values / widths of the out_av_sig_clone (in some way compared to out_av_sig_nucl?) to separate real from FPs
   '''
   #TESTING
   out_wedge = out_av_sig_clone #[inds_sigwid,:]
   in_wedge = in_av_sig_clone #[inds_sigwid,:]
   NIQR = 0.8
   NBINS = 3.
   if (clonal_mark_type=="P" or clonal_mark_type=="PNN" or clonal_mark_type=="BP"):
      clone_outlier_val_out = outlier_level_calc_above(out_wedge, NIQR)
      clone_outlier_val_in = outlier_level_calc_above(in_wedge, NIQR)
   if (clonal_mark_type=="N" or clonal_mark_type=="NNN" or clonal_mark_type=="BN"): 
      clone_outlier_val_out = outlier_level_calc(out_wedge, NIQR)
      clone_outlier_val_in = outlier_level_calc(in_wedge, NIQR)
   clone_signal_width = np.zeros(out_av_sig_nucl.shape[0])
   clone_signal_total = np.zeros(out_av_sig_nucl.shape[0])
   if (clonal_mark_type=="BN" or clonal_mark_type=="BP"):
      clone_signal_width_in = np.zeros(out_av_sig_nucl.shape[0])
      clone_signal_total_in = np.zeros(out_av_sig_nucl.shape[0])
      clone_signal_width_out = np.zeros(out_av_sig_nucl.shape[0])
      clone_signal_total_out = np.zeros(out_av_sig_nucl.shape[0])
   for i in range(out_av_sig_clone.shape[0]):
      if (clonal_mark_type=="P" or clonal_mark_type=="N"):
         clone_signal_width[i]     = signal_width_ndo(out_av_sig_clone[i,:], clone_outlier_val_out, out_av_sig_nucl[i,:], clonal_mark_type)
      if (clonal_mark_type=="PNN" or clonal_mark_type=="NNN"):
         clone_signal_width[i]     = signal_width_ndo(in_av_sig_clone[i,:], clone_outlier_val_in, in_av_sig_nucl[i,:], clonal_mark_type)
      if (clonal_mark_type=="BP"):
         clone_signal_width_out[i] = signal_width_ep(out_av_sig_clone[i,:], clone_outlier_val_out, out_av_sig_nucl[i,:], "P")
         clone_signal_width_in[i]  = signal_width_ndo(in_av_sig_clone[i,:], clone_outlier_val_in, in_av_sig_nucl[i,:], "PNN")
      if (clonal_mark_type=="BN"):
         clone_signal_width_out[i] = signal_width_ndo(out_av_sig_clone[i,:], clone_outlier_val_out, out_av_sig_nucl[i,:], "N")
         clone_signal_width_in[i]  = signal_width_ndo(in_av_sig_clone[i,:], clone_outlier_val_in, in_av_sig_nucl[i,:], "NNN")
   inds_emp = np.where(clone_signal_width >= NBINS/out_av_sig_clone.shape[1])[0]
   plotCnt(img, np.asarray(crypt_contours)[inds_emp])


   frac_halo = halo_c/halo_n
   inds_halo   = np.where( frac_halo < tukey_lower_thresholdval(frac_halo, numIQR=0.75))[0]
   inds_cumul = np.array([-1])
   niqr = (np.linspace(0.35, 1.25, 14))
   nbins = (np.linspace(2,8,14))[::-1]
   zipped = zip(niqr, nbins)
   for pair in zipped:
      NIQR = pair[0]
      NBINS = pair[1]
      if (clonal_mark_type=="P" or clonal_mark_type=="PNN" or clonal_mark_type=="BP"):
         clone_outlier_val_out = outlier_level_calc_above(out_wedge, NIQR)
         clone_outlier_val_in = outlier_level_calc_above(in_wedge, NIQR)
      if (clonal_mark_type=="N" or clonal_mark_type=="NNN" or clonal_mark_type=="BN"): 
         clone_outlier_val_out = outlier_level_calc(out_wedge, NIQR)
         clone_outlier_val_in = outlier_level_calc(in_wedge, NIQR)
      clone_signal_width = np.zeros(out_av_sig_nucl.shape[0])
      clone_signal_total = np.zeros(out_av_sig_nucl.shape[0])
      if (clonal_mark_type=="BN" or clonal_mark_type=="BP"):
         clone_signal_width_in = np.zeros(out_av_sig_nucl.shape[0])
         clone_signal_total_in = np.zeros(out_av_sig_nucl.shape[0])
         clone_signal_width_out = np.zeros(out_av_sig_nucl.shape[0])
         clone_signal_total_out = np.zeros(out_av_sig_nucl.shape[0])
      for i in range(out_av_sig_clone.shape[0]):
         if (clonal_mark_type=="P" or clonal_mark_type=="N"):
            clone_signal_width[i]     = signal_width_ndo(out_av_sig_clone[i,:], clone_outlier_val_out, out_av_sig_nucl[i,:], clonal_mark_type)
         if (clonal_mark_type=="PNN" or clonal_mark_type=="NNN"):
            clone_signal_width[i]     = signal_width_ndo(in_av_sig_clone[i,:], clone_outlier_val_in, in_av_sig_nucl[i,:], clonal_mark_type)
         if (clonal_mark_type=="BP"):
            clone_signal_width_out[i] = signal_width_ep(out_av_sig_clone[i,:], clone_outlier_val_out, out_av_sig_nucl[i,:], "P")
            clone_signal_width_in[i]  = signal_width_ndo(in_av_sig_clone[i,:], clone_outlier_val_in, in_av_sig_nucl[i,:], "PNN")
         if (clonal_mark_type=="BN"):
            clone_signal_width_out[i] = signal_width_ndo(out_av_sig_clone[i,:], clone_outlier_val_out, out_av_sig_nucl[i,:], "N")
            clone_signal_width_in[i]  = signal_width_ndo(in_av_sig_clone[i,:], clone_outlier_val_in, in_av_sig_nucl[i,:], "NNN")
      inds_emp = np.where(clone_signal_width >= NBINS/out_av_sig_clone.shape[1])[0]            
      inds_cumul = np.hstack([inds_cumul, inds_emp])
      print(inds_emp.shape[0])
   inds_cumul = np.unique(inds_cumul[1:])
   
   # Cull junk crypts in whitespace
   inds_nonwhitespace = np.where( content_n>tukey_lower_thresholdval(content_n, numIQR=1.5) )[0]
   inds = np.intersect1d(inds_cumul, inds_nonwhitespace)
   
   return inds, clone_signal_width[inds]
   
def no_threshold_signal_collating(cnt_i, img_nuc, img_clone, nbins):
   # Find max halo contour in nuclear channel, and nucl/clone halo scores
   nucl_halo, clone_halo, output_cnt = max_halocnt_nc(cnt_i, img_nuc, img_clone)
   # Find clone/nucl content
   content_n, content_c = get_contents(cnt_i, img_nuc, img_clone)
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
   return nucl_halo, clone_halo, av_sig_nucl, av_sig_clone, centre_xy, inav_sig_nucl, inav_sig_clone, content_n, content_c
   
def outlier_vec_calculator(av_sig_mat, numIQR = 1.25):
   big_outlier = np.zeros(av_sig_mat.shape[1])
   small_outlier = np.zeros(av_sig_mat.shape[1])
   av_small_out = np.ones(av_sig_mat.shape[1])
   av_big_out = np.ones(av_sig_mat.shape[1])
   for j in range(av_sig_mat.shape[1]):
      small_outlier[j] = np.percentile(av_sig_mat[:, j], 25)
      big_outlier[j] = tukey_upper_thresholdval(av_sig_mat[:, j], 75)
   av_small_out = av_small_out * (np.mean(small_outlier))
   av_big_out = av_big_out * (np.mean(big_outlier))
   return av_small_out , av_big_out



