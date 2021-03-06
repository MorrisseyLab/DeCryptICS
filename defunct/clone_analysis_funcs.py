# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:09:16 2018

@author: doran
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

def outlier_level_calc(av_sig_mat, numIQR = 0.5):
   big_outlier = tukey_lower_thresholdval(flat_signal, numIQR)
   return big_outlier

def outlier_level_calc_above(av_sig_mat, numIQR = 0.5):
   big_outlier = tukey_upper_thresholdval(flat_signal, numIQR)
   return big_outlier
   
def nucl_bin_dropout(av_sig_vec, nucl_outlier):
   use_bin_inds = np.where(av_sig_vec > nucl_outlier)[0]
   ll = use_bin_inds.shape[0] 
   while(ll == 0):
      nucl_outlier = nucl_outlier*0.99
      use_bin_inds = np.where(av_sig_vec > nucl_outlier)[0]
      ll = use_bin_inds.shape[0] 
   return use_bin_inds
   
def local_nucl_bin_dropout(av_sig_vec, numIQR=1.25):
   use_bin_inds = np.where(av_sig_vec > tukey_lower_thresholdval(av_sig_vec, numIQR))[0]
   return use_bin_inds
   
def signal_width_ndo(av_sig_clone, outlier_val, clonal_mark_type, numbins):
   if (av_sig_clone.shape[0]==0): return 0, 0 # bin whole crypt!
   outlier_vec = np.ones(av_sig_clone.shape[0]) * outlier_val
   if (clonal_mark_type[0]=='N'):
      clone_trues = av_sig_clone < outlier_val
   if (clonal_mark_type[0]=='P'):
      clone_trues = av_sig_clone > outlier_val
   i = 0
   wedges = []
   for key, group in itertools.groupby(clone_trues, lambda x: x):
           truefalse = next(group)
           elems = len(list(group)) + 1
           if truefalse == True and elems > 0:
               wedges.append([key, elems, i])
           i += elems   
   if (len(wedges)==0):
      return 0, 0
   if (len(wedges)>1):
      # join loop
      if (clone_trues[0]==True and clone_trues[-1]==True):
         wedges[0][1] = wedges[0][1] + wedges[-1][1]
         wedges[0][2] = wedges[-1][2]
         wedges = wedges[:-1]
   if (len(wedges)>0):
      maxwedge = 0
      total_sig = 0
      for i in range(len(wedges)):
         wedge = wedges[i]
         if (wedge[1] > 1):
            total_sig += wedge[1]
         if (wedge[1] > maxwedge):
            maxwedge, ind = wedge[1], i
      normed_wedge = maxwedge/float(numbins) ## [len(clone_trues)] # instead use total number of bins
      normed_totalsig = total_sig/float(numbins)
      return normed_wedge, normed_totalsig
 
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

def extractRingContour(cnt_roi, mask_fill_morphs, Start_ij_ROI):      
   halo_cnt, _ = cv2.findContours(mask_fill_morphs.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
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
    mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_ERODE, st_3, iterations = num_morphs)
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

#def test_for_inner_ring_NaNs(crypt_contours):
#    for i in range(len(crypt_contours)):
#       cnt_i = crypt_contours[i]       
#       expand_box    = 100
#       roi           = cv2.boundingRect(cnt_i)            
#       roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
#       roi[roi <1]   = 0
#       Start_ij_ROI  = roi[0:2] # get x,y of bounding box
#       cnt_roi       = cnt_i - Start_ij_ROI # change coords to start
#       mask_fill1    = np.zeros((roi[3], roi[2]), np.uint8)
#       cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1)
#       mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_CLOSE, st_5, iterations = 1) # get rid of inner black blobs
#       mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_ERODE, st_3, iterations = 1)
#       halo_cnt, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
#       num_hal = len(halo_cnt)
#       if (num_hal==0):        
#           NEWCNT = cnt_i.astype(np.int32)
#       elif (num_hal==1):
#           NEWCNT = ((np.asarray(halo_cnt) + Start_ij_ROI).astype(np.int32))[0]
#       else:        
#           areas = []
#           for i in range(num_hal):
#               areas.append(contour_Area(halo_cnt[i]))
#           maxarea = np.where(areas==np.max(areas))[0][0]
#           NEWCNT = ((np.asarray(halo_cnt[maxarea]) + Start_ij_ROI).astype(np.int32))
#       print(i)
#       print(len(cnt_i))
#       print(len(NEWCNT))
#       inav_sig_nucl = bin_intensities_flattened(NEWCNT, mask_fill1, 20)
   
   
def bin_intensities_flattened(output_cnt, img1, nbins = 20):
   roi           = cv2.boundingRect(output_cnt)
   Start_ij_ROI  = roi[0:2] # get x,y of bounding box
   cnt_j     = output_cnt - Start_ij_ROI # change coords to start from x,y
   img_ROI   = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] # note here the use of y coord first!
   flat_cnt  = flatten_contour(cnt_j, img_ROI)
   numpixels = len(flat_cnt)
   flat_cnt = np.asarray(flat_cnt)
   av_sig = np.zeros(nbins)
   if numpixels>=nbins:
      overhang  = numpixels % nbins
      normal_bin_width = numpixels // nbins
      # (nbins - overhang) * normal_bin_width + overhang*(normal_bin_width + 1) == numpixels
      cw     = normal_bin_width
      for i in range(nbins-overhang):
         av_sig[i] = np.mean(flat_cnt[i*cw : (i+1)*cw])
      done = (nbins-overhang)*normal_bin_width
      cw   = normal_bin_width + 1
      for i in range(overhang):
         av_sig[i + nbins-overhang] = np.mean(flat_cnt[done + i*cw : done + (i+1)*cw])
   if numpixels<nbins:
      div = nbins // numpixels
      overhang = nbins - div*numpixels
      j = 0
      for i in range(nbins-overhang):
         if (i%div==0 and i!=0): j += 1
         av_sig[i] = flat_cnt[j]
      for i in range(overhang):
         av_sig[i + nbins-overhang] = flat_cnt[-1] 
   return av_sig
   
def flatten_contour(cnt_j, img_ROI):
   flat_cnt = []
   for xy_i in cnt_j[:,0,:]:
      flat_cnt.append(img_ROI[xy_i[1],xy_i[0]])
   return flat_cnt
   
def get_contents(cnt, img_nuc, img_clone):
   # Get mean colour of object
   roi           = cv2.boundingRect(cnt)
   Start_ij_ROI  = np.array(roi)[0:2] # get x,y of bounding box
   cnt_roi       = cnt - Start_ij_ROI # chnage coords to start from x,y
   img_ROI_n     = img_nuc[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
   img_ROI_c     = img_clone[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
   mask_fill     = np.zeros(img_ROI_n.shape[0:2], np.uint8)
   cv2.drawContours(mask_fill, [cnt_roi], 0, 255, -1) ## Get mask
   content_nucl  = cv2.mean(img_ROI_n, mask_fill)[0]/255.
   content_clone = cv2.mean(img_ROI_c, mask_fill)[0]/255.
   return (content_nucl, content_clone)

def max_halocnt_nc(cnt_i, img_nuc, img_clone):
   # Max and min halo size to calculate
   start_diff = 1 # min diff to check 
   end_diff   = 8 # end_diff-1 max diff to check
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
   max_morphs    = 20
   mask_list = [mask_fill1]
   
   sum_morphs    = np.zeros(max_morphs+1)
   areas_morphs  = np.zeros(max_morphs+1)
   # Area and sum pre-dilations
   areas_morphs[0] = cv2.countNonZero(mask_fill1)
   sum_morphs[0]   = cv2.mean(img_ROI, mask_fill1)[0] * areas_morphs[0]
   for i in range(1, max_morphs+1):
      mask_fill1    = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = 1)
      mask_list.append(mask_fill1)
      areas_morphs[i]  = cv2.countNonZero(mask_fill1)
      sum_morphs[i]    = cv2.mean(img_ROI, mask_fill1)[0] * areas_morphs[i]
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
         maxindx    = np.where(halo_mean==max_each[ii])[0][0]        
         morph_pair = (indx_1[maxindx], indx_2[maxindx])
         indices.append(morph_pair)
   nucl_halo        = np.max(max_each)
   maxindx_global   = np.where(max_each==nucl_halo)[0][0]
   morph_pair_m     = indices[maxindx_global]
   # find clone_halo
   clone_halo = cv2.mean(img_ROI_c, mask_list[morph_pair_m[0]]-mask_list[morph_pair_m[1]])[0]
   # extract the mid contour for use in flattened signal calculations
   maxmiddlecontour = int( np.ceil( (morph_pair_m[0]+morph_pair_m[1])/2. ) )
   mid_halo_cnt     = extractRingContour(cnt_roi, mask_list[maxmiddlecontour], Start_ij_ROI)
   output_cnt       = mid_halo_cnt.astype(np.int32)
   return nucl_halo, clone_halo, output_cnt

#   mask_list = [mask_fill1]
#   for i in range(1, max_morphs+1):
#      mask_fill1    = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = 1)
#      mask_list.append(mask_fill1)
#      
#   # Finding maximum halo
#   num_diffs  = end_diff-start_diff
#   running_differences = np.zeros([num_diffs, max_morphs+1])
#   #bin_intensities = np.zeros([num_diffs, max_morphs+1])
#   for i in range(max_morphs+1):
#      for j in range(1,num_diffs+1): # first row is one morph difference
#         if ( (i-j) < 0 ): 
#            pass
#         else:
#            running_differences[j-1,i] = cv2.mean(img_ROI, mask_list[i]-mask_list[i-j])[0]
#            #maxmiddlecontour = int( np.ceil( (i+(i-j))/2. ) )
#            #mid_halo_cnt     = extractRingContour(cnt_roi, img_ROI, maxmiddlecontour, 0)
#            #output_cnt       = mid_halo_cnt.astype(np.int32)
#            #bin_intensities[j-1,i]  = np.mean(bin_intensities_flattened(output_cnt, img_ROI))
#   # define max nuclear and clone halos
#   nucl_halo = np.max(running_differences)
#   indsmax = np.where(running_differences==nucl_halo)
#   morph_pair_m = ( indsmax[1][0] , indsmax[1][0]-(indsmax[0][0]+1) ) # ( outer morphs , outer - num morphs )

