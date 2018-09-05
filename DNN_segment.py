#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:47:40 2018

@author: doran
"""
import tensorflow as tf
from keras import backend as K
import cv2, os, time
import numpy as np
#import pyvips
import keras
import pickle
from keras.preprocessing.image import img_to_array
import DNN.u_net as unet
import DNN.params as params
from deconv_mat               import *
from automaticThresh_func     import calculate_deconvolution_matrix_and_ROI, find_deconmat_fromtiles
from MiscFunctions            import simplify_contours, col_deconvol_and_blur2, mkdir_p, write_clone_image_snips
from MiscFunctions            import getROI_img_osl, add_offset, write_cnt_text_file, plot_img, rescale_contours, write_score_text_file
from cnt_Feature_Functions    import joinContoursIfClose_OnlyKeepPatches, st_3, contour_Area, plotCnt
from multicore_morphology     import getForeground_mc
from GUI_ChooseROI_class      import getROI_svs
from Segment_clone_from_crypt import find_clone_statistics, combine_feature_lists, determine_clones, determine_clones_gridways
from Segment_clone_from_crypt import subset_clone_features, add_xy_offset_to_clone_features, write_clone_features_to_file
from knn_prune                import remove_tiling_overlaps_knn

# Load DNN model
model = params.model_factory(input_shape=(params.input_size, params.input_size, 3))
model.load_weights("./DNN/weights/tile256_for_X_best_weights.hdf5")

def get_tile_indices(maxvals, overlap = 50, SIZE = (2048, 2048)):
    all_indx = []
    width = SIZE[0]
    height = SIZE[1]
    x_max = maxvals[0] # x -> cols
    y_max = maxvals[1] # y -> rows
    num_tiles_x = x_max // (width-overlap)
    endpoint_x  = num_tiles_x*(width-overlap) + overlap    
    overhang_x  = x_max - endpoint_x
    if (overhang_x>0): num_tiles_x += 1
    
    num_tiles_y = y_max // (height-overlap)
    endpoint_y  = num_tiles_y*(height-overlap) + overlap    
    overhang_y  = y_max - endpoint_y
    if (overhang_y>0): num_tiles_y += 1   
     
    for i in range(num_tiles_x):
        x0 = i*(width - overlap)
        if (i == (num_tiles_x-1)): x0 = x_max - width
        all_indx.append([])
        for j in range(num_tiles_y):
            y0 = j*(height - overlap)
            if (j == (num_tiles_y-1)): y0 = y_max - height
            all_indx[i].append((x0, y0, width, height))
    return all_indx

def predict_svs_slide(file_name, folder_to_analyse, clonal_mark_type, find_clones = False, prob_thresh = 0.5):
   start_time = time.time()
   imnumber = file_name.split("/")[-1].split(".")[0]
   mkdir_p(folder_to_analyse)
   crypt_contours  = []
   clone_feature_list = []
   
   ## Find deconvolution matrix for clone/nucl channel separation
   if find_clones:
      _, _, deconv_mat = calculate_deconvolution_matrix_and_ROI(file_name, clonal_mark_type)
      nbins = 20
      
   ## Tiling
   obj_svs  = getROI_svs(file_name, get_roi_plot = False)
   scaling_val = obj_svs.dims_slides[0][0] / float(obj_svs.dims_slides[1][0])
   size = (params.input_size, params.input_size)
   all_indx = get_tile_indices(obj_svs.dims_slides[1], overlap = 50, SIZE = size)
   x_tiles = len(all_indx)
   y_tiles = len(all_indx[0])
   
   for i in range(x_tiles):
      for j in range(y_tiles):
         xy_vals = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
         wh_vals = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
         img     = getROI_img_osl(file_name, xy_vals, wh_vals, level = 1)
         x_batch = [img]
         x_batch = np.array(x_batch, np.float32) / 255.

         # Perform prediction and find contours
         predicted_mask_batch = model.predict(x_batch)

         newcnts = mask_to_contours(predicted_mask_batch, prob_thresh)
         newcnts = [cc for cc in newcnts if len(cc)>4] # throw away points and lines (needed in contour class)
         
         #newcnts = [cc for cc in newcnts if contour_Area(cc)>(500./(scaling_val*scaling_val))] # areas are scaled down by a scale_factor^2
         #newcnts = cull_tile_edge_contours(newcnts, size) # REMOVING TOO MANY CONTOURS
         #newcnts = cull_bad_contours(predicted_mask_batch, upper_thresh, newcnts) # NOT A GOOD METHOD

         if find_clones:
            # ADD CHOICE TO DO CLONE FINDING IN ZOOMED IN OR ZOOMED OUT IMAGE (ZOOMED IN WILL BE MUCH SLOWER!) (mouse data may need zoomed-in for crypt finding due to size difference)
            # ALSO, THE DILATIONS USED IN clone_analysis_funcs.py SHOULD BE REDUCED IN STRENGTH IF USING ZOOMED OUT IMAGE
            # Find clone channel features
            bigxy = tuple(np.asarray([xy_vals[0]*scaling_val, xy_vals[1]*scaling_val], dtype=int))
            bigwh = tuple(np.asarray([wh_vals[0]*scaling_val, wh_vals[1]*scaling_val], dtype=int))
            rs_cnts = rescale_contours(newcnts, scaling_val)
            img = getROI_img_osl(file_name, bigxy, bigwh, level = 0)
            img_nuc, img_clone = get_channel_images_for_clone_finding(img, deconv_mat)
            clone_features = find_clone_statistics(rs_cnts, img_nuc, img_clone, nbins)
            clone_features = add_xy_offset_to_clone_features(clone_features, bigxy) # xy now untiled and in original unscaled coordinates
            #clone_features['mid_contour'] = add_offset(clone_features['mid_contour'], bigxy)
            
         # Add x, y tile offset to all contours (which have been calculated from a tile) for use in full (scaled) image 
         newcnts = add_offset(newcnts, xy_vals)

         # Add to lists
         if (len(newcnts)>0):
            crypt_contours += newcnts
            if find_clones:
               clone_feature_list.append(clone_features)
               del img_nuc, img_clone, clone_features
      print("Found %d contours so far, tile %d of %d" % (len(crypt_contours), i*y_tiles+j + 1, x_tiles*y_tiles))
         
   del img, predicted_mask_batch, newcnts
   if find_clones:
      cfl = combine_feature_lists(clone_feature_list, len(crypt_contours), nbins)
      
   ## Remove tiling overlaps and simplify remaining contours
   print("Of %d contours..." % len(crypt_contours))
   oldlen = 1
   newlen = 0
   while newlen!=oldlen:
      oldlen = len(crypt_contours)
      crypt_contours, kept_indices = remove_tiling_overlaps_knn(crypt_contours)
      if find_clones:
         cfl = subset_clone_features(cfl, kept_indices, keep_global_inds=False)    
      newlen = len(crypt_contours)
   print("...Keeping only %d due to tiling overlaps." % kept_indices.shape[0])

   #write_clone_features_to_file(clone_feature_list, folder_to_analyse) # output clone_feature_list matrices for analysis in R
   
   if find_clones:
       clone_inds, clone_scores = determine_clones(cfl, clonal_mark_type, crypt_contours = crypt_contours)
  
   ## Reduce number of vertices per contour to save space/QuPath loading time
   crypt_contours = simplify_contours(crypt_contours)

   ## Convert contours to fullscale image coordinates
   crypt_contours = rescale_contours(crypt_contours, scaling_val)
   if find_clones:
      clone_contours = list(np.asarray(crypt_contours)[clone_inds])
      ## Join patches
      if len(clone_contours) < 0.25*len(crypt_contours) and len(crypt_contours)>0:
         patch_contours, patch_sizes, patch_indices = joinContoursIfClose_OnlyKeepPatches(cfl, crypt_contours, clone_inds)
         patch_indices = convert_to_local_clone_indices(patch_indices, clone_inds)
      else:
         patch_contours, patch_sizes, patch_indices = [], [], []

   write_cnt_text_file(crypt_contours, folder_to_analyse + "/crypt_contours.txt")
   if find_clones:
      write_cnt_text_file(clone_contours, folder_to_analyse + "/clone_contours.txt")
      write_cnt_text_file(patch_contours, folder_to_analyse + "/patch_contours.txt")
      write_score_text_file(clone_scores, folder_to_analyse + "/clone_scores.txt")
      write_score_text_file(patch_sizes, folder_to_analyse + "/patch_sizes.txt")
      pickle.dump(patch_indices, open( folder_to_analyse + "/patch_indices.pickle", "wb" ) )
      write_clone_image_snips(folder_to_analyse, file_name, clone_contours, scaling_val)
      
   print("Done " + imnumber + " in " +  str((time.time() - start_time)/60.) + " min =========================================")   

def get_channel_images_for_clone_finding(img, deconv_mat):
    img_nuc, img_clone = col_deconvol_and_blur2(img, deconv_mat, (11, 11), (13, 13))
    return img_nuc, img_clone

#def cull_bad_contours(preds, upper_thresh, contours):
#   # for a single prediction probability distribution
#   pred = (preds[0,:,:,0]*255).astype(np.uint8)
#   newconts = []
#   # Throw those with small mean probability
#   for cnt_i in contours:
#      roi           = cv2.boundingRect(cnt_i)
#      Start_ij_ROI  = roi[0:2] # get x,y of bounding box
#      cnt_roi       = cnt_i - Start_ij_ROI # change coords to start from x,y
#      pred_ROI      = pred[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
#      mask_fill     = np.zeros(pred_ROI.shape[0:2], np.uint8)
#      cv2.drawContours(mask_fill, [cnt_roi], 0, 255, -1) ## Get mask
#      mean_prob   = cv2.mean(pred_ROI, mask_fill)[0]/255.
#      if (mean_prob > upper_thresh):
#         newconts.append(cnt_i)   
#   return newconts
   
def cull_tile_edge_contours(contours, img_dims):
   newconts = []
   uplim = img_dims[0]-2
   lowlim = 1
   # Throw those contours touching the tile edge
   for cnt_i in contours:
      num_high = np.sum((cnt_i>=uplim).astype(int))
      num_low = np.sum(( cnt_i<=lowlim).astype(int))
      if (num_low+num_high < 2):
         newconts.append(cnt_i)
   return newconts
    
def mask_to_contours(preds, thresh):
   contours = []
   for i in range(preds.shape[0]):
      # convert to np.uint8
      pred = (preds[i,:,:,0]*255).astype(np.uint8)
      # perform threshold
      _, mask = cv2.threshold(pred, thresh*255, 255, cv2.THRESH_BINARY)
      # find contours
      cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
      # Possible improvement:
      # for each contour, found initially with stringent/leniant prob thresh, go to ROI of predicted_mask
      # and slowly decrease/increase the probability theshold until the mean of the new contour is within
      # desired limits. Decide what to do if more than one contour is found in the ROI. May take too long
      contours += cnts
   return contours



