#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:47:40 2018

@author: doran
"""
import cv2, os, time
import numpy as np
import pyvips
import keras
from keras.preprocessing.image import img_to_array
import DNN.model.u_net as unet
import DNN.params as params
from deconv_mat               import *
from automaticThresh_func     import auto_choose_ROI, calculate_thresholds, calculate_deconvolution_matrix
from MiscFunctions            import col_deconvol, col_deconvol_and_blur, simplify_contours, col_deconvol_and_blur2
from MiscFunctions            import getROI_img_vips, add_offset, write_cnt_text_file, plot_img
from cnt_Feature_Functions    import joinContoursIfClose_OnlyKeepPatches, st_3, contour_Area, plotCnt
from multicore_morphology     import getForeground_mc
from GUI_ChooseROI_class      import getROI_svs
from Segment_clone_from_crypt import find_clone_statistics, combine_feature_lists, determine_clones, determine_clones_gridways
from Segment_clone_from_crypt import remove_thrown_indices_clone_features, add_xy_offset_to_clone_features
from knn_prune                import remove_tiling_overlaps_knn

# Load DNN model
model = params.model_factory()
model.load_weights("./DNN/weights/best_weights.hdf5")

def get_tile_indices(maxvals, overlap = 200, SIZE = (1024, 1024)):
    all_indx = []
    width = SIZE[0]
    height = SIZE[1]
    x_max = maxvals[0] # x -> cols
    y_max = maxvals[1] # y -> rows
    num_tiles_x = x_max // (width-overlap)
    endpoint_x = num_tiles_x*(width-overlap) + overlap    
    overhang_x = x_max - endpoint_x
    if (overhang_x>0): num_tiles_x += 1
    
    num_tiles_y = y_max // (height-overlap)
    endpoint_y = num_tiles_y*(height-overlap) + overlap    
    overhang_y = y_max - endpoint_y
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
    
def predict_single_image(img, clonal_mark_type,  prob_thresh = 0.24, upper_thresh = 0.75):
    crypt_contours  = []
    size = (1024, 1024)
    all_indx = get_tile_indices((img.shape[1], img.shape[0]), overlap = 200, SIZE = size)
    x_tiles = len(all_indx)
    y_tiles = len(all_indx[0])
    crypt_contours  = []
    clone_feature_list = []
    nbins = 20 # for clone finding
    for i in range(x_tiles):
        for j in range(y_tiles):            
            # Find next small tile
            xy_vals     = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
            wh_vals     = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
            img_s         = img[xy_vals[1]:(xy_vals[1]+wh_vals[1]) , xy_vals[0]:(xy_vals[0]+wh_vals[0]) ] # i,j rather than x,y
            x_batch = [img_s]
            x_batch = np.array(x_batch, np.float32) / 255.
            # Perform prediction and find contours
            predicted_mask_batch = model.predict(x_batch)
            newcnts = mask_to_contours(predicted_mask_batch, prob_thresh)
            newcnts = [cc for cc in newcnts if len(cc)>4] # throw away points and lines (needed in contour class)
            newcnts = [cc for cc in newcnts if contour_Area(cc)>400]
            newcnts = cull_tile_edge_contours(newcnts, size)
            newcnts = cull_bad_contours(predicted_mask_batch, upper_thresh, newcnts)
            # Find clone channel features
            img_nuc, img_clone = get_channel_images_for_clone_finding(img_s, clonal_mark_type) ## this is now wrong -- need to find deconv_mat
            clone_features = find_clone_statistics(newcnts, img_nuc, img_clone, nbins)            
            # Add x, y tile offset to all contours (which have been calculated from a tile) for use in full image 
            newcnts = add_offset(newcnts, xy_vals)
            clone_features = add_xy_offset_to_clone_features(clone_features, xy_vals)
            # Add to lists
            if (len(newcnts)>0):
               clone_feature_list.append(clone_features)
               crypt_contours += newcnts
            del img_nuc, img_clone, img_s, predicted_mask_batch, clone_features, newcnts
    clone_feature_list = combine_feature_lists(clone_feature_list, len(crypt_contours), nbins) 
    ## Remove tiling overlaps and simplify remaining contours
    print("Of %d contours..." % len(crypt_contours))
    crypt_contours, kept_indices = remove_tiling_overlaps_knn(crypt_contours)
    print("...Keeping only %d due to tiling overlaps." % kept_indices.shape[0])
    clone_feature_list = remove_thrown_indices_clone_features(clone_feature_list, kept_indices)
        
    ## Find clones
    clone_inds, full_partial_statistics = determine_clones(clone_feature_list, clonal_mark_type)
    clone_contours = list(np.asarray(crypt_contours)[clone_inds])

def predict_svs_slide(file_name, folder_to_analyse, clonal_mark_type, prob_thresh = 0.24, upper_thresh = 0.75):
    start_time = time.time()
    imnumber = file_name.split("/")[-1].split(".")[0]
    try:
        os.mkdir(folder_to_analyse)
    except:
        pass
    crypt_contours  = []
    clone_feature_list = []
    ## Find deconvolution matrix for clone/nucl channel separation
    deconv_mat = calculate_deconvolution_matrix(file_name, clonal_mark_type)
    ## Tiling
    obj_svs  = getROI_svs(file_name, get_roi_plot = False)
    size = (1024, 1024)
    all_indx = get_tile_indices(obj_svs.dims_slides[0], overlap = 200, SIZE = size)
    x_tiles = len(all_indx)
    y_tiles = len(all_indx[0])
    nbins = 20 # for clone finding
    for i in range(x_tiles):
        for j in range(y_tiles):
            xy_vals     = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
            wh_vals     = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
            img         = getROI_img_vips(file_name, xy_vals, wh_vals)
            img_nuc, img_clone = get_channel_images_for_clone_finding(img, deconv_mat)
            x_batch = [img]
            x_batch = np.array(x_batch, np.float32) / 255.

            # Perform prediction and find contours
            predicted_mask_batch = model.predict(x_batch)
            newcnts = mask_to_contours(predicted_mask_batch, prob_thresh)
            newcnts = [cc for cc in newcnts if len(cc)>4] # throw away points and lines (needed in contour class)
            newcnts = [cc for cc in newcnts if contour_Area(cc)>400]
            newcnts = cull_tile_edge_contours(newcnts, size)
            newcnts = cull_bad_contours(predicted_mask_batch, upper_thresh, newcnts)
            # Find clone channel features            
            clone_features = find_clone_statistics(newcnts, img_nuc, img_clone, nbins)
            # Add x, y tile offset to all contours (which have been calculated from a tile) for use in full image 
            newcnts = add_offset(newcnts, xy_vals)
            clone_features = add_xy_offset_to_clone_features(clone_features, xy_vals)
            # Add to lists
            if (len(newcnts)>0):
               clone_feature_list.append(clone_features)
               crypt_contours += newcnts
            del img_nuc, img_clone, img, predicted_mask_batch, clone_features, newcnts
        print("Found %d contours so far, tile %d of %d" % (len(crypt_contours), i*y_tiles+j, x_tiles*y_tiles))
            
    clone_feature_list = combine_feature_lists(clone_feature_list, len(crypt_contours), nbins) 
    ## Remove tiling overlaps and simplify remaining contours
    print("Of %d contours..." % len(crypt_contours))
    crypt_contours, kept_indices = remove_tiling_overlaps_knn(crypt_contours)
    print("...Keeping only %d due to tiling overlaps." % kept_indices.shape[0])
    clone_feature_list = remove_thrown_indices_clone_features(clone_feature_list, kept_indices)
    
    ## Find clones
    clone_inds, full_partial_statistics = determine_clones_gridways(clone_feature_list, clonal_mark_type)
    clone_contours = list(np.asarray(crypt_contours)[clone_inds])
    np.savetxt(folder_to_analyse + '/.csv', full_partial_statistics, delimiter=",")    
    
    # Join neighbouring clones to make cluster (clone patches that originate via crypt fission)
    # Don't do this if more than 25% of crypts are positive as it's hom tissue
    if len(clone_contours) < 0.25*len(crypt_contours) and len(crypt_contours)>0:
        patch_contours = joinContoursIfClose_OnlyKeepPatches(clone_contours, max_distance = 400)
    else:
        patch_contours = []

    ## Reduce number of vertices per contour to save space/QuPath loading time
    crypt_contours = simplify_contours(crypt_contours)
    clone_contours = simplify_contours(clone_contours)
    patch_contours = simplify_contours(patch_contours)

    write_cnt_text_file(crypt_contours, folder_to_analyse + "/crypt_contours.txt")
    write_cnt_text_file(clone_contours, folder_to_analyse + "/clone_contours.txt")
    write_cnt_text_file(patch_contours, folder_to_analyse + "/patch_contours.txt")
    print("Done " + imnumber + " in " +  str((time.time() - start_time)/60.) + " min =========================================")

def get_channel_images_for_clone_finding(img, deconv_mat):
    ## Choose deconv mat
    #if (clonal_mark_type=="P"): deconv_mat = deconv_mat_KDM6A # Don't have an example of this for a deconvolution matrix        
    #if (clonal_mark_type=="N"): deconv_mat = deconv_mat_KDM6A
    #if (clonal_mark_type=="PNN"): deconv_mat = deconv_mat_MPAS
    #if (clonal_mark_type=="NNN"): deconv_mat = deconv_mat_MAOA
    #if (clonal_mark_type=="BN"): deconv_mat = deconv_mat_MAOA
    #if (clonal_mark_type=="BP"): deconv_mat = deconv_mat_MAOA # Don't have an example of this for a deconvolution matrix
    img_nuc, img_clone = col_deconvol_and_blur2(img, deconv_mat, (11, 11), (13, 13))
    return img_nuc, img_clone

def cull_bad_contours(preds, upperthresh, contours):
   # for a single prediction probability distribution
   pred = (preds[0,:,:,0]*255).astype(np.uint8)
   newconts = []
   # Throw those with small mean probability
   for cnt_i in contours:
      roi           = cv2.boundingRect(cnt_i)
      Start_ij_ROI  = roi[0:2] # get x,y of bounding box
      cnt_roi       = cnt_i - Start_ij_ROI # change coords to start from x,y
      pred_ROI      = pred[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
      mask_fill     = np.zeros(pred_ROI.shape[0:2], np.uint8)
      cv2.drawContours(mask_fill, [cnt_roi], 0, 255, -1) ## Get mask
      mean_prob   = cv2.mean(pred_ROI, mask_fill)[0]/255.
      if (mean_prob > upperthresh):
         newconts.append(cnt_i)   
   return newconts
   
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
      contours += cnts
   return contours
        
'''
def clone_finding(file_name, clonal_mark_type):
    ## Choose deconv mat
    if (clonal_mark_type=="P"): deconv_mat = deconv_mat_KDM6A # Don't have an example of this for a deconvolution matrix        
    if (clonal_mark_type=="N"): deconv_mat = deconv_mat_KDM6A
    if (clonal_mark_type=="PNN"): deconv_mat = deconv_mat_MPAS
    if (clonal_mark_type=="NNN"): deconv_mat = deconv_mat_MAOA
    if (clonal_mark_type=="BN"): deconv_mat = deconv_mat_MAOA
    if (clonal_mark_type=="BP"): deconv_mat = deconv_mat_MAOA # Don't have an example of this for a deconvolution matrix
    xyhw = auto_choose_ROI(file_name, deconv_mat, plot_images = False)
    img_for_thresh = getROI_img_vips(file_name, xyhw[0], xyhw[1])
    thresh_cut_nucl, thresh_cut_nucl_blur, th_clone = calculate_thresholds(img_for_thresh, deconv_mat)
    return [thresh_cut_nucl, thresh_cut_nucl_blur, th_clone], deconv_mat
'''
'''
def get_thresholded_channel_images_for_clone_finding(img, deconv_mat, thresh_three):
    smallBlur_img_nuc, blurred_img_nuc, blurred_img_clone = col_deconvol_and_blur(img, deconv_mat, (11, 11), (37, 37), (27, 27))
    _, nuclei_ch_raw = cv2.threshold( smallBlur_img_nuc, thresh_three[0], 255, cv2.THRESH_BINARY)
    _, nucl_thresh   = cv2.threshold( blurred_img_nuc, thresh_three[1], 255, cv2.THRESH_BINARY)
    _, clone_ch_raw = cv2.threshold(blurred_img_clone, thresh_three[2], 255, cv2.THRESH_BINARY)
    nuclei_ch_raw   = cv2.morphologyEx(nuclei_ch_raw, cv2.MORPH_OPEN,  st_3, iterations=1)
    nucl_thresh_aux = cv2.morphologyEx(  nucl_thresh, cv2.MORPH_OPEN,  st_3, iterations=1)
    foreground = getForeground_mc(nucl_thresh_aux)    
    backgrd    = 255 - foreground
    return smallBlur_img_nuc, nuclei_ch_raw, clone_ch_raw, backgrd
''' 
