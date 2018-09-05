# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:49:10 2015

@author: edward
"""
import numpy as np
import pickle, os, time
from MiscFunctions              import getROI_img_osl, getIndexesTileImage, add_offset
from MiscFunctions              import write_cnt_text_file, read_cnt_text_file, simplify_contours
from GUI_ChooseROI_class        import getROI_svs
from cnt_Feature_Functions      import joinContoursIfClose_OnlyKeepPatches
from knn_prune                  import remove_tiling_overlaps_knn
from automaticThresh_func       import auto_choose_ROI, calculate_thresholds, calculate_deconvolution_matrix_and_ROI
from Segment_clone_from_crypt   import find_clone_statistics, combine_feature_lists, determine_clones, subset_clone_features, add_xy_offset_to_clone_features
from Bayes_segment_crypts       import Segment_crypts
from deconv_mat                 import *

def GetThresholdsPrepareRun(full_path, file_in, folder_out, clonal_mark_type):
    ## Open slide and select regions for Crop and Zoom
    ########################################################
    obj_svs  = getROI_svs(full_path, get_roi_plot = False)
    ROI_crop = obj_svs.roi_full_thmb
    ROI_zoom = [] # obj_svs.chosenROI
    
    #xyhw = auto_choose_ROI(full_path, clonal_mark_type, plot_images = False)
    xy, wh, deconv_mat = calculate_deconvolution_matrix_and_ROI(full_path, clonal_mark_type)
    img_for_thresh = getROI_img_osl(full_path, xy, wh)
    thresh_cut_nucl, thresh_cut_nucl_blur, th_clone = calculate_thresholds(img_for_thresh, deconv_mat)
    thresh_three = (thresh_cut_nucl, thresh_cut_nucl_blur, th_clone)

    ## Make directory, get indexes to tile image and save info
    ########################################################        
    # Make directory for images     
    full_folder_out = folder_out + '/Analysed_' + file_in
    try:
        os.mkdir(full_folder_out)
    except:
        pass
    ## Get indexes to split
    all_indx = getIndexesTileImage(obj_svs.dims_slides[0], obj_svs.scalingVal, ROI_crop, max_num_pix  = 12500)      
    with open(full_folder_out + '/params.pickle', 'wb') as f:
        pickle.dump([full_path, file_in, ROI_zoom, all_indx, deconv_mat, thresh_three], f)      

def SegmentFromFolder(folder_name, clonal_mark_type, find_clones = False):
   pkl_file    = open(folder_name + '/params.pickle', 'rb')
   full_path, file_in, ROI_zoom, all_indx, deconv_mat, thresh_cut = pickle.load(pkl_file)
   pkl_file.close()

   ## Set up storage structures
   crypt_contours  = []
   x_tiles = len(all_indx)
   y_tiles = len(all_indx[0])
   start_time = time.time()
   
   ## Find deconvolution matrix for clone/nucl channel separation
   if find_clones:
      _, _, deconv_mat = calculate_deconvolution_matrix_and_ROI(file_name, clonal_mark_type)
      nbins = 20
      clone_feature_list = []
      
   for i in range(x_tiles):
      for j in range(y_tiles):
         xy_vals     = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
         wh_vals     = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
         img         = getROI_img_osl(full_path, xy_vals, wh_vals)
         newcnts, clone_features = Segment_crypts(img, thresh_cut, deconv_mat, nbins, find_clones)    
                 
         ## Add x, y tile offset to all contours (which have been calculated from a tile) for use in full image 
         newcnts        = add_offset(newcnts, xy_vals)
         newcnts = [cc for cc in newcnts if len(cc)>4] # throw away points and lines (needed in contour class)
         newcnts = [cc for cc in newcnts if contour_Area(cc)>(800./(scaling_val*scaling_val))] # areas are scaled down by a scale_factor^2
         crypt_contours += newcnts
         
         if (find_clones==True):
            ## Add the clone channel features to the list
            clone_features = add_xy_offset_to_clone_features(clone_features, xy_vals)
            clone_feature_list.append(clone_features)  
         print("Done " + str(i) + '.' + str(j) + " (" + str(x_tiles-1) + '.' + str(y_tiles-1) + ")")
         del img         

   if find_clones:
      cfl = combine_feature_lists(clone_feature_list, len(crypt_contours), nbins)
      
   ## Remove tiling overlaps
   print("Of %d contours..." % len(crypt_contours))
   crypt_contours, kept_indices = remove_tiling_overlaps_knn(crypt_contours)
   print("...Keeping only %d due to tiling overlaps." % kept_indices.shape[0])
   if find_clones:
      cfl = subset_clone_features(cfl, kept_indices, keep_global_inds=False)
   
   if find_clones:
       clone_inds = determine_clones(cfl, clonal_mark_type, crypt_contours = crypt_contours)
  
   ## Reduce number of vertices per contour to save space/QuPath loading time
   crypt_contours = simplify_contours(crypt_contours)

   ## Convert contours to fullscale image coordinates
   crypt_contours = rescale_contours(crypt_contours, scaling_val)
   if find_clones:
      clone_contours = list(np.asarray(crypt_contours)[clone_inds])
      clone_contours = simplify_contours(clone_contours)
      ## Join patches
      if len(clone_contours) < 0.25*len(crypt_contours) and len(crypt_contours)>0:
         patch_contours = joinContoursIfClose_OnlyKeepPatches(clone_contours, max_distance = 400)
      else:
         patch_contours = []

   write_cnt_text_file(crypt_contours, folder_to_analyse + "/crypt_contours.txt")
   if find_clones:
      write_cnt_text_file(clone_contours, folder_to_analyse + "/clone_contours.txt")
      write_cnt_text_file(patch_contours, folder_to_analyse + "/patch_contours.txt")

   print("Done " + imnumber + " in " +  str((time.time() - start_time)/60.) + " min =========================================")
    
