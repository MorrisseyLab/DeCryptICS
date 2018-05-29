# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:49:10 2015

@author: edward
"""
import numpy as np
import pickle, os, time
from MiscFunctions              import getROI_img_vips, getIndexesTileImage, add_offset
from MiscFunctions              import write_cnt_text_file, read_cnt_text_file, simplify_contours
from GUI_ChooseROI_class        import getROI_svs
from cnt_Feature_Functions      import joinContoursIfClose_OnlyKeepPatches
from knn_prune                  import remove_tiling_overlaps_knn
from automaticThresh_func       import auto_choose_ROI, calculate_thresholds, calculate_deconvolution_matrix_and_ROI
from Segment_clone_from_crypt   import find_clone_statistics, combine_feature_lists, determine_clones, remove_thrown_indices_clone_features, add_xy_offset_to_clone_features
from SegmentCrypts              import Segment_crypts
from deconv_mat                 import *

def GetThresholdsPrepareRun(folder_in, file_in, folder_out, clonal_mark_type):
    ## Make file name
    file_name   = folder_in + "/" + file_in + ".svs"
    
    ## Open slide and select regions for Crop and Zoom
    ########################################################
    obj_svs  = getROI_svs(file_name, get_roi_plot = False)
    ROI_crop = obj_svs.roi_full_thmb
    ROI_zoom = [] # obj_svs.chosenROI
    
    #xyhw = auto_choose_ROI(file_name, clonal_mark_type, plot_images = False)
    xy, wh, deconv_mat = calculate_deconvolution_matrix_and_ROI(file_name, clonal_mark_type)
    img_for_thresh = getROI_img_vips(file_name, xy, wh)
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
        pickle.dump([folder_in, file_in, ROI_zoom, all_indx, deconv_mat, thresh_three], f)      

def SegmentFromFolder(folder_name, clonal_mark_type):
    pkl_file    = open(folder_name + '/params.pickle', 'rb')
    folder_in, file_in, ROI_zoom, all_indx, deconv_mat, thresh_cut = pickle.load(pkl_file)
    pkl_file.close()
    
    file_name   = folder_in + "/" + file_in + ".svs"

    ## Set up storage structures
    crypt_contours  = []
    clone_features_list = []
    x_tiles = len(all_indx)
    y_tiles = len(all_indx[0])
    start_time = time.time()
    nbins = 20 # for clone finding
    for i in range(x_tiles):
        for j in range(y_tiles):
            xy_vals     = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
            wh_vals     = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
            img         = getROI_img_vips(file_name, xy_vals, wh_vals)
            crypt_cnt_ii, clone_features = Segment_crypts(img, thresh_cut, deconv_mat)    
                    
            ## Add x, y tile offset to all contours (which have been calculated from a tile) for use in full image 
            crypt_cnt_ii        = add_offset(crypt_cnt_ii, xy_vals)
            crypt_contours     += crypt_cnt_ii
            
            ## Add the clone channel features to the list
            clone_features = add_xy_offset_to_clone_features(clone_features, xy_vals)
            clone_features_list.append(find_clone_statistics(crypt_cnt, img_nuc, img_clone, nbins))  
            print("Done " + str(i) + '.' + str(j) + " (" + str(x_tiles-1) + '.' + str(y_tiles-1) + ")")
            del img         
    
    clone_features_list = combine_feature_lists(clone_features_list, len(crypt_cnt), nbins) 
    ## Remove tiling overlaps and simplify remaining contours
    print("Of %d contours..." % len(crypt_contours))
    crypt_contours, kept_indices = remove_tiling_overlaps_knn(crypt_contours)
    print("...Keeping only %d due to tiling overlaps." % kept_indices.shape[0])
    clone_features_list = remove_thrown_indices_clone_features(clone_features_list, kept_indices)
    
    ## Find clones
    clone_inds, full_partial_statistics = determine_clones(clone_features_list, clonal_mark_type)
    clone_contours = list(np.asarray(crypt_contours)[clone_inds])
    np.savetxt(folder_to_analyse + '/.csv', full_partial_statistics, delimiter=",")   

    # Join neighbouring clones to make cluster (clone patches that originate via crypt fission)
    # Don't do this if more than 25% of crypts are positive as it's hom tissue
    if len(clone_contours) < 0.25*len(crypt_contours) and len(crypt_contours)>0:
        patch_contours = joinContoursIfClose_OnlyKeepPatches(clone_contours, max_distance = 400)
    else:
        patch_contours = [] # return empty if just single clones ?

    ## Reduce number of vertices per contour to save space/QuPath loading time
    crypt_contours = simplify_contours(crypt_contours)
    clone_contours = simplify_contours(clone_contours)
    patch_contours = simplify_contours(patch_contours)
    
    write_cnt_text_file(crypt_contours, folder_name + "/crypt_contours.txt")
    write_cnt_text_file(clone_contours, folder_name + "/clone_contours.txt")
    write_cnt_text_file(patch_contours, folder_name + "/patch_contours.txt")
    print("Done " + file_in + " in " +  str((time.time() - start_time)/60.) + " min  =========================================")
    
    
