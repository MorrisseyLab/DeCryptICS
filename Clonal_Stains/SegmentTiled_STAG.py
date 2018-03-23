#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:28:30 2018

@author: doran
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:49:10 2015

@author: edward
"""
import numpy as np
import pickle, os, time
from Clonal_Stains.SegmentCryptAndCloneSTAG  import SegmentSTAG
from MiscFunctions              import getROI_img_vips, getIndexesTileImage, add_offset
from MiscFunctions              import write_cnt_text_file, read_cnt_text_file, simplify_contours
from GUI_ChooseROI_class        import getROI_svs
from func_Process_ClonesImageAndInfoFile import orderClones_Annotate_WriteImg
from deconv_mat import deconv_mat_STAG
from devel_knn_prune import remove_tiling_overlaps_knn
from automaticThresh_func import auto_choose_ROI, calculate_thresholds

# All these functions, and the main Segment() functions
# should be made general for implementation of different
# clonal marks. Or, if greater variation of implementation
# is required, make copies of all relevant functions and
# write a wrapper function with a "clonal mark" argument
# that then calls the correct analysis function.

def GetThresholdsPrepareRun_STAG(folder_in, file_in, folder_out):
    ## Make file name
    file_name   = folder_in + "/" + file_in + ".svs"
    
    ## Open slide and select regions for Crop and Zoom
    ########################################################
    obj_svs  = getROI_svs(file_name, get_roi_plot = False)
    ROI_crop = obj_svs.roi_full_thmb
    ROI_zoom = [] # obj_svs.chosenROI
    
    xyhw = auto_choose_ROI(file_name, deconv_mat_STAG, plot_images = False)
    img_for_thresh = getROI_img_vips(file_name, xyhw[0], xyhw[1])
    thresh_cut_nucl, thresh_cut_nucl_blur, th_clone = calculate_thresholds(img_for_thresh, deconv_mat_STAG)
    thresh_three = (thresh_cut_nucl, thresh_cut_nucl_blur, th_clone)

    ## Make directory, get indexes to tile image and save info
    ########################################################        
    # Make directory for images     
    full_folder_out = folder_out + '/Analysed_' + file_in
    try:
        os.mkdir(full_folder_out)
    except:
        pass
    ## Get indexes to split    # Test with smaller tiles by lowering max_num_pix
    all_indx = getIndexesTileImage(obj_svs.dims_slides[0], obj_svs.scalingVal, ROI_crop, max_num_pix  = 12500)      
    with open(full_folder_out + '/params.pickle', 'wb') as f:
        pickle.dump([folder_in, file_in, ROI_zoom, all_indx, deconv_mat_STAG, thresh_three], f)      

def SegmentFromFolder_STAG(folder_name):
    pkl_file    = open(folder_name + '/params.pickle', 'rb')
    folder_in, file_in, ROI_zoom, all_indx, deconv_mat_MAOA, thresh_cut = pickle.load(pkl_file)
    pkl_file.close()
    
    file_name   = folder_in + "/" + file_in + ".svs"
    
    crypt_features   = np.empty((0, 6))
    clone_features   = np.empty((0, 6))
    summary_features = np.empty((0, 4))
    img_all_clones     = []
    cluster_clones_all = []
    crypt_contours     = []
    patch_contours     = []    
    x_tiles = len(all_indx)
    y_tiles = len(all_indx[0])
    start_time = time.time()
    for i in range(x_tiles):
        for j in range(y_tiles):
            xy_vals     = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
            wh_vals     = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
            img         = getROI_img_vips(file_name, xy_vals, wh_vals)
            (img_segm_ii, clones_img_ii, crypt_features_ii, clone_features_ii, summary_ii, 
                 cluster_clones_ii, crypt_cnt_ii, mPAS_cluster_cnt_ii) = SegmentSTAG(img, thresh_cut)
            
            ## Add x, y tile offset to all contours (which have been calculated from a tile) for use in full image 
            crypt_cnt_ii        = add_offset(crypt_cnt_ii, xy_vals)
            mPAS_cluster_cnt_ii = add_offset(mPAS_cluster_cnt_ii, xy_vals)
            crypt_contours     += crypt_cnt_ii
            patch_contours     += mPAS_cluster_cnt_ii
            ## Append x, y location of that image
            cluster_clones_all += [xy_vals] + cluster_clones_ii
            img_all_clones     += clones_img_ii
            
            crypt_features     = np.vstack([crypt_features, crypt_features_ii]) 
            clone_features     = np.vstack([clone_features, clone_features_ii]) 
            summary_features   = np.vstack([summary_features,      summary_ii]) 
            print("Done " + str(i) + '.' + str(j) + " (" + str(x_tiles-1) + '.' + str(y_tiles-1) + ")")
            del(img, img_segm_ii)         
    
    ## Remove overlapping contours due to tiling effects
    crypt_contours, num_removed = remove_tiling_overlaps_knn(crypt_contours)
    print("Removing %d contours due to tiling overlaps." % num_removed)
    
    ## Reduce number of vertices per contour to save space/QuPath loading time
    crypt_contours = simplify_contours(crypt_contours)
    patch_contours = simplify_contours(patch_contours)
    
    # Array contains numCrypts, numGob, correctedNumGob, numGob/crypt, quantiles_gobSize (from 10% to 90%)
    np.savetxt(folder_name + '/summary_out.csv',    summary_features, delimiter=",")
    np.savetxt(folder_name + '/crypt_feat_out.csv',   crypt_features, delimiter=",")
    np.savetxt(folder_name + '/clone_feat_out.csv',  clone_features, delimiter=",")
    ## Order clones and make list
    if crypt_features.shape[0] != 0:
        frac_tot_crypts_wclones = (1.*clone_features.shape[0])/crypt_features.shape[0]
    else: 
        frac_tot_crypts_wclones = 0
    orderClones_Annotate_WriteImg(folder_name, img_all_clones, cluster_clones_all, frac_tot_crypts_wclones, maxSize = 200)
    print("Done " + file_in + " in " +  str((time.time() - start_time)/60.) + " min  =========================================")
    write_cnt_text_file(crypt_contours, folder_name + "/crypt_contours.txt")
    write_cnt_text_file(patch_contours, folder_name + "/clone_contours.txt")
  

    
    
