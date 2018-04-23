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
from automaticThresh_func       import auto_choose_ROI, calculate_thresholds
from Segment_clone_from_crypt   import find_clones
from SegmentCrypts              import Segment_crypts
from deconv_mat                 import *

def GetThresholdsPrepareRun(folder_in, file_in, folder_out, deconv_mat):
    ## Make file name
    file_name   = folder_in + "/" + file_in + ".svs"
    
    ## Open slide and select regions for Crop and Zoom
    ########################################################
    obj_svs  = getROI_svs(file_name, get_roi_plot = False)
    ROI_crop = obj_svs.roi_full_thmb
    ROI_zoom = [] # obj_svs.chosenROI
    
    xyhw = auto_choose_ROI(file_name, deconv_mat, plot_images = False)
    img_for_thresh = getROI_img_vips(file_name, xyhw[0], xyhw[1])
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
    
    ## Choose deconv mat
    if (clonal_mark_type=="P"): deconv_mat = deconv_mat_KDM6A # Don't have an example of this for a deconvolution matrix        
    if (clonal_mark_type=="N"): deconv_mat = deconv_mat_KDM6A
    if (clonal_mark_type=="PNN"): deconv_mat = deconv_mat_MPAS
    if (clonal_mark_type=="NNN"): deconv_mat = deconv_mat_MAOA
    
    ## Set up storage structures
    crypt_contours  = []
    frac_halo       = np.array([])
    frac_halogap    = np.array([]) 
    clone_content   = np.array([])
    x_tiles = len(all_indx)
    y_tiles = len(all_indx[0])
    start_time = time.time()
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
            frac_halo       = np.hstack([frac_halo    , clone_features[0]])
            frac_halogap    = np.hstack([frac_halogap , clone_features[1]])
            clone_content   = np.hstack([clone_content, clone_features[2]])
            print("Done " + str(i) + '.' + str(j) + " (" + str(x_tiles-1) + '.' + str(y_tiles-1) + ")")
            del img         
    
    ## Remove overlapping contours due to tiling effects and the corresponding clone_features
    print("Of %d contours..." % len(crypt_contours))
    crypt_contours, kept_indices = remove_tiling_overlaps_knn(crypt_contours)
    print("...Keeping only %d due to tiling overlaps." % kept_indices.shape[0])
    frac_halo       =     frac_halo[kept_indices]
    frac_halogap    =  frac_halogap[kept_indices]
    clone_content   = clone_content[kept_indices]
    clone_channel_feats = (frac_halo , frac_halogap , clone_content)
    clone_contours, full_partial_statistics = find_clones(crypt_cnt, clone_channel_feats, clonal_mark_type, numIQR=2)
    np.savetxt(folder_name + '/clone_statistics.csv', full_partial_statistics, delimiter=",")

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
    
    
