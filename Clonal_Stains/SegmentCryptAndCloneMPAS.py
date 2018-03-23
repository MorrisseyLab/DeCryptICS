# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:12:15 2015

@author: edward
"""
## Main function for segmentation. Takes as input an image, a set of thresholds (if None given it will calculate them) and
## colour deconvolution matrix

import cv2
import numpy as np
from MiscFunctions            import col_deconvol, col_deconvol_and_blur, plot_img #, plotImageAndFit
from cnt_Feature_Functions    import filterSmallArea, st_3, st_5, drawAllCont
from cnt_Feature_Functions    import getPercentileInts, filterSmallArea_outer, contour_MajorMinorAxis
from cnt_Feature_Functions    import joinContoursIfClose, plotCntAndFeat, plotCnt, plotThrownCnts
from classContourFeat         import getAllFeatures
from deconv_mat               import deconv_mat_MPAS
from makeCloneImage           import makeImageClones
from func_FindAndFilterLumens import mergeAllContours
from func_FindAndFilterLumens import GetContAndFilter_TwoBlur, plot_multi_thresh
from multicore_morphology     import getForeground_mc
from automaticThresh_func     import calculate_thresholds
from devel_knn_prune          import prune_contours_knn, drop_broken_runs, prune_attributes
from devel_knn_prune          import prune_minoraxes, tukey_outliers_above, tukey_outliers_below, tukey_lower_thresholdval, tukey_upper_thresholdval
from Clonal_Stains.mPAS_Segment_Clone       import get_mPAS_Stains2
from Segment_clone_from_crypt               import find_clones_posspace
from Clonal_Stains.func_mPAS_crypt_clone    import getCloneFeatures, getCryptFeatures

## If thresh_cut is None, thresholds will be calculated from the image
def SegmentMPAS(img, thresh_cut):
    
    ## Avoid problems with multicore
    cv2.setNumThreads(0)       
          
    ## Colour Deconvolve to split channles into nuclear and clone stain
    ## Blur and threshold image
    ####################################################################

    ## If thesh_cut is a list then unpack, otherwise calculate theresholds
    if thresh_cut is None:
        thresh_cut_nucl, thresh_cut_nucl_blur, th_clone = calculate_thresholds(img, deconv_mat_MPAS)
    else:
        thresh_cut_nucl, thresh_cut_nucl_blur, th_clone = thresh_cut #thresh_stringent, thresh_lax 

    ## Colour deconvolution
    smallBlur_img_nuc, blurred_img_nuc, blurred_img_clone = col_deconvol_and_blur(img, deconv_mat_MPAS, (11, 11), (37, 37), (27, 27))
    #plot_img((smallBlur_img_nuc, blurred_img_nuc, blurred_img_clone), hold_plot=True)

    # Threshold
    _, nuclei_ch_raw = cv2.threshold( smallBlur_img_nuc,      thresh_cut_nucl, 255, cv2.THRESH_BINARY)
    _, nucl_thresh   = cv2.threshold(   blurred_img_nuc, thresh_cut_nucl_blur, 255, cv2.THRESH_BINARY)

    ## Clone segment 
    ###########################################
    clone_thresh = get_mPAS_Stains2(blurred_img_clone, smallBlur_img_nuc, th_clone)
    _, clone_ch_raw = cv2.threshold(blurred_img_clone, th_clone, 255, cv2.THRESH_BINARY)
          
    ## Clean up, filter nuclei and get foreground from nuclei
    ###########################################
    ## Small erosion
    nuclei_ch_raw   = cv2.morphologyEx(nuclei_ch_raw, cv2.MORPH_OPEN,  st_3, iterations=1)
    nucl_thresh_aux = cv2.morphologyEx(  nucl_thresh, cv2.MORPH_OPEN,  st_3, iterations=1)

    foreground = getForeground_mc(nucl_thresh_aux)    
    backgrd    = 255 - foreground

    ## Finish if there is no foreground
    ###########################################    
    del foreground, nucl_thresh_aux
    
    ## Segment crypts lumen
    ###########################################
    qq_both = GetContAndFilter_TwoBlur([thresh_cut_nucl_blur, thresh_cut_nucl], [blurred_img_nuc, smallBlur_img_nuc], [-0.2,-0.1, 0, 0.1], 
                                       backgrd, clone_thresh, nuclei_ch_raw, smallBlur_img_nuc, n_cores = 8)
    #plot_multi_thresh(qq_both , img)
    #print("Number of runs = %d" % len(qq_both))
#    kkk = 0
#    for run in qq_both:
#        print("Run %d" % kkk)
#        feat_cnt_nuc_m = getAllFeatures(run[1], nuclei_ch_raw, backgrd, smallBlur_img_nuc)
#        plotCntAndFeat(img, run[1], np.vstack([feat_cnt_nuc_m.allHalo, feat_cnt_nuc_m.allSizes, feat_cnt_nuc_m.allMeanNucl]).T, run[2])
#        feat_cnt_nuc_m.plotFeatures()
#        feat_cnt_nuc_m.plotHistograms()
#        kkk+=1
    num_contours_kept = 0
    for run in qq_both:
        num_contours_kept += len(run[0])    
    if (not num_contours_kept==0):
        med_minor_axis = [np.median([contour_MajorMinorAxis(contours_i)[1] for contours_i in qq_i[0]]) for qq_i in qq_both if qq_i[0]!=[]]
        qq_new = drop_broken_runs(qq_both, med_minor_axis, nuclei_ch_raw)
        if (not len(qq_new)==0):
            crypt_cnt  = mergeAllContours(qq_new, img.shape[0:2])    
            features = getAllFeatures(crypt_cnt, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
            #plotCntAndFeat(img, crypt_cnt, np.vstack([feat_cnt_nuc3.allHalo, feat_cnt_nuc3.allHaloGap, feat_cnt_nuc3.allEcc]).T)
            #plotCntAndFeat(img, crypt_cnt, np.vstack([feat_cnt_nuc3.allSolid, feat_cnt_nuc3.allMeanNucl, feat_cnt_nuc3.allMinorAxis]).T)
        
            # Prune on individual attributes
            minor_ax_thresh_individualcnts = max(med_minor_axis)*5./12.
            prune_cnts = prune_minoraxes(crypt_cnt, features, minor_ax_thresh_individualcnts)
            features = getAllFeatures(prune_cnts, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
            #plotCntAndFeat(img, prune_cnts, np.vstack([features.allSolid, features.allHaloGap, features.allEcc]).T)
            
            # Combining the above prunings:
            prune_cnts = prune_attributes(prune_cnts, features)
            features = getAllFeatures(prune_cnts, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
            #plotCntAndFeat(img, prune_cnts, np.vstack([feat_cnt_nuc5.allSolid, feat_cnt_nuc5.allMeanNucl]).T)
            
            # If broken, check for lots of small nonsense contours
            if (np.median(features.allSizes)<800):# or np.median(feat_cnt_nuc5.allMeanNucl)>0.25):
                prune_cnts = []
                print("Broken: too many small nonsense contours found!")
            
            # Prune on knn attributes
            if (len(prune_cnts)>6):
                prune_cnts = prune_contours_knn(prune_cnts, features, nuclei_ch_raw, backgrd, smallBlur_img_nuc, stddevmult = 3.6, nn = 7)
                features = getAllFeatures(prune_cnts, nuclei_ch_raw, backgrd, smallBlur_img_nuc)
                #plotCntAndFeat(img, prune_cnts, np.vstack([feat_cnt_nuc8.allHalo, feat_cnt_nuc8.allSolid, feat_cnt_nuc8.allHaloGap, feat_cnt_nuc8.allEcc, feat_cnt_nuc8.allSizes]).T)
        if (len(qq_new)==0):
            num_contours_kept = 0
    if (num_contours_kept==0): 
        prune_cnts = []
        
    crypt_cnt = prune_cnts

    ## Make binary image from crypt contours
    seg_crypts = np.zeros(backgrd.shape, np.uint8)                                           
    drawAllCont(seg_crypts, crypt_cnt, -1, 255, -1)
        
    ## Process mPAS
    ###########################################
    ## Filter mPAS with no inside lumen
    # Erode lumen    
    crypts_erode        = cv2.morphologyEx(seg_crypts, cv2.MORPH_ERODE,  st_5, iterations=2)
    crypts_erode_cnt, _ = cv2.findContours(crypts_erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]  
    # Filter mpas not inside eroded crypts
    clone_thresh  = cv2.bitwise_and(seg_crypts, clone_thresh)
    clone_thresh  = filterSmallArea_outer(clone_thresh, 200)
    clone_cnt, _  = cv2.findContours(clone_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:] 
    
    vec_intens_clone = getPercentileInts(clone_thresh, blurred_img_clone, 90)
    
    # Find clone contours by looking for clone signal inside crypts
    clone_cnt = find_clones_posspace(crypt_cnt, clone_ch_raw, backgrd, smallBlur_img_nuc)
    
    # Join neighbouring clones to make cluster (clone patches that originate via crypt fission)
    # Don't do this if more than 25% of crypts are positive as it's hom tissue
    if len(clone_cnt) < 0.25*len(crypt_cnt) and len(clone_cnt)>0:
        clone_cluster_cnt = joinContoursIfClose(clone_cnt, max_distance = 400)
    else:
        clone_cluster_cnt = clone_cnt[:]
    
    ## Gather all and summarise for output
    ###########################################
    crypt_features = getCryptFeatures(crypt_cnt) 
    clone_features = getCloneFeatures(clone_cnt, crypts_erode_cnt, crypt_cnt, clone_cluster_cnt, clone_thresh, vec_intens_clone)
            
    num_crypts         =              crypt_features.shape[0]
    num_clones         = len(np.unique(clone_features[:, 4]))
    num_clusters       = len(np.unique(clone_features[:, 0]))
    num_put_singleCell = np.sum(np.bitwise_and(clone_features[:, 1] < 0.15, clone_features[:, 2] < 2000))
    
    img_plot = img.copy()
    drawAllCont(img_plot,         crypt_cnt, -1, (255, 0,   0), 7) 
    drawAllCont(img_plot,  clone_cluster_cnt, -1, (  0, 0, 255), 7)
    clones_img_all, cluster_clones_all = makeImageClones(clone_features, clone_cluster_cnt, crypt_cnt, img, img_plot)    
    
    print("Crypts %d" % num_crypts)
    print("Clones %d" % num_clones)
    print("Patches %d" % num_clusters)
    print("Single cell %d" % num_put_singleCell)
    
    return img_plot, clones_img_all, crypt_features, clone_features, [num_crypts, num_clones, num_clusters, num_put_singleCell], cluster_clones_all, crypt_cnt, clone_cluster_cnt
    