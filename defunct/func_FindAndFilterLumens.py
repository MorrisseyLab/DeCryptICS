# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:55:36 2015

@author: edward
"""

import cv2
from random import randint
import numpy as np
from cnt_Feature_Functions import st_3, st_7, st_5, contour_Area, contour_xy, filterList, remove_wiggles
from cnt_Feature_Functions import filterSmallArea, getContourWhiteBlobs, drawAllCont, plot_img, plotCnt, getContourWhiteHalos
from MixModel.mixModelThreeDataTypes_funcs import cbind, initGammas_kmeans, GibbsMixFunnel2Clusters3d_2, predictVals, indx_filterData, plotMAP_fit
from classContourFeat      import getAllFeatures
from joblib                import Parallel, delayed

def getForeground(nucl_thresh_aux):
    foreground = cv2.morphologyEx(  nucl_thresh_aux, cv2.MORPH_CLOSE,   st_3, iterations = 160) # st_5, iterations = 50)  
    foreground = cv2.morphologyEx(      foreground,   cv2.MORPH_OPEN,   st_3, iterations = 10) 
    foreground = cv2.morphologyEx(      foreground, cv2.MORPH_DILATE,   st_3, iterations = 10) 
    foreground = filterSmallArea(foreground, 5e5)  
    return foreground

def getContoursCandidateLumen(mpas_cnt_eroded, lum_cand, backgrd):
    ## Hard limits
    min_size_thresh = 400
    max_size_thresh = 3e5

    # D.K. these repeated Openings are trying to "snap off" the crypts whose halos aren't connected
    cnt_i0 = getContourWhiteBlobs(lum_cand, maxSize =  max_size_thresh, minSize = 500)
#    plotCnt(img, cnt_i0)
    lum_cand1  = cv2.morphologyEx( lum_cand, cv2.MORPH_OPEN,  st_7, iterations = 1)
    cnt_i1     = getContourWhiteBlobs(lum_cand1, maxSize =  max_size_thresh, minSize = 475)
#    plotCnt(img, cnt_i1)
    lum_cand1  = cv2.morphologyEx( lum_cand, cv2.MORPH_OPEN,  st_7, iterations = 2)
    cnt_i2     = getContourWhiteBlobs(lum_cand1, maxSize = max_size_thresh, minSize = 425)
#    plotCnt(img, cnt_i2)
    lum_cand1  = cv2.morphologyEx( lum_cand, cv2.MORPH_OPEN,  st_7, iterations = 3) #4
    all_crypts = cv2.subtract(lum_cand1,backgrd)
    cnt_i3     = getContourWhiteBlobs(all_crypts, maxSize = max_size_thresh, minSize = min_size_thresh)
#    plotCnt(img, cnt_i3)
   
    crypt_cnt_raw = cnt_i0 + cnt_i1 + cnt_i2 + cnt_i3 + mpas_cnt_eroded
    crypt_cnt_raw = [cnt_i for cnt_i in crypt_cnt_raw if contour_Area(cnt_i) > min_size_thresh ]
    crypt_cnt_raw = [cnt_i for cnt_i in crypt_cnt_raw if contour_Area(cnt_i) < max_size_thresh ]
    return crypt_cnt_raw

def rescueLostCrypts(nucl_thresh, indx_i):
    # This function uses the uninverted image, dilation and convex hull to find inner halos
    if (indx_i==0):    
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh, cv2.MORPH_CLOSE,  st_3, iterations=2)
        nucl_thresh_aux = filterSmallArea(nucl_thresh_aux, 350)
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_CLOSE,  st_3, iterations=3)
        nucl_thresh_aux = filterSmallArea(nucl_thresh_aux, 500)    
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_DILATE,  st_3, iterations=1)
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_CLOSE,  st_3, iterations=2)
    if (indx_i==1):        
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh, cv2.MORPH_CLOSE,  st_3, iterations=5)        
        nucl_thresh_aux = filterSmallArea(nucl_thresh_aux, 350)
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_CLOSE,  st_3, iterations=5)
        nucl_thresh_aux = filterSmallArea(nucl_thresh_aux, 500)
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_DILATE,  st_3, iterations=4)
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_CLOSE,  st_3, iterations=3)
    # Find contours round circular halos...
    halo_cnts = getContourWhiteHalos(nucl_thresh_aux, throw_outer = True)
    halo_cnts = [cv2.convexHull(i) for i in halo_cnts]
    # Throw small or large contours
    min_size_thresh = 450
    max_size_thresh = 3e5
    halo_cnts = [cnt for cnt in halo_cnts if contour_Area(cnt) > min_size_thresh ]
    halo_cnts = [cnt for cnt in halo_cnts if contour_Area(cnt) < max_size_thresh ]
    return halo_cnts
        
def getCntsSeveralThresh(thresh_cut_nucl_blur, blurred_img_nuc, pcnt_use, backgrd, mpas_thresh):
    cnt_all  = []
    indx_all = np.zeros(0, dtype = np.int)
    for pcnt_i,ii in zip(pcnt_use, range(len(pcnt_use))):
        #percent_change  = -0.2 # Lower thresh
        _, nucl_thresh  = cv2.threshold(blurred_img_nuc, int(thresh_cut_nucl_blur*(1 + pcnt_i)), 255, cv2.THRESH_BINARY)
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh, cv2.MORPH_OPEN,  st_3, iterations=1)
        ## Candidate lumens
        filt_nucl  = filterSmallArea(nucl_thresh_aux, 500)
        lum_cand   = 255 - filt_nucl
        cnt_out    = getContoursCandidateLumen(mpas_thresh, lum_cand, backgrd)
        cnt_all    += cnt_out
        indx_all   = np.concatenate([indx_all, ii*np.ones(len(cnt_out))])
    return cnt_all, indx_all

def filterGibbs(feat_cnt_nuc, plot_me = False): 
    ## Shift areas so as to start at size 1 
    xx_all          = cbind((feat_cnt_nuc.allHalo, feat_cnt_nuc.allMeanNucl, feat_cnt_nuc.allSizes-np.min(feat_cnt_nuc.allSizes)+1))
    indx_use_these  = indx_filterData(xx_all, perc_halo = 5, perc_nucl = 95, perc_area = 95, hard_lim_nucl = 0.5, hard_lim_halo = 0.3) # Remove outliers
    # If nothing left after filtering return empty    
    if  np.sum(indx_use_these) < 10:
        return [],[],[]

    xx_filt         = xx_all[indx_use_these,:] ## Filter outliers
    
    ## Subsample for speed max size 10000
    max_num = 10000
    if xx_filt.shape[0] > max_num:
        indx_subsmpl = np.random.choice(np.arange(xx_filt.shape[0]), size=max_num, replace=False, p=None) 
        xx_filt = xx_filt[indx_subsmpl,:]
    gamma_i    = initGammas_kmeans(xx_filt)

    gamma_all, mcmc_out =  GibbsMixFunnel2Clusters3d_2(xx_filt, gamma_i,  1000, 10)
    p_full, p_all = predictVals(mcmc_out, xx_all[:,0], xx_all[:,1], xx_all[:,2])
    if plot_me:
        plotMAP_fit(mcmc_out, xx_all[:,0], xx_all[:,1], xx_all[:,2], p_full > 0.5)
    return p_full > 0.5 , p_all, np.mean(mcmc_out,0)

## I remove from mem matrices that are passed as input, could cuase problems...
def GetContAndFilter_oneThresh(thresh_cut_nucl_blur, blurred_img_file, pcnt_i, mpas_cnt_eroded, indx_i, plot_me = False):
    # Load image using file name
    blurred_img_nuc     = np.load(blurred_img_file)
    # Correct for thresholds over the 255 limit
    if (thresh_cut_nucl_blur*(1 + pcnt_i)>=255): threshold = 0.5*(255+thresh_cut_nucl_blur)
    else: threshold = thresh_cut_nucl_blur*(1 + pcnt_i)
    # Threshold image
    _, nucl_thresh      = cv2.threshold(blurred_img_nuc, int(threshold), 255, cv2.THRESH_BINARY)
    del blurred_img_nuc
    if (indx_i==0): # blurred version
        # Join small breaks in halo
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh, cv2.MORPH_CLOSE,  st_3, iterations=3)        
        # Filter out small external nuclei
        nucl_thresh_aux = filterSmallArea(nucl_thresh_aux, 400)
        # Snap off any joining edges...
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_OPEN,  st_3, iterations=5)
        # ..but then dilate and close to create sturdy rings
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_DILATE,  st_3, iterations=1)        
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_CLOSE,  st_3, iterations=1)        
    if (indx_i==1): # non-blurred version
        # Snap off any joining edges...
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh, cv2.MORPH_OPEN,  st_3, iterations=4)
        # Filter out small external nuclei
        nucl_thresh_aux = filterSmallArea(nucl_thresh_aux, 500)
        # Join small breaks in halo
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_CLOSE,  st_3, iterations=5)
        # ..but then dilate and close to create sturdy rings
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_DILATE,  st_3, iterations=3)
        nucl_thresh_aux = cv2.morphologyEx(nucl_thresh_aux, cv2.MORPH_CLOSE,  st_3, iterations=3)

    # Invert to find candidate lumens
    lum_cand             = 255 - nucl_thresh_aux
    del nucl_thresh_aux
    backgrd              = np.load("backgrd.npy")
    cnt_out              = getContoursCandidateLumen(mpas_cnt_eroded, lum_cand, backgrd)
    del lum_cand
    # Rescue lost crypts using univerted image, concatonate
    halo_cnts = rescueLostCrypts(nucl_thresh, indx_i)
    cnt_out = cnt_out + halo_cnts
    ## If there are few crypt candidates ignore
    if len(cnt_out) < 10:
        return [], [], [], [], [], []
    nuclei_ch_raw        = np.load("nuclei_ch_raw.npy")  
    smallBlur_img_nuc    = np.load("smallBlur_img_nuc.npy")
#    plotCnt(nuclei_ch_raw, cnt_out)
    # Remove wiggles and smooth contours using area-based erosions
    cnt_out = [remove_wiggles(K, nuclei_ch_raw, indx_i) for K in cnt_out]
    # filter contours for too few points (squares and triangles!)
    cnt_out = [i for i in cnt_out if len(i)>4]
#    plotCnt(nuclei_ch_raw, cnt_out)
    feat_cnt_nuc2        = getAllFeatures(cnt_out, nuclei_ch_raw, backgrd, smallBlur_img_nuc)        
    del nuclei_ch_raw, backgrd, smallBlur_img_nuc
    
    indx_keep, all_probs, means_dists = filterGibbs(feat_cnt_nuc2, plot_me = plot_me) 
    crypt_cnt       = filterList(cnt_out, indx_keep)
    if plot_me:
        print("Option removed")
    img_plot = None
    return crypt_cnt, cnt_out, indx_keep, all_probs, img_plot, means_dists
  
def GetContAndFilter_TwoBlur(thresh_cut_list, blurred_img_list, percent_change_list, backgrd, mpas_thresh, nuclei_ch_raw, smallBlur_img_nuc, n_cores = 6):
    mpas_eroded        = cv2.morphologyEx(mpas_thresh, cv2.MORPH_ERODE, st_5, iterations=10)
    mpas_cnt_eroded, _ = cv2.findContours(mpas_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    del mpas_eroded
    
    ## Write matrices to file for memory usage efficiency
    np.save("blurred_img_nuc.npy",    blurred_img_list[0])
    np.save("smallBlur_img_nuc.npy",  blurred_img_list[1])
    np.save("backgrd.npy", backgrd)
    np.save("nuclei_ch_raw.npy", nuclei_ch_raw)
    blurred_img_list_files = ["blurred_img_nuc.npy", "smallBlur_img_nuc.npy"]
    run_indx = [0]*len(percent_change_list) + [1]*len(percent_change_list) # D.K.
    dbl_indx = percent_change_list + percent_change_list # Run twice for each indx 0, 1

    ## Make combination 
    results = Parallel(n_jobs = n_cores)(delayed(GetContAndFilter_oneThresh)(thresh_cut_list[indx_i], blurred_img_list_files[indx_i], 
                       percent_change_i, mpas_cnt_eroded, indx_i) for percent_change_i,indx_i in zip(dbl_indx, run_indx))
    return results

def plot_multi_thresh(res_list, img, plot_indx = None, ind = 0):
    if plot_indx is not None: res_list = [res_list[plot_indx]]
    img_plot = img.copy()
    num_runs = len(res_list)
    for res_i,ii in zip(res_list, range(num_runs)):
        drawAllCont(img_plot, res_i[ind], -1, (randint(0,255), randint(0,255),  randint(0,255)), 7+2*(num_runs-ii))
    plot_img(img_plot, hold_plot=True)

def mergeAllContours(res_list,size_img):
    bin_image = np.zeros(size_img, np.uint8)
    num_runs  = len(res_list)
    for res_i,ii in zip(res_list, range(num_runs)):        
        drawAllCont(bin_image, res_i[0], -1, 255, -1)
    all_cont, _ = cv2.findContours(bin_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    return all_cont

