# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:47:59 2017

@author: edward
"""
import numpy as np
import cv2
from MiscFunctions import plot_img, getROI_img_vips
from GUI_ChooseROI_class        import getROI_svs
from deconv_mat import deconv_mat_AB, deconv_mat_betaCat, deconv_mat_MAOA, deconv_mat_MPAS
import os
from automaticThresh_func import auto_choose_ROI
from deconv_mat import *
imgfolder = os.path.expanduser("~/Work/images/")


#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_errant_staining_no_real_clone.jpg'
img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_singleCell.jpg'
img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_TwoClusters_FPs.jpg' # A few FP crypts
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel.jpg'                 # A few FP crypts
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_SingleClone.jpg'     # FP crypts and clones
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_Nothing1.jpg'         # FP + FN
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_Nothing2.jpg'         ## Loads of FP's + flipped dists!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_LargeBlock_nothing.jpg' # One FP crypt, all good.
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_LargeBlock_clone.jpg'  # Perfect! (1 FP ..)
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_PossibleWeakCell.jpg' ## Missed the clone !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_WeakPink.jpg'  ## No crypts found !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_PurpleClone.jpg' # Great (couple of FP)

## Negative space clone test images
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_clone_cluster.png' # finds right number
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_patch_with_blobs.png' # finds either too many or too few?
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_quarter_threequarter_partrials_and_full.png' # good, obvious split between full and two partials
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_edge_clone_cluster.png' # finds way too many -- but image too zoomed out?
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_single_clone.png'
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_full_and_partial.png'
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_two_clones_dark_halos.png'
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_large_patch.png'

img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_1024_full_and_threequarter.png'
cnt_file = imgfolder+'negspace_Clone_test_images/Analysed_slides/contours_KDM6A_1024_full_and_threequarter.txt'
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_1024_quarter_partial.png'
#cnt_file = imgfolder+'negspace_Clone_test_images/Analysed_slides/contours_KDM6A_1024_quarter_partial.txt'
cnts = read_cnt_text_file(cnt_file)

img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_quarter_threequarter_partrials_and_full.png'
img_file = imgfolder+'negspace_Clone_test_images/raw_images/MAOA_586574_double_clone.png'

#deconv_mat = deconv_mat_MPAS
clonal_mark_type = "PNN"
big_img = cv2.imread(img_file)
img = big_img
thresh_cut = None

# Regions from svs slides
img_file = imgfolder+'KDM6A_March2018/raw_images/642739.svs'
obj_svs  = getROI_svs(img_file)
big_img = obj_svs.getROI_img()
img = big_img

       
 
 


