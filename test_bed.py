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
img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_quarter_threequarter_partrials_and_full.png'
img_file = imgfolder+'negspace_Clone_test_images/raw_images/MAOA_586574_double_clone.png'
img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_coldeconv_sample.png'

deconv_mat = deconv_mat_MPAS
clonal_mark_type = "P"
big_img = cv2.imread(img_file)
img = big_img
thresh_cut = None

# Regions from svs slides
#deconv_mat = deconv_mat_MPAS
clonal_mark_type = "BN"
img_file = imgfolder+'MAOA_March2018/raw_images/586572.svs'
obj_svs  = getROI_svs(img_file)
big_img = obj_svs.getROI_img()
img = big_img

## checking zoom levels of pyramid images for different sized svs files
img_file1 = imgfolder+'mPAS_subset_test/raw_images/575833.svs'
img_file2 = imgfolder+'mPAS_subset_test/raw_images/643873.svs'
img_file3 = imgfolder+'mPAS_WIMM/raw_images/575837.svs'
img_file4 = imgfolder+'mPAS_WIMM/raw_images/620681.svs'
img_file5 = imgfolder+'mPAS_WIMM/raw_images/643878.svs'
obj_svs1  = getROI_svs(img_file1, get_roi_plot = False)
obj_svs2  = getROI_svs(img_file2, get_roi_plot = False)
obj_svs3  = getROI_svs(img_file3, get_roi_plot = False)
obj_svs4  = getROI_svs(img_file4, get_roi_plot = False)
obj_svs5  = getROI_svs(img_file5, get_roi_plot = False)
print(obj_svs1.dims_slides[0][0]/obj_svs1.dims_slides[1][0])
print(obj_svs1.dims_slides[0][1]/obj_svs1.dims_slides[1][1])
print(obj_svs2.dims_slides[0][0]/obj_svs2.dims_slides[1][0])
print(obj_svs2.dims_slides[0][1]/obj_svs2.dims_slides[1][1])
print(obj_svs3.dims_slides[0][0]/obj_svs3.dims_slides[1][0])
print(obj_svs3.dims_slides[0][1]/obj_svs3.dims_slides[1][1])
print(obj_svs4.dims_slides[0][0]/obj_svs4.dims_slides[1][0])
print(obj_svs4.dims_slides[0][1]/obj_svs4.dims_slides[1][1])
print(obj_svs5.dims_slides[0][0]/obj_svs5.dims_slides[1][0])
print(obj_svs5.dims_slides[0][1]/obj_svs5.dims_slides[1][1])


# run several svs batches
run run_script.py /home/doran/Work/images/ NONO_March2018 N D
run "run_script.py" /home/doran/Work/images/ STAG_March2018 N D
run "run_script.py" /home/doran/Work/images/ KDM6A_March2018 N D
run "run_script.py" /home/doran/Work/images/ MAOA_March2018 N D

