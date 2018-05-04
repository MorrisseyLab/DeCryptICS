# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:47:59 2017

@author: edward
"""
import numpy as np
import cv2
from MiscFunctions import plot_img, getROI_img_vips
from GUI_ChooseROI_class        import getROI_svs
#from Clonal_Stains.SegmentCryptAndCloneMPAS import SegmentMPAS
from deconv_mat import deconv_mat_AB, deconv_mat_betaCat, deconv_mat_MAOA, deconv_mat_MPAS
import os
from automaticThresh_func import auto_choose_ROI
from deconv_mat import *
imgfolder = os.path.expanduser("~/Work/images/")


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
#img_file = imgfolder+'mPAS_Clone_test_images/raw_images/img_mPAS_devel_PurpleClone.jpg' # Great (couple of FP)

## Negative space clone test images
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_clone_cluster.png' # finds right number
img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_patch_with_blobs.png' # finds either too many or too few?
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_quarter_threequarter_partrials_and_full.png' # good, obvious split between full and two partials
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_edge_clone_cluster.png' # finds way too many -- but image too zoomed out?
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_single_clone.png'
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_full_and_partial.png'
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_two_clones_dark_halos.png'
img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_large_patch.png'

img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_1024_full_and_threequarter.png'
cnt_file = imgfolder+'negspace_Clone_test_images/Analysed_slides/contours_KDM6A_1024_full_and_threequarter.txt'
#img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_1024_quarter_partial.png'
#cnt_file = imgfolder+'negspace_Clone_test_images/Analysed_slides/contours_KDM6A_1024_quarter_partial.txt'
cnts = read_cnt_text_file(cnt_file)


img_file = imgfolder+'negspace_Clone_test_images/raw_images/KDM6A_quarter_threequarter_partrials_and_full.png'
img_file = imgfolder+'negspace_Clone_test_images/raw_images/MAOA_586574_double_clone.png'

#deconv_mat = deconv_mat_KDM6A
clonal_mark_type = "N"
big_img = cv2.imread(img_file)
img = big_img
thresh_cut = None

# choose region from svs with correct zoom
svsim = getROI_svs(imgfolder+'KDM6A_March2018/raw_images/642739.svs')
img = svsim.getROI_img()

       
 
# STAG snapshots
#img_file = imgfolder+'STAG_March2018/raw_images/601166_snapshots/img1.png'

## Notes: 
## img_mPAS_devel_PossibleWeakCell.jpg -- Larger blur for nucl?  -------------------

#big_img = cv2.imread(img_file)
#img = big_img
#thresh_cut = None
#uu = SegmentImage(big_img, deconv_mat_MPAS)
#plot_img((uu[0], big_img,))

## Running all the example files
import glob, os
from MiscFunctions import write_cnt_text_file
batch_ID = "/mPAS_Clone_test_images/"
folderimg = imgfolder + batch_ID + "/raw_images/"
rawims = glob.glob(os.path.expanduser(folderimg + "*.jpg"))
img_ids = [id[len(folderimg):-4] for id in rawims]
folder_out = imgfolder + batch_ID + "/Analysed_slides/"
try:
    os.mkdir(folder_out)
except:
    pass
for i in range(len(rawims)):
    big_img = cv2.imread(rawims[i])
    thresh_cut = None
    folder_name = folder_out + "/" + img_ids[i] + "/"
    try:
        os.mkdir(folder_name)
    except:
        pass
    uu = SegmentImage(big_img, deconv_mat_MPAS)
    crypt_contours = uu[-2]
    write_cnt_text_file(crypt_contours, folder_name + "/crypt_contours.txt")
    
    
    

#plot_img(img, hold_plot=True)
plot_img((uu[0], big_img,))
print("Crypts " + str(uu[4][0]))
print("Clones " + str(uu[4][1]))
print("Patches " + str(uu[4][2]))
print("Single cell " + str(uu[4][3]))
 


