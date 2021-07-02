#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:48:59 2021

@author: edward
"""

import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from training.read_svs_class import svs_file_w_labels



class correct_crypt_gui:
    def __init__(self, img_i, crypt_cnt, sld_inf_i):
        self.img_i, self.crypt_cnt, self.sld_inf_i = img_i, crypt_cnt, sld_inf_i
        self.clone_im = img_i.copy()
        self.sld_inf_i = self.sld_inf_i.astype(np.int32)
        self.cols_all = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255)]

        self.plot_clickable_image()

    # initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not
    def click_and_flipcrypt(self, event, x, y, flags, param):
        # grab references to the global variables
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONUP:   
            xy_click = np.array([x, y]).reshape([1,2])
            e_dist = euclidean_distances(xy_click, self.sld_inf_i[:, 0:2])
            indx_update = np.argmin(e_dist)
            new_val = self.sld_inf_i[indx_update, 3] + 1
            if new_val > 3:new_val = 0
            self.sld_inf_i[indx_update, 3] = new_val
            col_i   = self.cols_all[self.sld_inf_i[indx_update, 3]]
            self.clone_im = cv2.drawContours(self.clone_im, [self.crypt_cnt[indx_update]], -1, col_i, 2)
            cv2.imshow("image", self.clone_im)    
    
    def plot_clickable_image(self):
        # global sld_inf_i, crypt_cnt, clone, cols_all
        for ii, crypt_i in enumerate(self.crypt_cnt):
            col_i   = self.cols_all[self.sld_inf_i[ii, 3]]
            self.clone_im = cv2.drawContours(self.clone_im, [crypt_i], -1, col_i, 2)
            
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_flipcrypt)
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.clone_im)
            key = cv2.waitKey(1) & 0xFF 
            # print(key)             
           	# if the 'c' key is pressed, break from the loop
            if key == ord("c"):
                self.sld_inf_i = None
                break
            elif key != ord("c") and key!=255:
                break
        # close all open windows
        cv2.destroyAllWindows()


# Make a clickable image
# adapt to patches, show them only once
# shift click to scroll through options: clone, crypt, partial, remove contour 
# Make curated list

folder_out    = "/home/edward/WIMM/Decryptics_train/decryptics_code/manual_curation_files/"
imgpaths      = []
imgpaths      = imgpaths + glob.glob("/home/edward/WIMM/Decryptics_train/train/KM*/*.svs")
# imgpaths = imgpaths + glob.glob("/home/doran/Work/images/Anne-Claire_curated_2021/HR*/*.svs")
extraimgpaths = glob.glob("/home/edward/WIMM/Decryptics_train/train/extra/*.svs")
files_all     = imgpaths + extraimgpaths

already_curated = pd.read_csv(folder_out + 'curated_files_summary.txt', names = ["file_name", "slide_crtd"])
files_all = [x for x in files_all if x not in list(already_curated["file_name"])]

file_paths_filt = pd.read_csv("filtered_list_r.csv")
train_imgpaths  = file_paths_filt.sample(200, random_state=222)

files_all = [x for x in files_all if x in list(train_imgpaths["file_names"])]
# len([x for x in train_imgpaths["file_names"] if x in list(already_curated["file_name"])])

for file_i in files_all:
    # file_i          = files_all[4] 
    sld_i           = svs_file_w_labels(file_i, 1024, 1)
    sld_dat_i       = sld_i.sld_dat
    cl_inds         = np.where(sld_dat_i[:,3]>0)[0]
    print([file_i, len(cl_inds), sld_i.mark])
    updated_info = -1*np.ones([1,5])
    for cln_i in cl_inds:
        if (cln_i==updated_info[:,4]).any():
            continue
        img_i, crypt_cnt, sld_inf_i = sld_i.fetch_crypt(cln_i, ret_info = True)
        gui_obj = correct_crypt_gui(img_i, crypt_cnt, sld_inf_i)
        
        if gui_obj.sld_inf_i is None:
            # Abort and flag
            break        
        else:
            updated_info = np.vstack([updated_info, gui_obj.sld_inf_i]) 
    
    if gui_obj.sld_inf_i is None:
        with open(folder_out + 'curated_files_summary.txt', 'a') as file:
            file.write(file_i + "," + "cancel\n")
    else:
        # Remove dummy first row
        updated_info = updated_info[1:,:]
        # Make 0 - crypt, 1 - clone, 2 - partial, -1 remove
        updated_info[updated_info[:,3] > 2,3] = -1
        indx_update = updated_info[:,4].astype(np.int32)
        sld_dat_i[indx_update, 3] = updated_info[:, 3]
        file_out_cur = file_i.split("/")[-1].replace(".svs", "")
        file_out_cur = folder_out + file_out_cur + "_curtd.csv"
        pd.DataFrame(sld_dat_i[:, 0:4]).to_csv(file_out_cur)
        with open(folder_out + 'curated_files_summary.txt', 'a') as file:
            file.write(file_i + "," + file_out_cur + "\n")
        
    value = input("had enough [y]?\n")
    if value=="y":
        break
    else:
        print("continue!")

# resolution thing 
# file_i = '/home/edward/WIMM/Decryptics_train/train/KM16/KM16S_446554.svs'


# ## Has it worked?
# already_curated = pd.read_csv(folder_out + 'curated_files_summary.txt', names = ["file_name", "slide_crtd"])
# i = 4
# sld_info_cur = pd.read_csv(already_curated["slide_crtd"][i])
# sld_i        = svs_file2(already_curated["file_name"][i], 1024, 1)
# sld_dat_i    = sld_i.sld_dat[:, 0:-1]
# np_curr      = np.array(sld_info_cur)[:, 1:]
# diff_rows    = np.where(np.sum(sld_dat_i - np_curr, 1))
# sld_dat_i[diff_rows]
# np_curr[diff_rows]
# img_i = sld_i.fetch_crypt(diff_rows[0][0], contour = True)
# plot_img(img_i)



## Timgin tests
# # cv subsample # 54.2 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_subsample_cv = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) # 63 ms 
# plot_img(uu_subsample_cv)

# # resize # 54.8 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_resize = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) 
# plot_img(uu_resize)

# plot_img([uu_subsample_cv, uu_resize])

# # subsample # 53.4 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_subsample = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4)
# plot_img(uu_subsample)

# # shrink # 57 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_shrink = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) 
# plot_img(uu_shrink)

# # reduce # 55.2 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_reduce = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) 
# plot_img(uu_reduce)

# # no resize # 62.2 ms 
# kk = svs_file2(file_name, 1024, 1)
# uu_noresize = kk.fetch_crypt(4)
# %timeit uu = kk.fetch_crypt(4) 
# plot_img(uu_noresize)



