# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:26:20 2015

@author: edward
"""
import cv2
import numpy as np
from cnt_Feature_Functions import contour_xy, formatString

def getBoundingBox(cnt_i, expand_box = 50):   
    roi        = cv2.boundingRect(cnt_i) ## Returnx x y w h   
    roi        = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    ## Correct zero, no need to correct other boundaries as we pass x1:x2 which works if x2 overshoots 
    roi[roi<0] = 0
    return roi

def sub_img(img1, roi, pad_width = None, pad_top = 50):
    img_ret = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]   
    if pad_width is not None:
        img_ret = pad_image(img_ret, pad_width = pad_width, pad_top=pad_top)
    return img_ret

def resize_img(img, size_out):
    return cv2.resize(img, size_out, interpolation = cv2.INTER_AREA)

def pad_image(img_i, pad_width = 10, pad_top = 50):
    return cv2.copyMakeBorder(img_i,pad_width + pad_top,pad_width,pad_width,pad_width,cv2.BORDER_CONSTANT,value=[0,0,0])

def CloneSizes_Cluster_i(clone_clust_i, clust_cnt_i, i):
    x_y_cont = contour_xy(clust_cnt_i)
    ## Find single cells?
    clones_concat = formatString(clone_clust_i[:,1])
    # clust_num, x, y, clone f1;..;clone fn
    return (int(x_y_cont[0]), int(x_y_cont[1]), int(i), clones_concat)

def AnnotateImage(cluster_clone_sizes, clone_i):
    x_start = int(clone_i.shape[1]/2)
    cv2.putText(clone_i, str(int(cluster_clone_sizes[2])) + "-", (x_start,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    cv2.putText(clone_i, cluster_clone_sizes[3], (x_start + 25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
    return clone_i

def expandROI_prod(roi, expand_box):
    exp_1  = roi[2]*expand_box
    exp_2  = roi[3]*expand_box
    disp_1 = np.int(exp_1/3)
    disp_2 = np.int(exp_2/3)    
    roi        = np.array((roi[0]-(disp_1), roi[1]-(disp_2),  exp_1, exp_2))
    ## Correct zero, no need to correct other boundaries as we pass x1:x2 which works if x2 overshoots 
    roi[roi<0] = 0
    return roi
    
     
def makeImageClones(clone_features, mPAS_cluster_cnt, crypt_cnt, img, img_plot):
#    size_each          = (500, 290)
    size_each          = (700, 240)
    all_clust_nums     = np.sort(np.unique(clone_features[:,0]))
    clones_img_all     = []
    cluster_clones_all = []
    if len(mPAS_cluster_cnt) > 600: return clones_img_all, cluster_clones_all ## Abort if too many clones
    ## Make a list of bounding boxes
    for i in all_clust_nums:
        clone_clust_i = clone_features[clone_features[:,0]==i,:]
        max_intens_i  = np.max(clone_clust_i[:,5])
        max_frac_i    = np.max(clone_clust_i[:,1])
        clust_cnt_i   = mPAS_cluster_cnt[int(i)]
        ## If a cluster of crypts, just use that
        if clone_clust_i.shape[0] > 1:
            roi_i = getBoundingBox(clust_cnt_i,  expand_box = 50)
        else:        
            crypt_indx_i = int(clone_clust_i[0, 4])
            roi_i = getBoundingBox(crypt_cnt[crypt_indx_i],  expand_box = 50)
        # Add raw image
        clone_i_raw   = sub_img(img, roi_i, pad_width = 10)
        clone_i_detct = sub_img(img_plot, roi_i, pad_width = 10)   
        size_raw      = clone_i_raw.shape
        # Add zoomed image
        clone_zoom_out   = sub_img(img, expandROI_prod(roi_i, 3), pad_width = 0, pad_top = 0)
        clone_zoom_out   = pad_image(resize_img(clone_zoom_out, (size_raw[1]-2*10, size_raw[0]-2*10-50)))
        ## Pad with col, Concatenate, resize and join to other clones
        clone_i     = np.concatenate((clone_zoom_out, clone_i_raw, clone_i_detct), axis=1)
        clone_i     = resize_img(clone_i, size_each)
#        plot_img(clone_i, hold_plot=True)

        # Get clone sizes
        cluster_clone_sizes = CloneSizes_Cluster_i(clone_clust_i, clust_cnt_i, i)
        # Overlay clone sizes
#        clone_i     = AnnotateImage(cluster_clone_sizes, clone_i)
#        plot_img(clone_i, hold_plot=True)
#        clones_img_all = np.concatenate((clones_img_all, clone_i), axis=0) 
        clones_img_all.append([clone_i, max_intens_i, max_frac_i, cluster_clone_sizes[3]])
        cluster_clones_all.append(cluster_clone_sizes)
#        str_clust      = str(cluster_clone_sizes) + "\n"
#        str_clust      = str_clust.replace("(","").replace(")","")
#        cluster_clones_all += str_clust
    return clones_img_all, cluster_clones_all

#clones_img_all, cluster_clones_all = makeImageClones(clone_features, mPAS_cluster_cnt, crypt_cnt, img, img_plot)      
#plot_img(clones_img_all, hold_plot=True)

