# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:39:01 2015

@author: edward
"""

from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
#import pyvips
import openslide as osl
import h5py
import os, errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def add_offset(contour_list, xy_offset):
    cnt_list_out = []
    for elem_i in contour_list:
        elem_i[:,0,0] += xy_offset[0]
        elem_i[:,0,1] += xy_offset[1]
        cnt_list_out.append(elem_i)
    return cnt_list_out

def rescale_contours(contour_list, scaling_val):
    cnt_list_out = []
    for cc in contour_list:
      a = (cc*scaling_val).astype(np.int32)
      cnt_list_out.append(a)
    return cnt_list_out

def convert_to_local_clone_indices(patch_indices, clone_inds):
   newpatchinds = []
   for patch in patch_indices:
      patch = list(patch)
      patch = [np.where(clone_inds==i)[0][0] for i in patch]
      newpatchinds.append(patch)
   return newpatchinds

def write_cnt_text_file(cnt_list, file_name):
    with open(file_name, 'w') as file:
        for cnt_i in cnt_list:
            file.write(','.join(['%f' % num for num in cnt_i[:,0,0]])+"\n")
            file.write(','.join(['%f' % num for num in cnt_i[:,0,1]])+"\n")
            
def write_score_text_file(clone_scores, file_name):
    with open(file_name, 'w') as file:
        for score in clone_scores:
            file.write("%f\n" % score)
            
def write_cnt_hdf5(cnt_list, cnt_file_name):
   with h5py.File(cnt_file_name, 'w', libver='latest') as f:  # use 'latest' for performance
      for idx, arr in enumerate(cnt_list):
         dset = f.create_dataset(str(idx), shape=arr.shape, data=arr, chunks=arr.shape,
                                 compression='gzip', compression_opts=9)

def write_clone_image_snips(folder_to_analyse, file_name, clone_contours, scaling_val):
   imgout = folder_to_analyse + "/clone_images/"
   mkdir_p(imgout)
   #smallcnts = rescale_contours(clone_contours, 1./scaling_val) #for level=1
   i = 0
   for cc in clone_contours: # smallcnts here if level=1
      expand_box    = 25*scaling_val #remove scaling val here if level=1
      roi           = cv2.boundingRect(cc)
      roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box), dtype=np.uint64)
      roi[roi<1]   = 0
      img_ROI       = getROI_img_osl(file_name, (roi[0],roi[1]), (roi[2],roi[3]), level=0) # or level=1?
      outfile = "/clone_" + str(i) + ".png"
      cv2.imwrite(imgout + outfile, img_ROI)
      i += 1

## Example hdf5 saving from SPRING -- use as template to create contour saving function?
#def save_hdf5_genes(E, gene_list, filename):
#    '''SPRING standard: filename = main_spring_dir + "counts_norm_sparse_genes.hdf5"'''
#    
#    import h5py
#    
#    E = E.tocsc()
#    
#    hf = h5py.File(filename, 'w')
#    counts_group = hf.create_group('counts')
#    cix_group = hf.create_group('cell_ix')

#    hf.attrs['ncells'] = E.shape[0]
#    hf.attrs['ngenes'] = E.shape[1]

#    for iG, g in enumerate(gene_list):
#        counts = E[:,iG].A.squeeze()
#        cell_ix = np.nonzero(counts)[0]
#        counts = counts[cell_ix]
#        counts_group.create_dataset(g, data = counts)
#        cix_group.create_dataset(g, data = cell_ix)

#    hf.close()
#    
#def save_hdf5_cells(E, filename):
#    '''SPRING standard: filename = main_spring_dir + "counts_norm_sparse_cells.hdf5" '''
#    import h5py
#    
#    E = E.tocsr()
#    
#    hf = h5py.File(filename, 'w')
#    counts_group = hf.create_group('counts')
#    gix_group = hf.create_group('gene_ix')

#    hf.attrs['ncells'] = E.shape[0]
#    hf.attrs['ngenes'] = E.shape[1]

#    for iC in range(E.shape[0]):
#        counts = E[iC,:].A.squeeze()
#        gene_ix = np.nonzero(counts)[0]
#        counts = counts[gene_ix]
#        counts_group.create_dataset(str(iC), data = counts)
#        gix_group.create_dataset(str(iC), data = gene_ix)

#    hf.close()

            
def read_cnt_text_file(file_name):
    with open(file_name, 'r') as file:
        contours = file.readlines()
    numcnts = len(contours)//2
    cnts_out = []
    for i in range(numcnts):
        i *= 2 # x
        j = i+1 # y        
        lx = contours[i][:-1].split(',')
        lx = [int(float(x)) for x in lx]
        ly = contours[j][:-1].split(',')
        ly = [int(float(x)) for x in ly]
        numpnts = len(lx)
        a = np.zeros([numpnts,1,2], dtype=np.int32)
        for k in range(numpnts):
            a[k,0,0] = lx[k]
            a[k,0,1] = ly[k]
        cnts_out.append(a)
    return cnts_out         

def simplify_contours(cnt_list):
    new_cnts = []
    for i in cnt_list:
        approx = cv2.approxPolyDP(i, 0.01*cv2.arcLength(i,True), True)
        if (len(approx)<10):
            new_cnts.append(i)
        else:
            new_cnts.append(approx)
    return new_cnts

def plot_histogram(x, bins=50, norm_it = False):
    hist, bins = np.histogram(x, bins=bins, normed = norm_it)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()  

def thresh_img(img_deconv2, channel, thesh):
    _, img_threshed = cv2.threshold(img_deconv2[:,:,channel], thesh, 255, cv2.THRESH_BINARY)        
    if img_threshed.dtype != np.uint8: img_threshed = img_threshed.astype('uint8', copy=False) 
    return(img_threshed)

def binaryTo3C(img):
    return cv2.merge((img, img ,img))

## For memory management use just in place ops
def transform_OD(img):
    OD_data = img+np.float32(1.0)
    OD_data /= 256. ## In place
    OD_data = cv2.log(OD_data, OD_data) ## In place
    OD_data *= -1. ## In place
    return(OD_data)
    
def col_deconvol(img, deconv_mat):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False)
    img_deconv = cv2.transform(OD_data, deconv_mat)
    ## Convert to 8 bits
    img_deconv = np.clip(img_deconv, 0, 1, out=img_deconv)
    img_deconv *= 255
    img_deconv = img_deconv.astype('uint8', copy=False) 
    return(img_deconv)

def col_deconvol_blur_clone(img, deconv_mat, size_blur):
    # deconvolution
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)
    ## blur
    nucl_blur = cv2.GaussianBlur(img_deconv[:,:,0], size_blur1, 0)
    clone_blur      = cv2.GaussianBlur(img_deconv[:,:,1], size_blur, 0)
    ## convert to 8 bits
    clone_blur = np.clip(clone_blur, 0, 1, out=clone_blur)
    clone_blur *= 255
    clone_blur = clone_blur.astype('uint8', copy=False) 
    return clone_blur

## If you convert to 8bit before blurring you lose resolution
def col_deconvol_and_blur2(img, deconv_mat, size_blur1, size_blur2):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)    
    nucl_blur  = cv2.GaussianBlur(img_deconv[:,:,0], size_blur1, 0)
    clone_blur = cv2.GaussianBlur(img_deconv[:,:,1], size_blur2, 0)
    return nucl_blur, clone_blur

def col_deconvol_and_blur3(img, deconv_mat, size_blur1, size_blur2):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)    
    nucl_blur  = cv2.GaussianBlur(img_deconv[:,:,0], size_blur1, 0)
    clone_blur = cv2.GaussianBlur(img_deconv[:,:,1], size_blur2, 0)
    background_blur = cv2.GaussianBlur(img_deconv[:,:,2], size_blur1, 0)
    return nucl_blur, clone_blur, background_blur

## If you convert to 8bit before blurring you lose resolution
def col_deconvol_and_blur(img, deconv_mat, size_blur1, size_blur2, size_blur3):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)    
    nucl_blur_small = cv2.GaussianBlur(img_deconv[:,:,0], size_blur1, 0)
    nucl_blur_large = cv2.GaussianBlur(img_deconv[:,:,0], size_blur2, 0)
    clone_blur      = cv2.GaussianBlur(img_deconv[:,:,1], size_blur3, 0)
    
    ## Convert to 8 bits
    nucl_blur_small = np.clip(nucl_blur_small, 0, 1, out=nucl_blur_small)
    nucl_blur_small *= 255
    nucl_blur_small = nucl_blur_small.astype('uint8', copy=False) 

    ## Convert to 8 bits
    nucl_blur_large = np.clip(nucl_blur_large, 0, 1, out=nucl_blur_large)
    nucl_blur_large *= 255
    nucl_blur_large = nucl_blur_large.astype('uint8', copy=False) 

    ## Convert to 8 bits
    clone_blur = np.clip(clone_blur, 0, 1, out=clone_blur)
    clone_blur *= 255
    clone_blur = clone_blur.astype('uint8', copy=False) 
    
    return nucl_blur_small, nucl_blur_large, clone_blur

def col_deconvol_32(img, deconv_mat):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)
    return(img_deconv)

def plot_img(list_to_plot, nrow = 1, nameWindow = 'Plots', NewWindow = True, hold_plot = True):
    num_images = len(list_to_plot)
    num_cols   = int(num_images/nrow)
    if num_images%nrow != 0:
        raise(UserWarning, "If more than one row make sure there are enough images!")
    if NewWindow:
        screen_res = 1600, 1000
        cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nameWindow, screen_res[0], screen_res[1])
    if isinstance(list_to_plot, tuple) == 0: 
        vis = list_to_plot
    else:
        last_val = num_cols 
        vis      = np.concatenate(list_to_plot[0:last_val], axis=1)
        for row_i in range(1, nrow):
            first_val = last_val
            last_val  = first_val + num_cols
#            print (first_val,last_val)
            vis_aux   = np.concatenate(list_to_plot[first_val:last_val], axis=1)
            vis       = np.concatenate((vis, vis_aux), axis=0)        
    cv2.imshow(nameWindow, vis)
    if(hold_plot):
        0xFF & cv2.waitKey(0)
        cv2.destroyWindow(nameWindow)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def make8bit(img_in):
    # Saturate values
    img_deconv_255 = 255*np.clip(img_in, 0, 1)
    # Covert to 8 bit
    img_deconv_255 = img_deconv_255.astype('uint8', copy=False) 
    return(img_deconv_255)

## Correct width and height for cropping so that it never overshoots the 
## size of the image
def correct_wh(max_vals, xy_vals, wh_vals):
    final_x = xy_vals[0] + wh_vals[0]
    final_y = xy_vals[1] + wh_vals[1]
    new_wh_x = wh_vals[0]
    new_wh_y = wh_vals[1]
    if final_x > max_vals[0] : new_wh_x = max_vals[0] - xy_vals[0]
    if final_y > max_vals[1] : new_wh_y = max_vals[1] - xy_vals[1]
    return new_wh_x, new_wh_y

#def getROI_img_vips(file_name, x_y, w_h, level = 0):
#    vim           = pyvips.Image.openslideload(file_name, level = level)   #openslideload
#    max_vals      = (vim.width, vim.height)
#    max_vals      = vim.dimensions
#    wh_vals_final = correct_wh(max_vals, x_y, w_h) ## Correct rounding errors 
#    area          = vim.extract_area(x_y[0], x_y[1], wh_vals_final[0], wh_vals_final[1])
#    size          = (area.width, area.height)
#    data          = area.write_to_memory()
#    new_img       = np.fromstring(data, dtype=np.uint8).reshape(size[1], size[0], 4)  ## Remove alpha channel  
#    new_img       = cv2.cvtColor(new_img[:,:,0:3], cv2.COLOR_RGB2BGR)
#    return new_img

def getROI_img_osl(file_name, x_y, w_h, level = 0):
    vim           = osl.OpenSlide(file_name)
    max_vals      = vim.dimensions
    wh_vals_final = correct_wh(max_vals, x_y, w_h) ## Correct rounding errors 
    new_img = np.array(vim.read_region(location = x_y, level = level, size = w_h))
    new_img       = cv2.cvtColor(new_img[:,:,0:3], cv2.COLOR_RGB2BGR)
    return new_img
    
def getIndexesTileImage(max_vals, scalingVal, ROI_crop, max_num_pix  = 10000): # 22000
    ## Index stuff to tile image
    start_indx      = (int(ROI_crop[0][0]*scalingVal), int(ROI_crop[0][1]*scalingVal))
    full_delta_s    = (int(ROI_crop[1][0]*scalingVal) - start_indx[0], int(ROI_crop[1][1]*scalingVal) - start_indx[1])
    last_x          = start_indx[0] + full_delta_s[0]
    last_y          = start_indx[1] + full_delta_s[1]
    ## Make sure you don't overshoot the image
    if last_x>max_vals[0]: last_x = max_vals[0] # x -> cols
    if last_y>max_vals[1]: last_y = max_vals[1] # y -> rows
    last_indx       = (last_x, last_y)
    num_tiles       = [int(np.ceil(float(full_delta_s[0])/max_num_pix)), int(np.ceil(float(full_delta_s[1])/max_num_pix))]
    if num_tiles[0] == 0: num_tiles[0] = 1
    if num_tiles[1] == 0: num_tiles[1] = 1
        
    delta_x         = np.ceil(full_delta_s[0]/num_tiles[0])
    delta_y         = np.ceil(full_delta_s[1]/num_tiles[1])
        
    overlap = 175
    if (delta_x<overlap):
        overlap = delta_x//2
    if (delta_y<overlap):
        overlap = delta_y//2
    all_indx = []
    for i in range(num_tiles[0]):
        if (i==0):
            x0      = start_indx[0]
            width_i = delta_x 
        else:
            x0 = start_indx[0] + i*delta_x - overlap
            width_i = delta_x + overlap
        if (i == (num_tiles[0]-1)): width_i = last_indx[0] - x0
        all_indx.append([])
        for j in range(num_tiles[1]):
            if (j==0):    
                y0       = start_indx[1] + j*delta_y
                height_j = delta_y
            else:
                y0       = start_indx[1] + j*delta_y - overlap
                height_j = delta_y + overlap
            if j == (num_tiles[1]-1): height_j = last_indx[1] - y0
            all_indx[i].append((x0, y0, width_i, height_j))
    return all_indx

def plotImageAndFit(indx_True, indx_on, crypt_cnt_raw, img, indx_subset = None):
    if indx_subset is None: 
        indx_subset = np.ones(len(crypt_cnt_raw),dtype= np.bool)
    if indx_True is None: 
        indx_True = np.ones(len(crypt_cnt_raw),dtype= np.bool)
        
    ## Try to plot results on image
    crypt_cnt_subset = [cnt_i for is_crypt, cnt_i in zip(indx_subset, crypt_cnt_raw) if is_crypt]
    crypt_cnt_mine   = [cnt_i for is_crypt, cnt_i in zip(indx_on, crypt_cnt_subset) if is_crypt]
    crypt_cnt_EM     = [cnt_i for is_crypt, cnt_i in zip(indx_True, crypt_cnt_subset) if is_crypt]
    img_plot         = img.copy()
    #cv2.drawContours(img_plot,  crypts_erode_cnt, -1, (255,0,0), 7)    
    cv2.drawContours(img_plot,   crypt_cnt_EM, -1, (  0,  0, 255), 12) 
    cv2.drawContours(img_plot, crypt_cnt_mine, -1, (255,  0,   0),  6) 
    plot_img(img_plot, hold_plot=True)

