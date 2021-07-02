#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:05:04 2021

@author: edward
"""
import numpy as np
import pandas as pd
#from tensorflow.keras import mixed_precision
#from unet_sep_multiout import unet_sep
from model_set_parameter_dicts import set_params_ed
from slide_gen import slide_tile_gen
#import tensorflow as tf
#import cv2
#from MiscFunctions import plot_img
import time

params_gen_ed = set_params_ed()

#dnnfolder = "/home/doran/Work/py_code/new_DeCryptICS/newfiles/"
#mixed_precision.set_global_policy('mixed_float16')

#model, just_trnsf, just_unet = unet_sep(params_gen_ed, weight_ccpf = weight_ccpf, is_comp = False)
#weights_name = dnnfolder + "/weights/pixshuf_randtile_genbb_11mpp.hdf5"
#model.load_weights(weights_name)

a_full = time.time()

file_test = '/home/doran/Work/images/Leeds_May2019/curated_cryptdata/train/KM3/KM3M_428042.svs'
#file_test = '/home/doran/Work/images/Leeds_May2019/curated_cryptdata/train/KM21/KM21M_443097.svs'
wh_gen = slide_tile_gen(file_test, mpp=params_gen_ed['umpp'], tile_size=params_gen_ed['tilesize_train'], max_len=50, batch_size=16, min_micrn_overlap = 80)

from whole_slide_run_funcs import segment_tiles, process_mask_and_fmaps, find_contours_from_full_mask, predict_bbox_probs

## segment crypts
a_u = time.time()
mask_cr_all, fmap_all = segment_tiles(wh_gen)
b_u = time.time()
print('Segmentation done in %1.1f seconds' % (b_u - a_u))

## join kept tiles to create full mask and feature map
fullmask, fullfmap, tilesize_masks, tilesize_fmaps, indx_at22 = process_mask_and_fmaps(wh_gen, mask_cr_all, fmap_all, crypt_thresh=0.75)
del(mask_cr_all)
del(fmap_all)

## extract contours
cnts, cnt_xy = find_contours_from_full_mask(fullmask)
del(fullmask)

## predict clone, partial and fufi probabilities for bounding boxes
a_t = time.time()
out_df = predict_bbox_probs(cnts, cnt_xy, fullfmap, indx_at22, tilesize_masks, tilesize_fmaps, batchsize=25, max_crypts=400)
b_t = time.time()
print('Bounding box probability predictions done in %1.1f seconds' % (b_t - a_t))
del(fullfmap)

out_df.to_csv('/home/doran/Work/py_code/new_DeCryptICS/newfiles/test_output_batched2.csv', index=False)

b_full = time.time()
print('Full slide analysis done in %1.1f seconds' % (b_full - a_full))


# Plot grid
#wh_gen.get_img_thmb_tilerect()

#num_iters = int(np.ceil(wh_gen.total_batches / len(wh_gen)))
#done = 0
#mask_cr_all = np.empty((wh_gen.num_tiles,512,512,1), dtype=np.float32)
#fmap_all = np.empty((wh_gen.num_tiles,256,256,32), dtype=np.float32)
#for jj in range(num_iters):
#    wh_gen.set_start_indx(done)
#    masks_run, fmaps_run = just_unet.predict(x = wh_gen, verbose = 1, workers = 3)
#    done += masks_run.shape[0]
#    mask_cr_all[wh_gen.indx_zero:done,:,:,:] = masks_run
#    fmap_all[wh_gen.indx_zero:done,:,:,:] = fmaps_run
#    del(masks_run)
#    del(fmaps_run)




## Process mask and make bboxes 
#mask_cr       = mask_cr_all>0.5

#cr_px_p_tile  = np.sum(np.sum(mask_cr, axis = 1), axis = 1)

## Discard tiles 
#keep_tiles = np.where(cr_px_p_tile > 10)[0] # num crypt pixels detected

#tile_pos_thmb = wh_gen.tile_array # top left corner of tile in thmb image
#tile_pos_filt = tile_pos_thmb[keep_tiles]
#mask_cr       = mask_cr[keep_tiles]
#fmap_sub      = fmap_all[keep_tiles]

#tilesize_masks = 512
#indx_at22 = (tile_pos_filt * tilesize_masks / wh_gen.thmb_dx).astype(np.int32)
#dims_at22 = (np.array(wh_gen.thmb_dims) * tilesize_masks / wh_gen.thmb_dx).astype(np.int32)
#fullmask = np.ones([dims_at22[0], dims_at22[1],3], dtype = np.uint8)
#for ii, indx_i in enumerate(indx_at22):
#    for cci in range(3):        
#        fullmask[indx_i[0]:(indx_i[0] + tilesize_masks),
#                 indx_i[1]:(indx_i[1] + tilesize_masks),
#                 cci] += (np.random.randint(255)*mask_cr[ii, :,:,0]).astype(np.uint8)
#plot_img(fullmask)

#fullmask = np.zeros((dims_at22[0], dims_at22[1]), dtype=bool)
#for ii, indx_i in enumerate(indx_at22):
#    fullmask[indx_i[0]:(indx_i[0] + tilesize_masks),
#             indx_i[1]:(indx_i[1] + tilesize_masks)] += mask_cr[ii,:,:,0]
#fullmask = fullmask.astype(np.uint8) * 255
#plot_img(fullmask)

## features
#tilesize_fmaps = 256
#fullfmap = np.zeros((dims_at22[0]//2, dims_at22[1]//2, fmap_sub.shape[3]), dtype=np.float32)
#stack_map = np.zeros((dims_at22[0]//2, dims_at22[1]//2), dtype=np.int32)
#trim = 8
#for ii, indx_i in enumerate(indx_at22):
#    fullfmap[(indx_i[0]//2 + trim):(indx_i[0]//2 + tilesize_fmaps - trim), 
#             (indx_i[1]//2 + trim):(indx_i[1]//2 + tilesize_fmaps - trim), :] += fmap_sub[ii, trim:(-trim), trim:(-trim), :]
#    stack_map[(indx_i[0]//2 + trim):(indx_i[0]//2 + tilesize_fmaps - trim), 
#              (indx_i[1]//2 + trim):(indx_i[1]//2 + tilesize_fmaps - trim)] += 1

#for nn in range(2,5):
#   inds = np.where(stack_map==nn)
#   fullfmap[inds] = fullfmap[inds] / nn

#plot_img(((fullfmap[:,:,10] / np.quantile(fullfmap[:,:,10],0.95))*255).astype(np.uint8))

#for chan in range(fullfmap.shape[2]):
#   plot_img(((fullfmap[:,:,chan] / np.quantile(fullfmap[:,:,chan],0.95))*255).astype(np.uint8))

# tiles 512 mpp 2.2
# wh_gen.thmb_dx is 512 in other res
# index transf
# thmbnail width 1024
# 2.2mpp



# Output contour list and df with classification
# Contour list is in scale 1.1 mpp 1024 -> 512 so 2.2mpp?
# 

''' - use this as master contour list
    - cycle through tiles / fmap tiles
    - either subset full mask (and somehow subset to bounding boxes contained on tile) or draw 
      contours from master set like in training, and keep track of their position in master list
'''

#from MiscFunctions import contour_xy, add_offset
#max_crypts = 400
##bbox_i, cnt_i = get_bbox_from_masks(fullmask, max_crypts = max_crypts)
#cnts, _ = cv2.findContours(fullmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#del(fullmask)
#cnt_len = [len(cnt) for cnt in cnts]
#keepcnts = np.where(np.asarray(cnt_len)>3)[0]
#cnts = list(np.asarray(cnts, dtype=object)[keepcnts])
#cnt_xy = np.asarray([list(contour_xy(cnt_i)) for cnt_i in cnts])


## batch accumulate
#@tf.function
#def serve(x):
#   return just_trnsf(x, training=False)

#def run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, out_df):   
#   bbox_batch = np.array(bbox_batch)
#   fmap_batch = np.array(fmap_batch)
##   p_i = just_trnsf.predict([fmap_batch, bbox_batch])
#   p_i = serve([fmap_batch, bbox_batch])
#   for ba in range(len(xy_batch)):
#      nc = xy_batch[ba].shape[0]
#      out_df = out_df.append(pd.DataFrame({'crypt_num':indices_batch[ba],
#                                           'mask_x'   :xy_batch[ba][:,0],
#                                           'mask_y'   :xy_batch[ba][:,1],
#                                           'p_clone'  :p_i[0][ba,:nc,0],
#                                           'p_partial':p_i[1][ba,:nc,0],
#                                           'p_fufi'   :p_i[2][ba,:nc,0]}))
#   # set new batch                                                    
#   fmap_batch = []
#   bbox_batch = []
#   indices_batch = []
#   xy_batch = []
#   return fmap_batch, bbox_batch, indices_batch, xy_batch, out_df

#a1 = time.time()
#out_df = pd.DataFrame()
#batchsize = 20
#fmap_batch = []
#bbox_batch = []
#indices_batch = []
#xy_batch = []
#for tile_i in range(indx_at22.shape[0]):
#   print('tile %d of %d' % (tile_i, indx_at22.shape[0]))
#   # for a tile get mask xy and fmap ij
#   tile_xy = indx_at22[tile_i,::-1] # from ij to xy
#   fmap_ij = indx_at22[tile_i,:]//2
#   # get the contours present and the correct feature map, create mask dummy
#   these_cnts, these_inds, these_xy = get_cnts_in_tile(cnts, cnt_xy, tile_xy, tilesize_masks)
#   this_fmap = fullfmap[fmap_ij[0]:(fmap_ij[0]+tilesize_fmaps), fmap_ij[1]:(fmap_ij[1]+tilesize_fmaps), :]
#   # create bounding boxes in correct order and fix overhanging countours
#   bboxes, these_cnts, these_inds, these_xy = get_bbox_from_contours(these_cnts, these_inds, these_xy, tilesize_masks, max_crypts = max_crypts)
#   ncrypts = these_inds.shape[0]
#   # if ncrypts > 400: sep into several 400 chunks and predict with same fmap
#   if ncrypts>max_crypts:
#      numin = ncrypts // max_crypts
#      remdr = ncrypts % max_crypts
#      for j in range(numin):
#         index_range = np.array(range(j*max_crypts, (j+1)*max_crypts), dtype=np.int32)
#         subinds = these_inds[index_range]
#         subxy = these_xy[index_range,:]
#         bbox_i = bboxes[index_range,:]
#         fmap_batch.append(this_fmap)
#         bbox_batch.append(bbox_i)
#         indices_batch.append(subinds)
#         xy_batch.append(subxy)
#         if len(bbox_batch)==batchsize:
#            fmap_batch, bbox_batch, indices_batch, xy_batch, out_df = run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, out_df)
#      # remainder of tile bboxes
#      subinds = these_inds[-max_crypts:]
#      subxy = these_xy[-max_crypts:,:]
#      bbox_i = bboxes[-max_crypts:,:]
#      fmap_batch.append(this_fmap)
#      bbox_batch.append(bbox_i)
#      indices_batch.append(subinds)
#      xy_batch.append(subxy)
#      if len(bbox_batch)==batchsize:
#         fmap_batch, bbox_batch, indices_batch, xy_batch, out_df = run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, out_df)
#   else:
#      fmap_batch.append(this_fmap)
#      bbox_batch.append(bboxes)
#      indices_batch.append(these_inds)
#      xy_batch.append(these_xy)
#      if len(bbox_batch)==batchsize:
#         fmap_batch, bbox_batch, indices_batch, xy_batch, out_df = run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, out_df)
## check for leftover batch < batchsize
#if len(bbox_batch)>0:
#   fmap_batch, bbox_batch, indices_batch, xy_batch, out_df = run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, out_df)

#b1 = time.time()
#batched_time = b1-a1

#out_df.to_csv('/home/doran/Work/py_code/new_DeCryptICS/newfiles/test_output_batched.csv', index=False)
