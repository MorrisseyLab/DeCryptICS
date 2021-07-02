#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue May 11 14:43:54 2021

@author: edward
'''
import glob
import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from MiscFunctions import contour_xy, add_offset, rescale_contours, write_cnt_text_file, contour_EccMajorMinorAxis, contour_Area, simplify_contours
from training.augmentation import plot_img
from training.gen_v2 import DataGen_curt, bbox_y1_x1_y2_x2
from tensorflow.keras import mixed_precision
from unet_sep_multiout import unet_sep
from model_set_parameter_dicts import set_params
from sklearn.neighbors import NearestNeighbors
params = set_params()

def find_contours_from_mask(fullmask):
   cnts, _ = cv2.findContours(fullmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
   cnt_len = [len(cnt) for cnt in cnts]
   keepcnts = np.where(np.asarray(cnt_len)>3)[0]
   cnts = list(np.asarray(cnts, dtype=object)[keepcnts])
   cnt_xy = np.asarray([list(contour_xy(cnt_i.astype(np.int32))) for cnt_i in cnts])
   return cnts, cnt_xy

def get_bbox_from_contours(cnts, tilesize_masks):
   bboxes = np.array([bbox_y1_x1_y2_x2(cnti.astype(np.int32)) for cnti in cnts])
   bboxes[bboxes<0] = 0
   bboxes[bboxes>(tilesize_masks-1)] = (tilesize_masks-1)
   # make coords [0 , 1]
   bboxes = bboxes/tilesize_masks
   return bboxes

def pad_bboxes(bboxes, max_crypts = 400):
   null_bbox = np.array([[0, 0, -1, -1]])   
   n_crypts = bboxes.shape[0]
   if n_crypts == 0:
      bboxes   = null_bbox
      n_crypts = 1
   if n_crypts < max_crypts:        
      # pad with null bounding boxes
      indx_subsmp = np.zeros(max_crypts-n_crypts, dtype = np.uint8)
      bboxes0     = null_bbox[indx_subsmp,:]
      # stack them 
      bboxes      = np.vstack([bboxes, bboxes0])    
   return bboxes

def bbox_centroids(bbox_batch):
   bbox_x = (bbox_batch[:,3] - bbox_batch[:,1]) / 2. + bbox_batch[:,1]
   bbox_y = (bbox_batch[:,2] - bbox_batch[:,0]) / 2. + bbox_batch[:,0]
   bbox_xy = np.zeros((bbox_x.shape[0], 2))
   bbox_xy[:,0] = bbox_x
   bbox_xy[:,1] = bbox_y
   return bbox_xy

dnnfolder = '/home/doran/Work/py_code/new_DeCryptICS/newfiles'
logs_folder = dnnfolder + '/logs'

mixed_precision.set_global_policy('mixed_float16')

# Run paramaters
params['umpp']       = 1.1
params['num_bbox']   = 400 # 400
params['batch_size'] = 8 # 16, was 7
params['just_clone'] = False
params['cpfr_frac']  = [1,1,1,1]
params['aug'] = False
nsteps        = 5
nclone_factor = 7 #4
npartial_mult = 5 #10
use_CLR = False
weight_ccpf = [1., 1.5, 2., 1.25]

# Read curated data and filter bad ones
already_curated = pd.read_csv('./training/manual_curation_files/curated_files_summary.txt', 
                              names = ['file_name', 'slide_crtd'])
already_curated = already_curated[already_curated['slide_crtd'] != 'cancel']

# remove poorly stained slides or radiotherapy patients
bad_slides = pd.read_csv('./training/manual_curation_files/slidequality_scoring.csv')
radiother = np.where(bad_slides['quality_label']==2)[0]
staining = np.where(bad_slides['quality_label']==0)[0]
dontuse = np.asarray(bad_slides['path'])[np.hstack([radiother, staining])]
dontuse = pd.DataFrame({'file_name':list(dontuse)}).drop_duplicates(keep='first')
inds = ~already_curated.file_name.isin(dontuse.file_name)
already_curated = already_curated[inds]

train_data      = already_curated.sample(already_curated.shape[0]-100, random_state=22)
keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
val_data        = already_curated[keep_indx]

val_datagen = DataGen_curt(params, val_data, nsteps, nclone_factor, npartial_mult)

model, just_trnsf, just_unet = unet_sep(params, weight_ccpf = weight_ccpf)

print('Loading weights!!')
weights_name = dnnfolder + '/weights/pixshuf_genbb_balanceweights_11mpp_jitterboxes.hdf5'
model.load_weights(weights_name)


@tf.function
def serve(x):
   return just_trnsf(x, training=False)
   
run_batches = 10
crypt_thresh = 0.8
tilesize_masks = 512
max_crypts = 400
out_df = pd.DataFrame()
imgs = []
allbbs_gt = []
allbbs_pred = []
#for i in range(run_batches):
   i = 0
   #[x_batch, bboxes], [mask_crypt, p_i[:,:,0], p_i[:,:,1], p_i[:,:,2]]
   im_bb, ma_ps = val_datagen[i]
#   allbbs_gt.append(im_bb[1])
   masks_run, fmaps_run = just_unet.predict(x = im_bb[0], verbose = 1, workers = 3)
   masks_thresh = (masks_run > crypt_thresh).astype(np.uint8) * 255
   bbox_batch = []
   ncrypts_batch = []
   for ba in range(masks_thresh.shape[0]):
      cnts, cnt_xy = find_contours_from_mask(masks_thresh[ba])
      bboxes = get_bbox_from_contours(cnts, cnt_xy, tilesize_masks, max_crypts)
      ncrypts_batch.append(len(cnts))
      bbox_batch.append(bboxes)
#   allbbs_pred.append(bbox_batch)
   p_i = serve([fmaps_run, np.array(bbox_batch)])
   p_i_gt = serve([fmaps_run, im_bb[1]])
   imbatch = []
   for ba in range(masks_thresh.shape[0]):
      nc = ncrypts_batch[ba]
      if nc>0:
         bbox_xy = bbox_centroids(bbox_batch[ba][:nc,:])
         bbox_gt_xy = bbox_centroids(im_bb[1][ba])
         keep_gt_inds = np.where(bbox_gt_xy[:,0]!= -0.5)[0]
         pcs = p_i_gt[0][ba].numpy()[keep_gt_inds,0]
         pps = p_i_gt[1][ba].numpy()[keep_gt_inds,0]
         pfs = p_i_gt[2][ba].numpy()[keep_gt_inds,0]
         
         thisxy = (im_bb[1][ba][keep_gt_inds,:] * 1024).astype(np.int32)
         thisxy_p = (bbox_batch[ba][:nc,:] * 1024).astype(np.int32)
         im = (im_bb[0][ba] * val_datagen.norm_std) + val_datagen.norm_mean
         col1 = (0,255,0)
         col2 = (0,10,255)
         for ii in range(keep_gt_inds.shape[0]):
            cv2.rectangle(im,(thisxy[ii,:][1],thisxy[ii,:][0]),(thisxy[ii,:][3],thisxy[ii,:][2]),col1,2)
            if pfs[ii] > 0.25:
               cv2.putText(im, str(np.around(pfs[ii],2)),(thisxy[ii,:][3]+10,thisxy[ii,:][2]),0,0.3, col1)   
         for ii in range(nc):
            cv2.rectangle(im,(thisxy_p[ii,:][1],thisxy_p[ii,:][0]),(thisxy_p[ii,:][3],thisxy_p[ii,:][2]),col2,2)
            if p_i[2][ba,ii,0] > 0.25:
               cv2.putText(im, str(np.around(p_i[2][ba,ii,0].numpy(),2)),(thisxy_p[ii,:][3]+10,thisxy_p[ii,:][2]),0,0.3, col2)
         imbatch.append(im)
         nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(bbox_gt_xy[keep_gt_inds,:])
         distances, indices = nbrs.kneighbors(bbox_xy)
         out_df = out_df.append(pd.DataFrame({'batch': np.ones(nc)*i,
                                              'elem': np.ones(nc)*ba,
                                              'bbox_ind':list(range(nc)),
                                              'p_clone'  :p_i[0][ba,:nc,0],
                                              'p_partial':p_i[1][ba,:nc,0],
                                              'p_fufi'   :p_i[2][ba,:nc,0],
                                              'nn_ind_gt':indices[:,0],
                                              'nn_dist_gt':distances[:,0],
                                              'p_clone_gt'  :pcs[indices[:,0]],
                                              'p_partial_gt':pps[indices[:,0]],
                                              'p_fufi_gt'   :pfs[indices[:,0]]}))
   imgs.append(imbatch)
   
   
out_df.to_csv('./test/pred_bbs_vs_gt_bbs.csv', index=False)
      

inds = np.argsort(np.asarray(abs(out_df['p_fufi'] - out_df['p_fufi_gt'])))[::-1]
out_df['batch'].iloc[inds[:20]]
out_df['elem'].iloc[inds[:20]]
   
ba = 7
ii = 0
plot_img(imgs[ba][ii])




###############################################
## post processing tests for fufi finding

def find_possible_fufi_cnts(contours, nbrs):
   status = np.zeros(nbrs.shape)
   for i, cnt1 in enumerate(contours):
      for j in range(nbrs.shape[1]):
         cnt2 = contours[nbrs[i,j]]
         dist = find_if_close(cnt1, cnt2, thresh=7)
         if dist == True: status[i,j] = 1
   joininds = np.where(status>0)
   jointo = nbrs[joininds]
   unified = []
   for i in range(jointo.shape[0]):
      cont = np.vstack([contours[joininds[0][i]], contours[jointo[i]]])
      hull = cv2.convexHull(cont)
      unified.append(hull)
   return unified

def find_if_close(cnt1, cnt2, thresh=7):
    row1, row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < thresh:
                return True
            elif i==row1-1 and j==row2-1:
                return False

def join_for_extra_boxes(thismask, cnts, dummy_mask, n_dil):
   thismask = cv2.morphologyEx(thismask.copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)
   cnts_dil, _ = find_contours_from_mask(thismask)
   bimg = dummy_mask.copy()
   for n, cd in enumerate(cnts_dil):
      cv2.drawContours(bimg, [cd], 0, (n+1,  0,   0),  -1)
   meancols = np.array([contour_number(cnt, bimg) for cnt in cnts])
   uniq, nums = np.unique(meancols, return_counts=True)
   new_cnt_inds = uniq[np.where(nums>1)[0]] - 1 # undo the 1-starting from drawing cnts
   orig_cnt_inds = [np.where(meancols==(ni+1))[0].astype(np.int32) for ni in new_cnt_inds]
   extra_cnts = [cnts_dil[ni] for ni in new_cnt_inds]
   extra_bboxes = get_bbox_from_contours(extra_cnts, tilesize_masks)
   return extra_bboxes, extra_cnts, orig_cnt_inds

def contour_number(cnt, img):
    # Get mean colour of object
    dummy = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(dummy, [cnt], 0, (255,  0,   0),  -1)
    inds = cv2.findNonZero(dummy)    
    img[inds]
#    plot_img(dummy//2 + bimg.astype(np.uint8))
    return img[inds[:,0,1], inds[:,0,0]][0]
    
i = 0
#[x_batch, bboxes], [mask_crypt, p_i[:,:,0], p_i[:,:,1], p_i[:,:,2]]
im_bb, ma_ps = val_datagen[i]
masks_run, fmaps_run = just_unet.predict(x = im_bb[0], verbose = 1, workers = 3)
masks_thresh = (masks_run > crypt_thresh).astype(np.uint8) * 255
bbox_batch = []
ncrypts_batch = []
cnts_batch = []
extra_box_inds_batch = []
extra_box_orig_inds_batch = []
st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
n_dil = int(np.around(3./params['umpp']))
dummy_mask = np.zeros(masks_run[0].shape[:-1], dtype=np.uint16) # larger scale than uint8
for ba in range(masks_thresh.shape[0]):
   cnts, cnt_xy = find_contours_from_mask(masks_thresh[ba])   
   bboxes = get_bbox_from_contours(cnts, tilesize_masks)
   
   # option 1
   extra_bboxes, extra_cnts, orig_cnt_inds = join_for_extra_boxes(masks_thresh[ba,:,:,0], cnts, dummy_mask, n_dil)
   n_new_boxes = extra_bboxes.shape[0]
   new_box_inds = np.array(range(len(cnts), len(cnts)+n_new_boxes), dtype=np.int32)
   if n_new_boxes>0:
      bboxes = np.vstack([bboxes,extra_bboxes])
      cnts = cnts + extra_cnts
   bboxes = pad_bboxes(bboxes, max_crypts)
   # add to batch results
   ncrypts_batch.append(len(cnts))
   bbox_batch.append(bboxes)
   cnts_batch.append(cnts)
   extra_box_inds_batch.append(new_box_inds)
   extra_box_orig_inds_batch.append(orig_cnt_inds)
     
#   # option 2
#   nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(cnt_xy)
#   distances, indices = nbrs.kneighbors(cnt_xy)
#   indices = indices[:,1:].astype(np.int32)
#   new_cnts = find_possible_fufi_cnts(cnts, indices)
#   new_cnt_xy = np.asarray([list(contour_xy(cnt_i.astype(np.int32))) for cnt_i in new_cnts])
#   new_bboxes = get_bbox_from_contours(new_cnts, tilesize_masks)
#   bboxes = pad_bboxes(stack(orig, new), max_crypts)
   

p_i = just_trnsf([fmaps_run, np.array(bbox_batch)])
p_i = [p_i[b].numpy() for b in range(len(p_i))]
# compare extra boxes with those they overlay; keep better fufi prediction
for ba in range(masks_thresh.shape[0]):
   pf_n = p_i[2].numpy()
   pc_n = p_i[0].numpy()
   drop_inds = []
   for jj, xb in enumerate(extra_box_inds_batch[ba]):
      new_fufi_pred = pf_n[ba,xb,0]
      old_fufi_preds = np.mean(pf_n[ba,extra_box_orig_inds_batch[ba][jj],0])
      
      timg = cv2.drawContours(cv2.pyrDown(((im_bb[0][ba] * val_datagen.norm_std) + val_datagen.norm_mean).copy()), [cnts_batch[ba][xb]], -1, (0,255,0), 2)
      for kk in range(extra_box_orig_inds_batch[ba][jj].shape[0]):
         cv2.drawContours(timg, [cnts_batch[ba][extra_box_orig_inds_batch[ba][jj][kk]]], -1, (255,255,0), 2)
      plot_img(timg)
      
      if new_fufi_pred>old_fufi_preds:
         [drop_inds.append(extra_box_orig_inds_batch[ba][jj][kk]) for kk in range(extra_box_orig_inds_batch[ba][jj].shape[0])]
      else: drop_inds.append(xb)
   keep_inds = [i for i in range(len(cnts_batch[ba])) if i not in drop_inds]
   cnts_batch[ba] = [cnts_batch[ba][i] for i in keep_inds]
   ''' p_i subset this somehow?! need to incorporate this in the whole slide run function'''
      


































