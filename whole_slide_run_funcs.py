## whole slide analysis functions
import cv2
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from MiscFunctions import contour_xy, add_offset, rescale_contours, write_cnt_text_file, contour_EccMajorMinorAxis, contour_Area, simplify_contours, bbox_y1_x1_y2_x2
from tensorflow.keras import mixed_precision
from unet_sep_multiout import unet_sep
from model_set_parameter_dicts import set_params
from slide_gen import slide_tile_gen

params = set_params()
st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#dnnfolder = "/home/doran/Work/py_code/new_DeCryptICS/newfiles/"
dnnfolder = os.path.dirname(os.path.abspath(__file__))
mixed_precision.set_global_policy('mixed_float16')

model, just_trnsf, just_unet = unet_sep(params, is_comp = False)
weights_name = os.path.join(dnnfolder, 'weights', 'decrypt_weights.hdf5')
model.load_weights(weights_name)

def run_slide(file_name, output_folder, crypt_thresh=0.5, seg_max_len=20, seg_batch_size=16, min_micrn_overlap=80, bbox_batch_size=25, max_bboxes=400, repeat_method='largest', save_contours=True):
   a_full = time.time()

   wh_gen = slide_tile_gen(file_name, mpp=params['umpp'], tile_size=params['tilesize_train'], max_len=seg_max_len, batch_size=seg_batch_size, min_micrn_overlap = min_micrn_overlap)
   
#   wh_gen.get_img_thmb_tilerect()

   ## segment crypts
   a_u = time.time()
   mask_cr, fmap_all = segment_tiles(wh_gen)
   b_u = time.time()
   print('Segmentation done in %1.1f seconds' % (b_u - a_u))

   ## join kept tiles to create full mask and feature map
   fullmask, fullfmap, tilesize_masks, tilesize_fmaps, indx_at22 = process_mask_and_fmaps(wh_gen, mask_cr, fmap_all, crypt_thresh=crypt_thresh)
   del(mask_cr)
   del(fmap_all)

   ## extract contours
   cnts, cnt_xy = find_contours_from_full_mask(fullmask)
   
   ## if finding extra fufis, do so:   
   cnts, cnt_xy, orig_cnt_inds, new_cnt_inds = add_potential_fufi_cnts(cnts, cnt_xy, fullmask, params['umpp'])
   del(fullmask)

   ## predict clone, partial and fufi probabilities for bounding boxes
   a_t = time.time()
   raw_df = predict_bbox_probs(cnts, cnt_xy, fullfmap, indx_at22, tilesize_masks, tilesize_fmaps, batchsize=bbox_batch_size, max_crypts=params['num_bbox'])
   b_t = time.time()
   print('Bounding box probability predictions done in %1.1f seconds' % (b_t - a_t))
   del(fullfmap)

   ## take averages for repeated crypts? or take prediction for largest bbox?
   out_df = raw_df.reset_index().drop(['index'], axis=1)
   if repeat_method == 'average':
      if params['crypt_class'] is True:
         out_df = out_df.groupby('crypt_num').agg(
                                mask_x=('mask_x', 'mean'),
                                mask_y=('mask_y', 'mean'),
                                p_fufi=('p_fufi', 'mean'),
                                p_clone=('p_clone', 'mean'),
                                p_partial=('p_partial', 'mean'),
                                p_crypt=('p_crypt', 'mean'),
                                bbox_x1=('bbox_area', lambda x: out_df['bbox_x1'].iloc[x.idxmax()]),
                                bbox_y1=('bbox_area', lambda x: out_df['bbox_y1'].iloc[x.idxmax()]),
                                bbox_x2=('bbox_area', lambda x: out_df['bbox_x2'].iloc[x.idxmax()]),
                                bbox_y2=('bbox_area', lambda x: out_df['bbox_y2'].iloc[x.idxmax()]),
                                bbox_area=('bbox_area', 'max')
                                )
      else:
         out_df = out_df.groupby('crypt_num').agg(
                                mask_x=('mask_x', 'mean'),
                                mask_y=('mask_y', 'mean'),
                                p_fufi=('p_fufi', 'mean'),
                                p_clone=('p_clone', 'mean'),
                                p_partial=('p_partial', 'mean'),
                                bbox_x1=('bbox_area', lambda x: out_df['bbox_x1'].iloc[x.idxmax()]),
                                bbox_y1=('bbox_area', lambda x: out_df['bbox_y1'].iloc[x.idxmax()]),
                                bbox_x2=('bbox_area', lambda x: out_df['bbox_x2'].iloc[x.idxmax()]),
                                bbox_y2=('bbox_area', lambda x: out_df['bbox_y2'].iloc[x.idxmax()]),
                                bbox_area=('bbox_area', 'max')
                                )
   else:
      out_df = out_df.loc[out_df.groupby('crypt_num')['bbox_area'].idxmax()]
   
   ## replace crypts by expanded fufi contours if better prediction, reset crypt numbering in df
   out_df, cnts = decide_on_fufi_contours(out_df, cnts, new_cnt_inds, orig_cnt_inds)
       
   ## rescale contours and contour xy to full svs resolution
   # depends on params['stride_bool'] == True
   mask_scale = (wh_gen.mpp*2)/wh_gen.svs.mpp0
   bbox_scale = 2 * mask_scale   
   out_df.loc[:,'mask_x']  = out_df['mask_x']  * mask_scale
   out_df.loc[:,'mask_y']  = out_df['mask_y']  * mask_scale
   out_df.loc[:,'bbox_x1'] = out_df['bbox_x1'] * bbox_scale
   out_df.loc[:,'bbox_y1'] = out_df['bbox_y1'] * bbox_scale
   out_df.loc[:,'bbox_x2'] = out_df['bbox_x2'] * bbox_scale
   out_df.loc[:,'bbox_y2'] = out_df['bbox_y2'] * bbox_scale
   out_df = out_df.rename(columns = {'mask_x':'x', 'mask_y':'y'})
   out_df = out_df.reset_index().drop(labels = ['index', 'bbox_area'], axis=1)
   
   ## calculate contour features
   cnts = rescale_contours(cnts.copy(), mask_scale)
   out_df = get_contour_features(cnts, out_df)

   ## save output
   out_df.to_csv(output_folder + '/raw_crypt_output.csv', index=False)
   if save_contours:
      cnts = simplify_contours(cnts)
      write_cnt_text_file(cnts, output_folder + "/crypt_contours.txt")

   b_full = time.time()
   print('Full slide analysis for %s done in %1.1f seconds' % (file_name, b_full - a_full))

def get_contour_features(cnts, out_df):
   out_df['area'] = np.asarray([contour_Area(i) for i in cnts])
   eccmajorminor = [contour_EccMajorMinorAxis(i) for i in cnts]
   out_df['ecc'] = [eccmajorminor[i][0] for i in range(len(eccmajorminor))]
   out_df['majorax'] = [eccmajorminor[i][1] for i in range(len(eccmajorminor))]
   out_df['minorax'] = [eccmajorminor[i][2] for i in range(len(eccmajorminor))]
   return out_df

@tf.function
def serve(x):
   return just_trnsf(x, training=False)

def run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, fmap_xy_batch, tilesize_fmaps, out_df):   
   bbox_batch = np.array(bbox_batch)
   fmap_batch = np.array(fmap_batch)
#   p_i = just_trnsf.predict([fmap_batch, bbox_batch])
   p_i = serve([fmap_batch, bbox_batch])
   for ba in range(len(xy_batch)):
      nc = xy_batch[ba].shape[0]
      bboxsize = (bbox_batch[ba][:nc,3]-bbox_batch[ba][:nc,1])*(bbox_batch[ba][:nc,2]-bbox_batch[ba][:nc,0])   
      if params['crypt_class'] is True:
         out_df = out_df.append(pd.DataFrame({'crypt_num':indices_batch[ba],
                                              'mask_x'   :xy_batch[ba][:,0],
                                              'mask_y'   :xy_batch[ba][:,1],
                                              'p_clone'  :p_i[0][ba,:nc,0],
                                              'p_partial':p_i[1][ba,:nc,0],
                                              'p_fufi'   :p_i[2][ba,:nc,0],
                                              'p_crypt'  :p_i[3][ba,:nc,0],
             'bbox_x1'  :(fmap_xy_batch[ba][0] + bbox_batch[ba][:nc,1]*tilesize_fmaps),
             'bbox_y1'  :(fmap_xy_batch[ba][1] + bbox_batch[ba][:nc,0]*tilesize_fmaps),
             'bbox_x2'  :(fmap_xy_batch[ba][0] + bbox_batch[ba][:nc,3]*tilesize_fmaps),
             'bbox_y2'  :(fmap_xy_batch[ba][1] + bbox_batch[ba][:nc,2]*tilesize_fmaps),
             'bbox_area':bboxsize*tilesize_fmaps*tilesize_fmaps}))
      else:
         out_df = out_df.append(pd.DataFrame({'crypt_num':indices_batch[ba],
                                              'mask_x'   :xy_batch[ba][:,0],
                                              'mask_y'   :xy_batch[ba][:,1],
                                              'p_clone'  :p_i[0][ba,:nc,0],
                                              'p_partial':p_i[1][ba,:nc,0],
                                              'p_fufi'   :p_i[2][ba,:nc,0],
             'bbox_x1'  :(fmap_xy_batch[ba][0] + bbox_batch[ba][:nc,1]*tilesize_fmaps),
             'bbox_y1'  :(fmap_xy_batch[ba][1] + bbox_batch[ba][:nc,0]*tilesize_fmaps),
             'bbox_x2'  :(fmap_xy_batch[ba][0] + bbox_batch[ba][:nc,3]*tilesize_fmaps),
             'bbox_y2'  :(fmap_xy_batch[ba][1] + bbox_batch[ba][:nc,2]*tilesize_fmaps),
             'bbox_area':bboxsize*tilesize_fmaps*tilesize_fmaps}))
   # set new batch                                                    
   fmap_batch = []
   bbox_batch = []
   indices_batch = []
   xy_batch = []
   fmap_xy_batch = []
   return fmap_batch, bbox_batch, indices_batch, xy_batch, fmap_xy_batch, out_df

def find_contours_from_full_mask(fullmask):
   cnts, _ = cv2.findContours(fullmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
   cnt_len = [len(cnt) for cnt in cnts]
   keepcnts = np.where(np.asarray(cnt_len)>3)[0]
   cnts = list(np.asarray(cnts, dtype=object)[keepcnts])
   cnt_xy = np.asarray([list(contour_xy(cnt_i.astype(np.int32))) for cnt_i in cnts])
   return cnts, cnt_xy

def predict_bbox_probs(cnts, cnt_xy, fullfmap, indx_at22, tilesize_masks, tilesize_fmaps, batchsize=20, max_crypts=400):
   out_df = pd.DataFrame()   
   fmap_batch = []
   bbox_batch = []
   indices_batch = []
   xy_batch = []
   fmap_xy_batch = []
   print("Tiling up for bounding box class predictions...")
   for tile_i in range(indx_at22.shape[0]):
      if tile_i % 100 == 0: print('tile %d of %d' % (tile_i, indx_at22.shape[0]))
      # for a tile get mask xy and fmap ij
      tile_xy = indx_at22[tile_i,::-1] # from ij to xy
      fmap_ij = indx_at22[tile_i,:]//2
      # get the contours present and the correct feature map, create mask dummy
      these_cnts, these_inds, these_xy = get_cnts_in_tile(cnts, cnt_xy, tile_xy, tilesize_masks)
      this_fmap = fullfmap[fmap_ij[0]:(fmap_ij[0]+tilesize_fmaps), fmap_ij[1]:(fmap_ij[1]+tilesize_fmaps), :]
      # create bounding boxes in correct order and fix overhanging countours
      bboxes, these_cnts, these_inds, these_xy = get_bbox_from_contours(these_cnts, these_inds, these_xy, tilesize_masks, max_crypts = max_crypts)
      ncrypts = these_inds.shape[0]
      # if ncrypts > 400: sep into several 400 chunks and predict with same fmap
      if ncrypts>max_crypts:
         numin = ncrypts // max_crypts
         remdr = ncrypts % max_crypts
         for j in range(numin):
            index_range = np.array(range(j*max_crypts, (j+1)*max_crypts), dtype=np.int32)
            subinds = these_inds[index_range]
            subxy = these_xy[index_range,:]
            bbox_i = bboxes[index_range,:]
            fmap_batch.append(this_fmap)
            bbox_batch.append(bbox_i)
            indices_batch.append(subinds)
            xy_batch.append(subxy)
            fmap_xy_batch.append((fmap_ij[1],fmap_ij[0]))
            if len(bbox_batch)==batchsize:
               fmap_batch, bbox_batch, indices_batch, xy_batch, fmap_xy_batch, out_df = run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, fmap_xy_batch, tilesize_fmaps, out_df)
         # remainder of tile bboxes
         subinds = these_inds[-max_crypts:]
         subxy = these_xy[-max_crypts:,:]
         bbox_i = bboxes[-max_crypts:,:]
         fmap_batch.append(this_fmap)
         bbox_batch.append(bbox_i)
         indices_batch.append(subinds)
         xy_batch.append(subxy)
         fmap_xy_batch.append((fmap_ij[1],fmap_ij[0]))
         if len(bbox_batch)==batchsize:
            fmap_batch, bbox_batch, indices_batch, xy_batch, fmap_xy_batch, out_df = run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, fmap_xy_batch, tilesize_fmaps, out_df)
      else:
         fmap_batch.append(this_fmap)
         bbox_batch.append(bboxes)
         indices_batch.append(these_inds)
         xy_batch.append(these_xy)
         fmap_xy_batch.append((fmap_ij[1],fmap_ij[0]))
         if len(bbox_batch)==batchsize:
            fmap_batch, bbox_batch, indices_batch, xy_batch, fmap_xy_batch, out_df = run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, fmap_xy_batch, tilesize_fmaps, out_df)
   # check for leftover batch < batchsize
   if len(bbox_batch)>0:
      fmap_batch, bbox_batch, indices_batch, xy_batch, fmap_xy_batch, out_df = run_batch(bbox_batch, fmap_batch, indices_batch, xy_batch, fmap_xy_batch, tilesize_fmaps, out_df)
   return out_df

def segment_tiles(wh_gen):   
   done = 0
   size = just_unet.layers[-1].output_shape[2]
   mask_cr_all = np.empty((wh_gen.num_tiles, size, size, 1), dtype=np.float32)
   fmap_all = np.empty((wh_gen.num_tiles, size//2, size//2, 32), dtype=np.float32)
   num_iters = wh_gen.total_batches / len(wh_gen)
   print("Running %1.1f bars worth of segmentation..." % num_iters)
   num_iters = int(np.ceil(num_iters))
   for jj in range(num_iters):
       wh_gen.set_start_indx(done)
       masks_run, fmaps_run = just_unet.predict(x = wh_gen, verbose = 1, workers = 3)
       done += masks_run.shape[0]
       mask_cr_all[wh_gen.indx_zero:done,:,:,:] = masks_run
       fmap_all[wh_gen.indx_zero:done,:,:,:] = fmaps_run
       del(masks_run)
       del(fmaps_run)
   return mask_cr_all, fmap_all
   
def process_mask_and_fmaps(wh_gen, mask_cr, fmap_all, crypt_thresh=0.5):
   # Process mask and make bboxes 
   mask_cr = mask_cr > crypt_thresh
   cr_px_p_tile  = np.sum(np.sum(mask_cr, axis = 1), axis = 1)

   # Discard tiles 
   keep_tiles = np.where(cr_px_p_tile > 10)[0] # num crypt pixels detected

   tile_pos_thmb = wh_gen.tile_array # top left corner of tile in thmb image
   tile_pos_filt = tile_pos_thmb[keep_tiles]
   mask_cr       = mask_cr[keep_tiles]
   fmap_all      = fmap_all[keep_tiles]

   tilesize_masks = mask_cr.shape[1]
   indx_at22 = np.rint(tile_pos_filt / (2*wh_gen.mpp)).astype(np.int32)
   dims_at22 = np.rint(wh_gen.full_img_size_um[::-1] / (2*wh_gen.mpp)).astype(np.int32)
     
#   fullmask = np.ones([dims_at22[0], dims_at22[1],3], dtype = np.uint8)
#   for ii, indx_i in enumerate(indx_at22):
#       for cci in range(3):        
#           fullmask[indx_i[0]:(indx_i[0] + tilesize_masks),
#                    indx_i[1]:(indx_i[1] + tilesize_masks),
#                    cci] += (np.random.randint(255)*mask_cr[ii, :,:,0]).astype(np.uint8)
#   plot_img(fullmask)

   fullmask = np.zeros((dims_at22[0], dims_at22[1]), dtype=bool)
   for ii, indx_i in enumerate(indx_at22):
       fullmask[indx_i[0]:(indx_i[0] + tilesize_masks),
                indx_i[1]:(indx_i[1] + tilesize_masks)] += mask_cr[ii,:,:,0]
   fullmask = fullmask.astype(np.uint8) * 255
#   plot_img(fullmask)

   # features
   tilesize_fmaps = fmap_all.shape[1]
   fullfmap = np.zeros((dims_at22[0]//2, dims_at22[1]//2, fmap_all.shape[3]), dtype=np.float32)
   stack_map = np.zeros((dims_at22[0]//2, dims_at22[1]//2), dtype=np.int32)
   trim = 8
   for ii, indx_i in enumerate(indx_at22):
       fullfmap[(indx_i[0]//2 + trim):(indx_i[0]//2 + tilesize_fmaps - trim), 
                (indx_i[1]//2 + trim):(indx_i[1]//2 + tilesize_fmaps - trim), :] += fmap_all[ii, trim:(-trim), trim:(-trim), :]
       stack_map[(indx_i[0]//2 + trim):(indx_i[0]//2 + tilesize_fmaps - trim), 
                 (indx_i[1]//2 + trim):(indx_i[1]//2 + tilesize_fmaps - trim)] += 1

   for nn in range(2,5):
      inds = np.where(stack_map==nn)
      fullfmap[inds] = fullfmap[inds] / nn
      
   return fullmask, fullfmap, tilesize_masks, tilesize_fmaps, indx_at22

def get_bbox_from_contours(these_cnts, these_inds, these_xy, tilesize_masks, max_crypts = 400):
   bboxes = np.array([bbox_y1_x1_y2_x2(cnti) for cnti in these_cnts])
   # check for boxes off edge and remove (so we don't use predictions for that crypt)
   bboxes[bboxes<0] = 0
   bboxes[bboxes>(tilesize_masks-1)] = (tilesize_masks-1)   
   sums_x = np.sum(bboxes[:,np.array([1,3])], axis=1)
   sums_y = np.sum(bboxes[:,np.array([0,2])], axis=1)
   bad_inds = np.hstack([np.where(sums_x==0)[0], np.where(sums_y==0)[0], np.where(sums_x==2*((tilesize_masks-1)))[0], np.where(sums_y==2*((tilesize_masks-1)))[0]])
   keep_inds = np.setdiff1d(range(these_inds.shape[0]), bad_inds)
   these_inds = these_inds[keep_inds]
   these_cnts = list(np.asarray(these_cnts, dtype=object)[keep_inds])
   these_xy = these_xy[keep_inds,:]
   bboxes = bboxes[keep_inds, :]
   # make coords [0 , 1]   
   bboxes = bboxes/tilesize_masks
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
   return bboxes, these_cnts, these_inds, these_xy

def get_cnts_in_tile(cnts, cnt_xy, tile_xy, tilesize):
   # Find indexes of crypts and retrun crypts and info
   these_inds = np.where(np.logical_and(
     np.logical_and(cnt_xy[:,0]>(tile_xy[0] - 0.1*tilesize),
                    cnt_xy[:,0]<(tile_xy[0] + 1.1*tilesize)),
     np.logical_and(cnt_xy[:,1]>(tile_xy[1] - 0.1*tilesize),
                    cnt_xy[:,1]<(tile_xy[1] + 1.1*tilesize))))[0]          
   cnts_out = [cnts[ii] for ii in these_inds]
   cnts_out = add_offset(cnts_out.copy(), -tile_xy)   
   return cnts_out, these_inds, cnt_xy[these_inds,:]

def join_for_extra_contours(thismask, cnts, dummy_mask, n_dil):   
   thismask = cv2.morphologyEx(thismask.copy(), cv2.MORPH_DILATE, st_3, iterations = n_dil)
   cnts_dil, cnts_dil_xy = find_contours_from_mask(thismask)
   bimg = dummy_mask.copy()
   for n, cd in enumerate(cnts_dil):
      cv2.drawContours(bimg, [cd], 0, (n+1,  0,   0),  -1)
   # find the contour numbers underneath orig contours
   meancols = [bimg[cnts[i][0,0,1], cnts[i][0,0,0]] for i in range(len(cnts))]
   
   # do several contours overlay same number (i.e. have been joined)?
   uniq, nums = np.unique(meancols, return_counts=True)
   new_cnt_inds = uniq[np.where(nums>1)[0]] - 1 # undo the 1-starting from drawing cnts
   orig_cnt_inds = [np.where(meancols==(ni+1))[0].astype(np.int32) for ni in new_cnt_inds]
   extra_cnts = [cnts_dil[ni] for ni in new_cnt_inds]
   extra_cnts_xy = cnts_dil_xy[new_cnt_inds,:]
   return extra_cnts, extra_cnts_xy, orig_cnt_inds

def find_contours_from_mask(mask):
   cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
   cnt_len = [len(cnt) for cnt in cnts]
   keepcnts = np.where(np.asarray(cnt_len)>3)[0]
   cnts = list(np.asarray(cnts, dtype=object)[keepcnts])
   cnt_xy = np.asarray([list(contour_xy(cnt_i.astype(np.int32))) for cnt_i in cnts])
   return cnts, cnt_xy
   
def add_potential_fufi_cnts(cnts, cnt_xy, fullmask, mpp):   
   n_dil = int(np.around(1/mpp))
   extra_cnts, extra_cnts_xy, orig_cnt_inds = join_for_extra_contours(fullmask, cnts, np.zeros(fullmask.shape[:2], dtype=np.uint16), n_dil)
   n_new_cnts = len(extra_cnts)
   new_cnt_inds = np.array(range(len(cnts), len(cnts)+len(extra_cnts)), dtype=np.int32)
   if len(extra_cnts)>0:
      cnts = cnts + extra_cnts
   cnt_xy = np.vstack([cnt_xy, extra_cnts_xy])
   return cnts, cnt_xy, orig_cnt_inds, new_cnt_inds
    
def decide_on_fufi_contours(out_df, cnts, new_cnt_inds, orig_cnt_inds):
   drop_inds = []
   for jj, xb in enumerate(new_cnt_inds):
      new_dat = out_df[out_df['crypt_num']==xb]
      orig_dat = out_df[out_df['crypt_num'].isin(orig_cnt_inds[jj])]
      new_fufi_pred = float(new_dat['p_fufi'])
      old_fufi_pred = np.mean(orig_dat['p_fufi'])
      # also check crypt probability?
      if params['crypt_class'] is True:
         new_crypt_pred = float(new_dat['p_crypt'])
         old_crypt_pred = np.mean(orig_dat['p_crypt'])
         if new_fufi_pred>old_fufi_pred and new_fufi_pred>0.25:
            [drop_inds.append(orig_cnt_inds[jj][kk]) for kk in range(orig_cnt_inds[jj].shape[0])]
            # if we are replacing with fufi, bring over the old crypt prob if higher
            if new_crypt_pred<old_crypt_pred:
               out_df.at[new_dat.index[0], 'p_crypt'] = old_crypt_pred
         elif new_crypt_pred>old_crypt_pred and new_crypt_pred>0.5:
            [drop_inds.append(orig_cnt_inds[jj][kk]) for kk in range(orig_cnt_inds[jj].shape[0])]
         else:
            drop_inds.append(xb)
      else:
         if new_fufi_pred>old_fufi_pred and new_fufi_pred>0.25:
            [drop_inds.append(orig_cnt_inds[jj][kk]) for kk in range(orig_cnt_inds[jj].shape[0])]
         else:
            drop_inds.append(xb)
   keep_inds = np.setdiff1d(range(len(cnts)),drop_inds)
   out_df = out_df[out_df['crypt_num'].isin(keep_inds)]
   out_df.loc[:,'crypt_num'] = out_df['crypt_num'].astype(np.int32)
   cnts = [cnts[i] for i in out_df['crypt_num']]
   # reset numbering
   out_df = out_df.drop(['crypt_num'], axis=1).reset_index().drop(['index'], axis=1).reset_index().rename(columns={'index':'crypt_num'})
   out_df.loc[:,'crypt_num'] = out_df['crypt_num'].astype(np.int32)
   return out_df, cnts
   
