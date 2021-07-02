#!/usr/bin/env python3
import glob
import numpy as np
import pandas as pd
from augmentation import plot_img
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from gen_ed      import DataGen_curt, CloneGen_curt
from tensorflow.keras import mixed_precision
from unet_sep_multiout import unet_sep
from MiscFunctions import contour_xy, add_offset, plot_img
from sklearn.neighbors import NearestNeighbors
import cv2

from model_set_parameter_dicts import set_params_ed
params_gen_ed = set_params_ed()

from LRcycle import CyclicLR

dnnfolder = '/home/doran/Work/py_code/new_DeCryptICS/newfiles'
logs_folder = dnnfolder + "/logs"

mixed_precision.set_global_policy('mixed_float16')

# Run paramaters
params_gen_ed['umpp']       = 1.1
params_gen_ed['num_bbox']   = 400 # 400
params_gen_ed['batch_size'] = 8 # 16, was 7
params_gen_ed["just_clone"] = False
params_gen_ed['cpfr_frac']  = [4,5,3,10] #[4,5,3,3]
nsteps        = 2
nclone_factor = 10 #4
npartial_mult = 5 #10
use_CLR = False
weight_ccpf = [50., 50., 75., 50.] #[25., 50., 75., 35.]

# Read curated data and filter bad ones
already_curated = pd.read_csv("manual_curation_files/curated_files_summary.txt", 
                              names = ["file_name", "slide_crtd"])
already_curated = already_curated[already_curated['slide_crtd'] != "cancel"]

# remove poorly stained slides or radiotherapy patients
bad_slides = pd.read_csv('/home/doran/Work/py_code/new_DeCryptICS/slidequality_scoring.csv')
radiother = np.where(bad_slides['quality_label']==2)[0]
staining = np.where(bad_slides['quality_label']==0)[0]
dontuse = np.asarray(bad_slides['path'])[np.hstack([radiother, staining])]
dontuse = pd.DataFrame({'file_name':list(dontuse)}).drop_duplicates(keep='first')
inds = ~already_curated.file_name.isin(dontuse.file_name)
already_curated = already_curated[inds]

train_data      = already_curated.sample(150, random_state=22)

train_datagen = DataGen_curt(params_gen_ed, train_data, nsteps, nclone_factor, npartial_mult)

model, just_trnsf, just_unet = unet_sep(params_gen_ed, weight_ccpf = weight_ccpf, use_CLR = use_CLR)

print("Loading weights!!")
weights_name = dnnfolder + "/weights/pixshuf_randtile_400bb_11mpp.hdf5"
model.load_weights(weights_name)


def bbox_y1_x1_y2_x2(cnti):
   bb_cv2 = cv2.boundingRect(cnti)
   # x,y,w,h -> y1, x1, y2, x2
   return np.array([bb_cv2[1], bb_cv2[0], bb_cv2[1] + bb_cv2[3], bb_cv2[0]+ bb_cv2[2]])
   
def get_bboxes(these_cnts, tilesize_masks):
   bboxes = np.array([bbox_y1_x1_y2_x2(cnti) for cnti in these_cnts])
   bboxes = bboxes/tilesize_masks
   return bboxes

cr_thresh = 0.5
null_bb  = np.array([ 0.,  0., -1., -1.])
ut_fm_bb_pcpf = [] # untouched original batches
re_fm_bb_pcpf = [] # repaired batches with extra bboxes
## run through some epochs generating batches
num_epochs = 1
for ee in range(num_epochs):
   for nb in range(15,len(train_datagen)):
      (im, b_bboxes), (b_ma, b_p_c, b_p_p, b_p_f) = train_datagen[nb]
      ## predict with the current unet to find new masks/bboxes
      masks, fmaps = just_unet.predict(x = im, verbose = 1, workers = 3)
      for ba in range(masks.shape[0]):
         num_to_add = 0
         cnts, _ = cv2.findContours((masks[ba,:,:,0]>cr_thresh).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
         cnts_gt, _ = cv2.findContours((b_ma[ba,:,:]>cr_thresh).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
         if len(cnts)>0:
            if len(cnts_gt)>0:
               cnt_xy = np.asarray([list(contour_xy(cnt_i)) for cnt_i in cnts])
               cnt_xy_gt = np.asarray([list(contour_xy(cnt_i)) for cnt_i in cnts_gt])
               if (cnt_xy_gt.shape[0]>1 and cnt_xy.shape[0]>1) : 
                  nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(cnt_xy_gt)
                  distances, indices = nbrs.kneighbors(cnt_xy)
                  icd, selfinds = nbrs.kneighbors(cnt_xy_gt)
                  new_inds = np.where(distances[:,0]>np.min(icd[:,1]))[0]
               else:
                  nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cnt_xy_gt)
                  distances, indices = nbrs.kneighbors(cnt_xy)
                  new_inds = np.where(distances[:,0]>30)[0]
            elif len(cnts_gt)==0 and len(cnts)>0:
               new_inds = np.array(range(len(cnts_gt)))
            if len(new_inds)>0: 
               new_boxes = get_bboxes(cnts, 512)
               new_boxes = new_boxes[new_inds]
               # find where b_bboxes[ba] is null box and replace with new boxes, make b_p_c[ba] zero
               free_inds = np.where(np.sum(np.equal(b_bboxes[ba], null_bb), axis=1)==4)[0]
               num_to_add = np.minimum(new_boxes.shape[0], free_inds.shape[0])
               for ii in range(num_to_add):
                  b_bboxes[ba][free_inds[ii],:] = new_boxes[ii,:]
                  b_p_c[ba][free_inds[ii]] = 0
                  b_p_p[ba][free_inds[ii]] = 0
                  b_p_f[ba][free_inds[ii]] = 0                                 
         # add this tile / mask / bbox / probs to a larger data structure to add to batches for tuning transformer
         if num_to_add>0:
            re_fm_bb_pcpf.append((fmaps[ba], b_bboxes[ba], b_p_c[ba], b_p_p[ba], b_p_f[ba]))
         else:
            ut_fm_bb_pcpf.append((fmaps[ba], b_bboxes[ba], b_p_c[ba], b_p_p[ba], b_p_f[ba]))
      print("length of orig batches: %d" % len(ut_fm_bb_pcpf))
      print("length of fixed batches: %d" % len(re_fm_bb_pcpf))
      
      
np.random.seed(seed=42)
t_inds_ut = np.random.choice(range(len(ut_fm_bb_pcpf)), size=150, replace=False)
t_inds_re = np.random.choice(range(len(re_fm_bb_pcpf)), size=150, replace=False)
v_inds_ut = np.random.choice(np.setdiff1d(range(len(ut_fm_bb_pcpf)), t_inds_ut), size=100, replace=False)
v_inds_re = np.random.choice(np.setdiff1d(range(len(re_fm_bb_pcpf)), t_inds_re), size=100, replace=False)
train_ut = np.asarray(ut_fm_bb_pcpf, dtype=object)[t_inds_ut]
train_re = np.asarray(re_fm_bb_pcpf, dtype=object)[t_inds_re]
valid_ut = np.asarray(ut_fm_bb_pcpf, dtype=object)[v_inds_ut]
valid_re = np.asarray(re_fm_bb_pcpf, dtype=object)[v_inds_re]
      
from tensorflow.keras.utils import Sequence
import math

class trnsf_ins(Sequence):

   def __init__(self, data_ut, data_re, n_ut, n_re):
      self.x_ut = data_ut
      self.x_re = data_re
      self.batch_size = n_ut + n_re
      self.n_ut = n_ut
      self.n_re = n_re

   def __len__(self):
      return math.ceil(len(self.x_ut) / self.n_ut)

   def __getitem__(self, idx):
      x_b_fm = []
      x_b_bb = []
      y_b_c = []
      y_b_p = []
      y_b_f = []
      orig_inds = np.random.choice(range(len(self.x_ut)), size = self.n_ut, replace = False)
      fix_inds = np.random.choice(range(len(self.x_re)), size = self.n_re, replace = False)
      for i in range(num_orig):
         x_b_fm.append(self.x_ut[orig_inds[i]][0])
         x_b_bb.append(self.x_ut[orig_inds[i]][1])
         y_b_c.append(self.x_ut[orig_inds[i]][2])
         y_b_p.append(self.x_ut[orig_inds[i]][3])
         y_b_f.append(self.x_ut[orig_inds[i]][4])
      for i in range(num_fixed):
         x_b_fm.append(self.x_re[fix_inds[i]][0])
         x_b_bb.append(self.x_re[fix_inds[i]][1])
         y_b_c.append(self.x_re[fix_inds[i]][2])
         y_b_p.append(self.x_re[fix_inds[i]][3])
         y_b_f.append(self.x_re[fix_inds[i]][4])
      x_b_fm = np.asarray(x_b_fm)
      x_b_bb = np.asarray(x_b_bb)
      y_b_c = np.asarray(y_b_c)
      y_b_p = np.asarray(y_b_p)
      y_b_f = np.asarray(y_b_f)
      return [x_b_fm, x_b_bb], [y_b_c, y_b_p, y_b_f]
      
train_dat = trnsf_ins(train_ut, train_re, 7, 7)
valid_dat = trnsf_ins(valid_ut, valid_re, 7, 7)
                      
## fine-tune transformer
from tensorflow.keras              import metrics
from tensorflow.keras.optimizers   import RMSprop, SGD

losses = {'clone': "binary_crossentropy",
        'partial': "binary_crossentropy",
        'fufi': "binary_crossentropy"
}

lossWeights = {'clone': weight_ccpf[1],
             'partial': weight_ccpf[2],
             'fufi': weight_ccpf[3]
}

metrics_use = {'clone': [metrics.TruePositives(), 
                     metrics.FalseNegatives(), 
                     metrics.FalsePositives(), 
                     metrics.TrueNegatives()],
             'partial': [metrics.TruePositives(), 
                       metrics.FalseNegatives(), 
                       metrics.FalsePositives(), 
                       metrics.TrueNegatives()],
             'fufi': [metrics.TruePositives(), 
                       metrics.FalseNegatives(), 
                       metrics.FalsePositives(), 
                       metrics.TrueNegatives()]
}

just_trnsf.compile(optimizer=RMSprop(lr=0.0001), loss = losses,
                   loss_weights = lossWeights, metrics = metrics_use)   
                   
weights_name_next = dnnfolder + "/weights/trnsf_finetune_test.hdf5"

callbacks = [EarlyStopping(monitor='loss', patience=25, verbose=1, min_delta=1e-9),
             ReduceLROnPlateau(monitor='loss', factor=0.075, patience=8, verbose=1, min_delta=1e-9),
             ModelCheckpoint(monitor='val_loss', mode='min', filepath=weights_name_next, 
                             save_best_only=True, save_weights_only=True, verbose=1),
             CSVLogger(logs_folder + '/hist_'+weights_name_next.split('/')[-1].split('.')[0]+'.csv')]
         
res = just_trnsf.fit(
         x = train_dat,
         validation_data = valid_dat,
         verbose = 1,
         epochs = 1,
         callbacks = callbacks,
         workers = 2,
         )





