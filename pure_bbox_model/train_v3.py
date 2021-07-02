#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue May 11 14:43:54 2021

@author: edward
'''
import glob
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras import mixed_precision
from training.augmentation import plot_img
from training.gen_v3 import DataGen_curt, CloneGen_curt, CloneFufiGen_curt

from pred_bbox_network import pred_bbox_short#, pred_bbox_dense
from model_set_parameter_dicts import set_params
from training.losses import get_total_loss, batched_loss_closure, DummyMetric1, DummyMetric2, calc_loss_alone
from tensorflow.keras.optimizers import RMSprop, SGD
from training.train_fn import create_loss_metric_trackers, update_trackers, log_trackers, reset_trackers, aggregate_losses, setup_optimizer, gather_gradient, apply_gradient

physical_devices = tf.config.list_physical_devices('GPU')
mixed_precision.set_global_policy('mixed_float16')

dnnfolder = '/home/doran/Work/py_code/new_DeCryptICS/newfiles/'
logs_folder = dnnfolder + '/training/logs/'

# Run paramaters
params = set_params()
epochs               = 200
params['umpp']       = 1.1
params['num_bbox']   = 400
params['batch_size'] = 8
params['just_clone'] = False
params['cpfr_frac']  = [1,1,1,1]
loss_weights         = [1, 2, 5] # label_cost, giou_loss, l1_loss
class_weights        = [1., 1.5, 2., 1.25, 1.] # crypt, clone, partial, fufi, no object
nsteps        = 2 #5
nclone_factor = 2 #7
npartial_mult = 2 #5

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

train_data      = already_curated.sample(15, random_state=22)
keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
val_data        = already_curated[keep_indx].sample(5, random_state=223)

#train_data      = already_curated.sample(150, random_state=22)
#keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
#val_data        = already_curated[keep_indx].sample(100, random_state=223)

#train_data      = already_curated.sample(already_curated.shape[0]-100, random_state=22)
#keep_indx       = np.bitwise_not(already_curated.index.isin(train_data.index))
#val_data        = already_curated[keep_indx]

train_datagen = DataGen_curt(params, train_data, nsteps, nclone_factor, npartial_mult)
val_datagen   = CloneGen_curt(params, val_data)

model = pred_bbox_short(num_bbox = params['num_bbox'])
optimizer = setup_optimizer(model, lr = 1e-3)
##optimizer = RMSprop(learning_rate=1e-4)

#print('Loading weights!!')
#weights_name = dnnfolder + '/weights/bbox_pred_net.hdf5'
#model.load_weights(weights_name)
weights_name_next = dnnfolder + '/weights/bbox_pred_net.hdf5'
logpath = logs_folder + '/hist_'+weights_name_next.split('/')[-1].split('.')[0]


@tf.function
def train_step2(model, images, t_bbox, t_class):
   t_bbox = tf.convert_to_tensor(t_bbox)
   t_class = tf.convert_to_tensor(t_class)
   with tf.GradientTape() as tape:
      logits = model(images, training=True)
      total_loss = calc_loss_alone([t_bbox, t_class], logits, loss_weights, class_weights)
   grads = gather_gradient(model, total_loss, tape)
   return total_loss, logits, grads

@tf.function
def train_step(model, images, t_bbox, t_class):
   t_bbox = tf.convert_to_tensor(t_bbox)
   t_class = tf.convert_to_tensor(t_class)
   with tf.GradientTape() as tape:
      logits = model(images, training=True)
      losses = batched_loss_closure([t_bbox, t_class], logits, loss_weights, class_weights)
      total_loss = tf.gather(losses, [tf.shape(losses)[1]-1], axis=1)
      total_loss = tf.reduce_mean(total_loss)
#      total_loss = tf.reduce_mean(losses[:,-1]) # extract weighted total loss, take mean
   grads = gather_gradient(model, total_loss, tape)
   return total_loss, losses, grads

def fit(model, optimizer, train_datagen, epoch):
    print("\nStart of epoch %d" % (epoch))
    start_time = time.time()
    batch_times = []
    # Iterate over the batches of the dataset
    for step, (images, t_bbox, t_class) in enumerate(train_datagen):
        batch_start = time.time()
        
#        total_loss, logits, grads = train_step2(model, images, t_bbox, t_class)
        total_loss, losses, grads = train_step2(model, images, t_bbox, t_class)

        # apply the gradient
        apply_gradient(grads, optimizer)

        print("Total loss: %1.4f" % total_loss)
        
def fit_test(model, optimizer, epoch, images, t_bbox, t_class, steps):
    print("\nStart of epoch %d" % (epoch))
    start_time = time.time()
    batch_times = []
    # Iterate over the batches of the dataset
    for step in range(steps):        
#        total_loss, logits, grads = train_step2(model, images, t_bbox, t_class)
        total_loss, losses, grads = train_step(model, images, t_bbox, t_class)

        # apply the gradient
        apply_gradient(grads, optimizer)

        print("Total loss: %1.4f" % total_loss)

# Run the training for 5 epochs
#params['aug'] = False
#params['num_bbox'] = 300
#train_datagen = DataGen_curt(params, train_data, nsteps, nclone_factor, npartial_mult)
#model = pred_bbox_dense(num_bbox = params['num_bbox'])
#optimizer = setup_optimizer(model, lr = 1e-4)
#images, t_bbox, t_class = train_datagen[0]
for epoch in range(500):
#   fit_test(model, optimizer, epoch, images, t_bbox, t_class, 20)
   fit(model, optimizer, train_datagen, epoch)

model.save_weights(weights_name_next)

# test on a batch
logits = model(images)
losses = batched_loss_closure([t_bbox, t_class], logits, loss_weights, class_weights)
total_loss = tf.reduce_mean(tf.gather(losses, [tf.shape(losses)[1]-1], axis=1))
#total_loss = calc_loss_alone([t_bbox, t_class], logits)

from bbox import xcycwh_to_xy_min_xy_max
import cv2
bboxes = [xcycwh_to_xy_min_xy_max(logits[0][b,:,:]) for b in range(logits[0].shape[0])]
p_is = tf.math.sigmoid(logits[1])

norm_std = train_datagen.norm_std
norm_mean = train_datagen.norm_mean
for ba in range(images.shape[0]):
   img        = (255*(images[ba]*norm_std + norm_mean)).astype(np.uint8)
   bbx_ij     = bboxes[ba].numpy()
   p_i        = p_is[ba,:,:].numpy()
   for j in range(p_i.shape[0]):                
       bbx_j    = img.shape[0]*bbx_ij[j]
       pi_j     = np.round(p_i[j], 2)
       x = int(bbx_j[0]); y =  int(bbx_j[1]); x2 = int(bbx_j[2]); y2 =  int(bbx_j[3])
       # Rectangle   
       col = [0,255,0]
       if np.any(pi_j[:-1]>0.5): col[0] += 255
       cv2.rectangle(img,(x,y),(x2,y2),tuple(col),2)
       col = (0,255,0)
       if np.any(pi_j[:-1]>0.5):
           cv2.putText(img, str(pi_j),(x2+10,y2),0,0.3, col)
   plot_img(img)


#images = x_batch
#norm_std = train_datagen.norm_std
#norm_mean = train_datagen.norm_mean
#for ba in range(images.shape[0]):
#   img        = (255*(images[ba]*norm_std + norm_mean)).astype(np.uint8)
#   bbx_ij     = b_bboxes[ba]
#   p_i        = b_p_i[ba]
#   for j in range(p_i.shape[0]):                
#       bbx_j    = img.shape[0]*bbx_ij[j]
#       pi_j     = np.round(p_i[j], 2)
#       y = int(bbx_j[0]); x =  int(bbx_j[1]); y2 = int(bbx_j[2]); x2 =  int(bbx_j[3])
#       # Rectangle   
#       col = [0,255,0]
#       if np.any(pi_j[:-1]>0.5): col[0] += 255
#       cv2.rectangle(img,(x,y),(x2,y2),tuple(col),2)
#       col = (0,255,0)
#       if np.any(pi_j[:-1]>0.5):
#           cv2.putText(img, str(pi_j),(x2+10,y2),0,0.3, col)
#   plot_img(img)



#@tf.function
#def train_step(images, t_bbox, t_class):
#   t_bbox = tf.convert_to_tensor(t_bbox)
#   t_class = tf.convert_to_tensor(t_class)
#   with tf.GradientTape() as tape:
#      logits = model(images, training=True)
#      losses = batched_loss_closure([t_bbox, t_class], logits, loss_weights, class_weights)
#      total_loss = tf.reduce_mean(losses[:,-1]) # extract weighted total loss, take mean
#   grads = tape.gradient(total_loss, model.trainable_weights)
#   optimizer.apply_gradients(zip(grads, model.trainable_weights))
#   return total_loss, losses, grads

#@tf.function
#def test_step(model, images, t_bbox, t_class):
#   t_bbox = tf.convert_to_tensor(t_bbox)
#   t_class = tf.convert_to_tensor(t_class)
#   logits = model(images, training=False)
#   losses = batched_loss_closure([t_bbox, t_class], logits, loss_weights, class_weights)
#   total_loss = tf.reduce_mean(losses[:,-1]) # extract weighted total loss, take mean
#   return total_loss, losses
#   
#def apply_gradients(optimizer, grads):
#   optimizer.apply_gradients(zip(grads, model.trainable_weights))

loss_names = ['label_loss', 'giou_loss', 'l1_loss']
_classes = ['crypt', 'clone', 'partial', 'fufi', 'no_object']
_metrics = ['true_pos', 'false_neg', 'false_pos', 'true_neg', 'precision', 'recall', 'accuracy']
class_names = [l1+'_'+l2 for l2 in _classes for l1 in _metrics]
all_names = loss_names + class_names

loss_trackers = create_loss_metric_trackers(loss_names, _classes)
val_loss_trackers = create_loss_metric_trackers(loss_names, _classes)

## training loop
init_weights = model.trainable_weights
cur_best_val_loss = np.inf
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    batch_times = []
    # Iterate over the batches of the dataset.
    for step, (images, t_bbox, t_class) in enumerate(train_datagen):
        batch_start = time.time()
        total_loss, losses, grads = train_step(images, t_bbox, t_class)

        for kk in range(len(model.trainable_weights)):
            print(np.sum((model.trainable_weights[kk] - init_weights[kk]).numpy()))
        
        # Compute our metrics
        loss_trackers = update_trackers(losses, loss_trackers, all_names)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)        
        # Log every batch
        if step % 5 == 0:
            avbt = np.mean(batch_times)
            print("Batch loss at step %d: %.4f" % (step, float(total_loss)))
            print("Seen so far: %d samples" % ((step + 1) * params['batch_size']))
            for i, cl in enumerate(all_names):
               if type(loss_trackers[cl])==DummyMetric1:
                  print("%s to step %d: %d" % (cl, step, np.around(float(loss_trackers[cl].result()))))
               if type(loss_trackers[cl])==DummyMetric2:
                  print("%s to step %d: %1.4f" % (cl, step, float(loss_trackers[cl].result())))
            print('\n')
            print("~~~***~~~")
            print("%1.1f%% through epoch %d" % (float(step+1)/len(train_datagen) * 100, epoch) )
            print("Average batch time: %1.1fs" % avbt)
            print("Approx time left this epoch: %1.2f mins" % ((len(train_datagen)-step-1) * avbt / 60.))
            print("~~~***~~~")
            print('\n')
           
    # Display metrics at the end of each epoch.
    log_trackers(loss_trackers, all_names, loss_names, loss_weights, epoch, logpath, suffix='train')
    for i, cl in enumerate(all_names):
       print("%s over epoch: %.4f" % (cl, float(loss_trackers[cl].result())))
    
    # Reset training metrics at the end of each epoch
    loss_trackers = reset_trackers(loss_trackers, all_names)

    ## Run a validation loop at the end of each epoch.
    for (images, t_bbox, t_class) in val_datagen:
        total_loss, losses = test_step(images, t_bbox, t_class)
        # Update val metrics
        val_loss_trackers = update_trackers(losses, val_loss_trackers, all_names)
    
    # Display validation metrics
    log_trackers(val_loss_trackers, all_names, loss_names, loss_weights, epoch, logpath, suffix='val')
    for i, cl in enumerate(all_names):
       print("val %s: %.4f" % (cl, float(val_loss_trackers[cl].result())))
    
    # aggregate validation losses and save model if best
    total_val_loss = aggregate_losses(val_loss_trackers, loss_names, loss_weights)
    if total_val_loss<cur_best_val_loss:
       print('Loss improved from %1.4f to %1.4f, saving weights.' % (cur_best_val_loss, total_val_loss))
       model.save_weights(weights_name_next)
       cur_best_val_loss = total_val_loss
    else:
       print('Loss did not improve from %1.4f.' % cur_best_val_loss)
    
    # Reset validation metrics at the end of each epoch
    val_loss_trackers = reset_trackers(val_loss_trackers, all_names)
    print("Time taken: %.2fs" % (time.time() - start_time))




