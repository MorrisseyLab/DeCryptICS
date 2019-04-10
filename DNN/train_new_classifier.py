#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:16:23 2018

@author: doran
"""
import cv2
import glob
import io
import tensorflow          as tf
import keras.backend       as K
import numpy               as np
import matplotlib.pyplot   as plt
import DNN.u_net           as unet
import DNN.params          as params
import keras.callbacks     as KC
from random             import shuffle
from DNN.augmentation   import plot_img, randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip, fix_mask
from DNN.losses         import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss
from DNN.losses         import dice_coeff, MASK_VALUE, build_masked_loss, masked_accuracy, masked_dice_coeff
from keras.callbacks    import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers   import RMSprop
from PIL                import Image
from keras.preprocessing.image import img_to_array

samples = []

num_cores = 12
GPU = True
CPU = False

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

input_size = params.input_size
SIZE = (input_size, input_size)
epochs = params.max_epochs
batch_size = params.batch_size

def train_process(data):
   img_f, mask_f = data
   img = cv2.imread(img_f, cv2.IMREAD_COLOR)
   if (not img.shape==SIZE): img = cv2.resize(img, SIZE)
   
   mask = np.zeros([img.shape[0], img.shape[1], 5]) # for crypt, fufis + 3 mark types
   # Order clone channels: crypts, fufis, (KDM6A, MAOA, NONO, HDAC6, STAG2), p53, mPAS
   
   # choose which channel to load mask into
   mname = mask_f.split('/')[-1].split('.')[-2]
   if (mname[-5:]=="crypt"):
      mask[:,:,0] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
      dontmask = 0
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-100, 100),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-25, 25))
   elif (mname[-4:]=="fufi"):
      mask[:,:,1] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
      dontmask = 1
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-100, 100),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-25, 25))
   elif (mname[-5:]=="clone"):
      mname_broken = mask_f.split('/')[-1].split('_')
      if "KDM6A" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "MAOA" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "NONO" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "HDAC6" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "STAG2" in mname_broken:
         mask[:,:,2] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 2
      if "p53" in mname_broken:
         mask[:,:,3] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 3
      if "mPAS" in mname_broken:
         mask[:,:,4] = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
         dontmask = 4
      img = randomHueSaturationValue(img,
                                     hue_shift_limit=(-25, 25),
                                     sat_shift_limit=(0, 0),
                                     val_shift_limit=(-15, 15))


   img, mask = randomShiftScaleRotate(img, mask,
                                    shift_limit=(-0.0625, 0.0625),
                                    scale_limit=(-0.1, 0.1),
                                    rotate_limit=(-20, 20))
   img, mask = randomHorizontalFlip(img, mask)
   fix_mask(mask)
   
   ## Need to make masking values on outputs in float32 space, as uint8 arrays can't deal with it
   img = img.astype(np.float32) / 255
   mask = mask.astype(np.float32) / 255
   # choose which channel to mask (i.e. all other channels are masked)
   for i in range(mask.shape[2]):
      if (not i==dontmask):
         mask[:,:,i].fill(MASK_VALUE) 
   return (img, mask)

def train_generator():
    while True:
        for start in range(0, len(samples), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(samples))
            ids_train_batch = samples[start:end]
            for ids in ids_train_batch:
                img, mask = train_process(ids)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width, channel = tensor.shape
    image = Image.fromarray((tensor*255).astype(np.uint8))
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)

class TensorBoardImage(KC.Callback):
   def __init__(self, log_dir='./logs', tags=[], test_image_batches=[]):
      super().__init__()
      self.tags = tags
      self.log_dir = log_dir
      self.test_image_batches = test_image_batches

   def on_epoch_end(self, epoch, logs=None):
      writer = tf.summary.FileWriter(self.log_dir)
      for i in range(len(self.tags)):
         batch = self.test_image_batches[i]
         tag = self.tags[i]
         pred = model.predict(batch)
         pred1 = np.zeros(batch[0].shape, dtype=np.float32)
         pred2 = np.zeros(batch[0].shape, dtype=np.float32)
         for i in range(3):
            pred1[:,:,i] = pred[0,:,:,0]
            pred2[:,:,i] = pred[0,:,:,1]
         output = np.hstack([batch[0], pred1, pred2])
         image = make_image(output)
         summary_i = tf.Summary(value=[tf.Summary.Value(tag=tag, image=image)])
         writer.add_summary(summary_i, epoch)
      writer.close()
      return

if __name__=="__main__":
   base_folder = "/home/doran/Work/py_code/DeCryptICS/DNN/" # as training data is in DeCryptICS folder
   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/"
   
   # Loading old weights into all but the final layer
#   model = params.model_factory(input_shape=(params.input_size, params.input_size, 3), num_classes=7)
#   model.load_weights("./DNN/weights/cryptfuficlone_weights.hdf5")

#   # Getting weights layer by layer
#   weights_frozen = [l.get_weights() for l in model.layers]

   # Redefine new network with new classification
   model = params.model_factory(input_shape=(params.input_size, params.input_size, 3), num_classes=5)
   model.load_weights(dnnfolder+"/weights/cryptfuficlone_weights.hdf5")

   # Add in old weights
#   numlayers = len(model.layers)
#   for i in range(numlayers-1):
#      model.layers[i].set_weights(weights_frozen[i])

#   w_elems = []
#   w_f_elems = weights_frozen[-1]
#   for i in range(len(model.layers[-1].get_weights())):
#      w_elems.append(model.layers[-1].get_weights()[i])   
#   w_elems[0][:,:,:,0] = w_f_elems[0][:,:,:,0] # crypt
#   w_elems[0][:,:,:,1] = w_f_elems[0][:,:,:,1] # fufi
#   w_elems[0][:,:,:,2] = w_f_elems[0][:,:,:,2] # kdm6a/maoa/nono
#   w_elems[0][:,:,:,3] = w_f_elems[0][:,:,:,5] # stag2
#   w_elems[0][:,:,:,4] = w_f_elems[0][:,:,:,6] # mpas
#   w_elems[1][0] = w_f_elems[1][0]
#   w_elems[1][1] = w_f_elems[1][1]
#   w_elems[1][2] = w_f_elems[1][2]
#   w_elems[1][3] = w_f_elems[1][5]
#   w_elems[1][4] = w_f_elems[1][6]
#   model.layers[-1].set_weights(w_elems)

   # Freeze all layer but the last classification convolution (as difficult to freeze a subset of parameters within a layer -- but can load them back in afterwards)
#   for layer in model.layers[:-1]:
#      layer.trainable = False
#   # To check whether we have successfully frozen layers, check model.summary() before and after re-compiling
#   model.compile(optimizer=RMSprop(lr=0.0001), loss=build_masked_loss(K.binary_crossentropy), metrics=[masked_dice_coeff])
   
   # Set up training data   
   training_base_folder = "/home/doran/Work/py_code/DeCryptICS/DNN/"
   imgfolder = training_base_folder + "/input/train/"
   maskfolder = training_base_folder + "/input/train_masks/"
   crypts = glob.glob(imgfolder + "*_crypt.png")
   fufis = glob.glob(imgfolder + "*_fufi.png")
   clones = glob.glob(imgfolder + "*_clone.png")
   p53imgfolder = training_base_folder + "/input/p53/train/"
   p53maskfolder = training_base_folder + "/input/p53/train_masks/"
   p53clones = glob.glob(p53imgfolder + "*_clone.png")
   samples_cr = []
   for i in range(len(crypts)):
      mask = maskfolder+"mask"+crypts[i][(len(imgfolder)+3):]
      sample = (crypts[i], mask)
      samples_cr.append(sample)
   samples_fu = []
   for i in range(len(fufis)):
      mask = maskfolder+"mask"+fufis[i][(len(imgfolder)+3):]
      sample = (fufis[i], mask)
      samples_fu.append(sample)
   samples_cl = []
   for i in range(len(clones)):
      mask = maskfolder+"mask"+clones[i][(len(imgfolder)+3):]
      sample = (clones[i], mask)
      samples_cl.append(sample)
   for i in range(len(p53clones)):
      mask = p53maskfolder+"mask"+p53clones[i][(len(p53imgfolder)+3):]
      sample = (p53clones[i], mask)
      samples_cl.append(sample)
   
   # add crypt samples
   samples += samples_cr
   # add repeats of clone and fufi data to scale up to same as crypts?
   n1 = int(len(samples_cr)/len(samples_cl)/1.)
   n2 = int(len(samples_cr)/len(samples_fu)/1.)
   for i in range(n1): samples += samples_cl
   for i in range(n2): samples += samples_fu
   shuffle(samples)
   
   # Define test image batches for TensorBoard checking
#   test_img1 = cv2.imread(base_folder+"/input/train/img_674374_4.00-46080-24576-1024-1024_fufi.png")
#   test_img2 = cv2.imread(base_folder+"/input/train/img_618446_x6_y1_tile2_1_crypt.png")
#   test_img3 = cv2.imread(base_folder+"/input/train/img_618446_x6_y3_tile4_3_crypt.png")
#   test_img4 = cv2.imread(base_folder+"/input/train/img_652593_4.00-18432-16384-1024-1024_fufi.png")
#   test_img5 = cv2.imread(base_folder+"/input/train/img_601163_x3_y0_tile14_8_crypt.png")
#   test_images = [test_img1, test_img2, test_img3, test_img4, test_img5]
#   test_batches = []
#   for i in range(len(test_images)):
#      test_batches.append(np.array([test_images[i]], np.float32) / 255.)
#   test_tags = list(np.asarray(range(len(test_batches))).astype(str))
   
   ## subset samples for tensorboard test
#   images = [base_folder+"/input/train/img_674374_4.00-46080-24576-1024-1024_fufi.png", base_folder+"/input/train/img_618446_x6_y1_tile2_1_crypt.png", base_folder+"/input/train/img_618446_x6_y3_tile4_3_crypt.png", base_folder+"/input/train/img_652593_4.00-18432-16384-1024-1024_fufi.png", base_folder+"/input/train/img_601163_x3_y0_tile14_8_crypt.png"]
   
   
   weights_name = dnnfolder+"/weights/cryptfuficlone_weights2.hdf5"
   logs_folder = dnnfolder+"/logs"
   
   callbacks = [EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta=1e-8),
                ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, epsilon=1e-8),
                ModelCheckpoint(monitor='loss', filepath=weights_name, save_best_only=True, save_weights_only=True),
                TensorBoard(log_dir=logs_folder)]
                #TensorBoardImage(log_dir=logs_folder, tags=test_tags, test_image_batches=test_batches)]
                
   model.fit_generator(generator=train_generator(), steps_per_epoch=np.ceil(float(len(samples)) / float(batch_size)), epochs=epochs, verbose=1, callbacks=callbacks, validation_data=None)

