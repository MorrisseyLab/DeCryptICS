import tensorflow as tf
from keras import backend as K
import cv2, os, time
import numpy as np
import pyvips
import keras
import pickle
from keras.preprocessing.image import img_to_array
import DNN.u_net as unet
import DNN.params as params

## Loading old weights into all but the final layer
model = params.model_factory(input_shape=(params.input_size, params.input_size, 3))
model.load_weights("./DNN/weights/tile256_for_X_best_weights.hdf5")

# Getting weights layer by layer
weights_frozen = [l.get_weights() for l in model.layers]

# Redefine new network with new classification
model = params.model_factory(input_shape=(params.input_size, params.input_size, 3), num_classes=2)

# Add in old weights, not including final layer
numlayers = len(model.layers)
for i in range(numlayers-1):
   model.layers[i].set_weights(weights_frozen[i])
   
# Now add the old weights for the first classification convolution
w_elems = []
w_f_elems = weights_frozen[-1]
for i in range(len(model.layers[-1].get_weights())):
   w_elems.append(model.layers[-1].get_weights()[i])   
w_elems[0][:,:,:,0] = w_f_elems[0][:,:,:,0]
w_elems[1][0] = w_f_elems[1][0]
model.layers[-1].set_weights(w_elems)

# Freeze all layer but the last classification convolution (as difficult to freeze a subset of parameters within a layer -- but can load them back in afterwards)
for layer in model.layers[:-1]:
   layer.trainable = False
# To check whether we have successfully frozen layers, check model.summary() before and after re-compiling model?   
model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

# Train...


# Once trained with new fufi data, re-insert the old crypt classification weights into the classification convolution layer
# (though be careful to make sure the order of the classification dimensions hasn't changed)
w_elems = []
for i in range(len(model.layers[-1].get_weights())):
   w_elems.append(model.layers[-1].get_weights()[i])
w_elems[0][:,:,:,0] = w_f_elems[0][:,:,:,0]
w_elems[1][0] = w_f_elems[1][0]
model.layers[-1].set_weights(w_elems)
model.save_weights("./DNN/weights/tile256_for_X_best_weights_fufi.hdf5")
   
