#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 08:40:41 2021

@author: edward
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers   import RMSprop, SGD
from unet_stems.subpixel_conv import SubpixelConv2D
from roi_pool_xcycwh import ROIPoolingLayer, PatchEncoder_w_position
#from training.losses import get_losses, DummyMetric

#loss_names = ['label_loss', 'giou_loss', 'l1_loss']
#loss_trackers = {}
#for i, cl in enumerate(loss_names):
#    cldict = {        
#        "{}".format(cl): DummyMetric(name="{}".format(cl)),
#        "{}".format(cl): DummyMetric(name="{}".format(cl)),
#        "{}".format(cl): DummyMetric(name="{}".format(cl))
#    }
#    loss_trackers.update(cldict)

#class_names = ['crypt', 'clone', 'partial', 'fufi', 'no_object']
#metric_trackers = {}
#for i, cl in enumerate(class_names):
#    cldict = {        
#        "true_pos_{}".format(cl): DummyMetric(name="true_pos_{}".format(cl)),
#        "false_neg_{}".format(cl): DummyMetric(name="false_neg_{}".format(cl)),
#        "false_pos_{}".format(cl): DummyMetric(name="false_pos_{}".format(cl)),
#        "true_neg_{}".format(cl): DummyMetric(name="true_neg_{}".format(cl)),
#        "precision_{}".format(cl): DummyMetric(name="precision_{}".format(cl)),
#        "recall_{}".format(cl): DummyMetric(name="recall_{}".format(cl)),
#        "accuracy_{}".format(cl): DummyMetric(name="accuracy_{}".format(cl))
#    }
#    metric_trackers.update(cldict)

#class DeCryptICS(tf.keras.Model):
#    
#    def __init__(self, inputs, outputs, loss_weights, class_weights, name):
#        super(DeCryptICS, self).__init__(inputs=inputs, outputs=outputs, name=name)
#        self.loss_weights = loss_weights
#        self.class_weights = class_weights
#        self.loss_names = ['label_loss', 'giou_loss', 'l1_loss']
#        class_names = ['crypt', 'clone', 'partial', 'fufi', 'no_object']
#        metric_names = ['true_pos', 'false_neg', 'false_pos', 'true_neg', 'precision', 'recall', 'accuracy']
#        self.all_metrics = [l1+'_'+l2 for l1 in metric_names for l2 in class_names]

#        
#    def train_step(self, data_gen):
#        # Unpack the data. Its structure depends on your model and
#        # on what you pass to `fit()`.
#        images, t_bbox, t_class = data_gen
#        t_bbox = tf.convert_to_tensor(t_bbox)
#        t_class = tf.convert_to_tensor(t_class)
#        with tf.GradientTape() as tape:
#            y_pred = self(images, training=True)  # Forward pass
#            # Compute our own loss
#            total_loss, losses = get_losses([t_bbox, t_class], y_pred, self.loss_weights, self.class_weights)

#        # Compute gradients
#        trainable_vars = self.trainable_variables
#        gradients = tape.gradient(total_loss, trainable_vars)

#        # Update weights
#        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#        # Compute our own metrics
#        outdict = {}
#        for i, cl in enumerate(self.loss_names):
#            loss_trackers[cl].update_state(losses[cl])
#            outdict.update({'{}'.format(cl) : loss_trackers[cl].result()})
#        for i, cl in enumerate(self.all_metrics):
#            metric_trackers[cl].update_state(losses[cl])
#            outdict.update({'{}'.format(cl) : metric_trackers[cl].result()})
#        return outdict

#    @property
#    def metrics(self):
#        # We list our `Metric` objects here so that `reset_states()` can be
#        # called automatically at the start of each epoch
#        # or at the start of `evaluate()`.
#        # If you don't implement this property, you have to call
#        # `reset_states()` yourself at the time of your choosing.
#        outlist = []
#        for i, cl in enumerate(self.loss_names):
#            outlist.append(loss_trackers[cl])
#        for i, cl in enumerate(self.all_metrics):
#            outlist.append(metric_trackers[cl])
#        return outlist


## Construct an instance of CustomModel
#inputs = keras.Input(shape=(32,))
#outputs = keras.layers.Dense(1)(inputs)
#model = CustomModel(inputs, outputs)

## We don't passs a loss or metrics here.
#model.compile(optimizer="adam")

## Just use `fit` as usual -- you can use callbacks, etc.
#x = np.random.random((1000, 32))
#y = np.random.random((1000, 1))
#model.fit(x, y, epochs=5)

def conv_layer_gn(x_in, filters, strides=1, shape=3, groups=2, name = None):
   x = layers.Conv2D(filters, shape, strides=strides, padding='same')(x_in)
   x = tfa.layers.GroupNormalization(groups=groups, axis=3)(x)
   x = layers.Activation("relu", name = name)(x)
   return x
   
def upsample_layer_gn(x_in, filters, strides=1, shape=3, groups=2, name = None):
   x = SubpixelConv2D(scale=2)(x_in)
   x = layers.Conv2D(filters, shape, strides=strides, padding='same')(x)
   x = tfa.layers.GroupNormalization(groups=groups, axis=3)(x)
   x = layers.Activation("relu", name = name)(x)
   return x

def sep_conv_layer_gn(x_in, filters, strides=1, shape=3, groups=2, name = None):
   x = layers.SeparableConv2D(filters, shape, strides=strides, padding='same')(x_in)
   x = tfa.layers.GroupNormalization(groups=groups, axis=3)(x)   
   x = layers.Activation("relu", name = name)(x)
   return x   

def dwnsmp_add_residual(x_in, prev_res, filters):
   x = layers.MaxPooling2D(3, strides=2, padding='same')(x_in)
   residual = layers.Conv2D(filters, 1, strides=2, padding='same')(prev_res)
   x = layers.add([x, residual])  # Add back residual
   return x

def upsmp_add_residual(x_in, prev_res, filters):
   residual = layers.UpSampling2D(2)(prev_res)
   residual = layers.Conv2D(filters, 1, padding='same')(residual)
   x = layers.add([x_in, residual])  # Add back residual
   return x

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

#input_shape=(1024, 1024, 3); chan_num=3; roi_pool_size=[10, 10]; num_bbox=400; projection_dim=100; transformer_layers=4; num_heads=4; num_classes=5; encode_dim=100

def pred_bbox_short(input_shape=(1024, 1024, 3), chan_num=3, roi_pool_size=[10, 10], num_bbox=400, projection_dim=100, transformer_layers=4, num_heads=4, num_classes=5, encode_dim=100): #, loss_weights=[1, 2, 5], class_weights=[1,1.5,2,1.25,1]):
   imgs = layers.Input(shape=input_shape)
   # U-net stem for feature maps and encoding =========
   # 1024
   x = conv_layer_gn(imgs, 24, strides=2, shape=3, groups=2)
   # 512
   down1 = x

   x = conv_layer_gn(x, 32, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 32, strides=1, shape=3, groups=4)
   x = dwnsmp_add_residual(x, down1, 32)
   # 256
   down2 = x
   
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   feature_maps = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = dwnsmp_add_residual(feature_maps, down2, 64)
   # 128
   down3 = x
   
   x = conv_layer_gn(x, 128, strides=1, shape=3, groups=8)
   x = conv_layer_gn(x, 128, strides=1, shape=3, groups=8)
   x = dwnsmp_add_residual(x, down3, 128)
   # 64
   down4 = x
   
   x = conv_layer_gn(x, 256, strides=1, shape=3, groups=16)
   x = conv_layer_gn(x, num_bbox, strides=1, shape=3, groups=16)
   x = dwnsmp_add_residual(x, down4, num_bbox)
   # 32
   
   # Predict bounding boxes =========
   encoding = layers.Reshape((num_bbox,-1))(x)
   encoding = layers.Dense(2*encode_dim, activation = 'relu')(encoding)
   encoding = layers.Dense(encode_dim, activation = 'relu')(encoding)
   bboxes = layers.Dense(4, activation = 'sigmoid')(encoding)

   # Transformer part =========
   feature_maps = conv_layer_gn(feature_maps, 32, strides=1, shape=3, groups=4, name='fmaps')
   pooled_features = ROIPoolingLayer(roi_pool_size[0],roi_pool_size[1])([feature_maps, bboxes])
   c_p_f = PatchEncoder_w_position(num_bbox, projection_dim, 128)([pooled_features, bboxes])

   # Create multiple layers of the Transformer block.
   for _ in range(transformer_layers):
      # Layer normalization 1.
      x1 = layers.LayerNormalization(epsilon=1e-6)(c_p_f)
      # Create a multi-head attention layer.
      attention_output = layers.MultiHeadAttention(
         num_heads=num_heads, key_dim=projection_dim, dropout=0.15
      )(x1, x1)
      # Skip connection 1.
      x2 = layers.Add()([attention_output, c_p_f])
      # Layer normalization 2.
      x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
      # MLP.
      x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.15)
      # Skip connection 2.
      c_p_f = layers.Add()([x3, x2])

   # Create a [batch_size, projection_dim] tensor.
   c_p_f = layers.LayerNormalization(epsilon=1e-6)(c_p_f)
   c_p_f = layers.Dropout(0.3)(c_p_f) # increased from 0.2

   class_predict = layers.Dense(num_classes)(c_p_f)
   # probabilities...
#   class_predict = layers.Activation('sigmoid', dtype='float32', name='classes')(class_predict)
   # ... or logits
   class_predict = tf.cast(class_predict, 'float32', name='classes')
   bboxes = tf.cast(bboxes, 'float32', name='boxes')
   
   pbboxmodel = Model(inputs = imgs, outputs = [bboxes, class_predict], name = 'pbox_net') #, loss_weights = loss_weights, class_weights = class_weights)
   
   return pbboxmodel

def pred_bbox_dense(input_shape=(1024, 1024, 3), chan_num=3, roi_pool_size=[10, 10], num_bbox=300, projection_dim=100, transformer_layers=4, num_heads=4, num_classes=5):
   imgs = layers.Input(shape=input_shape)
   # U-net stem for feature maps and encoding =========
   # 1024
   x = conv_layer_gn(imgs, 24, strides=2, shape=3, groups=2)
   # 512
   down1 = x

   x = conv_layer_gn(x, 32, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 32, strides=1, shape=3, groups=4)
   x = dwnsmp_add_residual(x, down1, 32)
   # 256
   down2 = x
   
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   feature_maps = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = dwnsmp_add_residual(feature_maps, down2, 64)
   # 128
   down3 = x
   
   x = conv_layer_gn(x, 128, strides=1, shape=3, groups=8)
   x = conv_layer_gn(x, 128, strides=1, shape=3, groups=8)
   x = dwnsmp_add_residual(x, down3, 128)
   # 64
   down4 = x
   
   x = conv_layer_gn(x, 256, strides=1, shape=3, groups=16)
   x = conv_layer_gn(x, 256, strides=1, shape=3, groups=16)
   x = dwnsmp_add_residual(x, down4, 256)
   # 32
   down5 = x
   
   x = conv_layer_gn(x, 512, strides=1, shape=3, groups=16)
   x = conv_layer_gn(x, 512, strides=1, shape=3, groups=16)
   x = dwnsmp_add_residual(x, down5, 512)
   # 16
   down6 = x

   x = conv_layer_gn(x, 768, strides=1, shape=3, groups=16)
   x = conv_layer_gn(x, 768, strides=1, shape=3, groups=16)
   x = dwnsmp_add_residual(x, down6, 768)
   # 8
   down7 = x
   
   x = conv_layer_gn(x, 1024, strides=1, shape=3, groups=32)
   x = conv_layer_gn(x, 384, strides=1, shape=3, groups=16)
   x = dwnsmp_add_residual(x, down7, 384)
   # 4
   
   # Predict bounding boxes =========
   encoding = layers.Flatten()(x)   
   encoding = layers.Dense(8*num_bbox, activation = 'relu')(encoding)
   encoding = layers.Dense(4*num_bbox, activation = 'relu')(encoding)
   encoding = layers.Reshape((num_bbox,-1))(encoding)
   bboxes = layers.Dense(4, activation = 'sigmoid')(encoding)

   # Transformer part =========
   feature_maps = conv_layer_gn(feature_maps, 32, strides=1, shape=3, groups=4, name='fmaps')
   pooled_features = ROIPoolingLayer(roi_pool_size[0],roi_pool_size[1])([feature_maps, bboxes])
   c_p_f = PatchEncoder_w_position(num_bbox, projection_dim, 128)([pooled_features, bboxes])

   # Create multiple layers of the Transformer block.
   for _ in range(transformer_layers):
      # Layer normalization 1.
      x1 = layers.LayerNormalization(epsilon=1e-6)(c_p_f)
      # Create a multi-head attention layer.
      attention_output = layers.MultiHeadAttention(
         num_heads=num_heads, key_dim=projection_dim, dropout=0.15
      )(x1, x1)
      # Skip connection 1.
      x2 = layers.Add()([attention_output, c_p_f])
      # Layer normalization 2.
      x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
      # MLP.
      x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.15)
      # Skip connection 2.
      c_p_f = layers.Add()([x3, x2])

   # Create a [batch_size, projection_dim] tensor.
   c_p_f = layers.LayerNormalization(epsilon=1e-6)(c_p_f)
   c_p_f = layers.Dropout(0.3)(c_p_f) # increased from 0.2

   class_predict = layers.Dense(num_classes)(c_p_f)
   # probabilities...
#   class_predict = layers.Activation('sigmoid', dtype='float32', name='classes')(class_predict)
   # ... or logits
   class_predict = tf.cast(class_predict, 'float32', name='classes')
   bboxes = tf.cast(bboxes, 'float32', name='boxes')
   
   pbboxmodel = Model(inputs = imgs, outputs = [bboxes, class_predict], name = 'pbox_net') #, loss_weights = loss_weights, class_weights = class_weights)
   
   return pbboxmodel

