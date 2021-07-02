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
from unet_stems.subpixel_conv import SubpixelConv2D

def conv_layer_gn(x_in, filters, strides=1, shape=3, groups=2, name = None):
   x = layers.Conv2D(filters, shape, strides=strides, padding='same')(x_in)
   x = tfa.layers.GroupNormalization(groups=groups, axis=3)(x)
   x = layers.Activation("relu", name = name)(x)
   return x
   
def upsample_layer_gn(x_in, filters, strides=1, shape=3, groups=2, name = None):
#   x = layers.Conv2DTranspose(filters, shape, strides=strides, padding='same')(x_in)
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
#   residual = SubpixelConv2D(scale=2)(prev_res)
   residual = layers.Conv2D(filters, 1, padding='same')(residual)
   x = layers.add([x_in, residual])  # Add back residual
   return x

def strde_unet_xcept_gn_shallow(input_shape=(1024, 1024, 3), chan_num=3):
   inputs = layers.Input(shape=input_shape)
   # 1024
   x = conv_layer_gn(inputs, 16, strides=2, shape=3, groups=2)
   # 512
   down1 = x

   x = conv_layer_gn(x, 32, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 32, strides=1, shape=3, groups=4)
   x = dwnsmp_add_residual(x, down1, 32)
   # 256
   down2 = x
   
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = dwnsmp_add_residual(x, down2, 64)
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

   ######## Centre

   x = upsample_layer_gn(x, 128, strides=1, shape=3, groups=8) #strides=2
   # 64
   x = upsmp_add_residual(x, down5, 128)
   up1 = x  # Set aside next residual
   x = layers.concatenate([down4, x], axis=chan_num)
#   x = upsample_layer_gn(x, 128, strides=1, shape=3, groups=8)
   x = conv_layer_gn(x, 128, strides=1, shape=3, groups=8)
   
   x = upsample_layer_gn(x, 64, strides=1, shape=3, groups=4) #strides=2
   # 128
   x = upsmp_add_residual(x, up1, 64)
   up2 = x  # Set aside next residual
   x = layers.concatenate([down3, x], axis=chan_num)
#   x = upsample_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   
   x = upsample_layer_gn(x, 32, strides=1, shape=3, groups=4) #strides=2
   # 256
   x = upsmp_add_residual(x, up2, 32)
   up3 = x  # Set aside next residual
   x = layers.concatenate([down2, x], axis=chan_num)
#   feature_maps = upsample_layer_gn(x, 32, strides=1, shape=3, groups=4, name = 'fmaps')
   feature_maps = conv_layer_gn(x, 32, strides=1, shape=3, groups=4, name = 'fmaps')

   x = upsample_layer_gn(feature_maps, 16, strides=1, shape=3, groups=2) #strides=2
   # 512
   x = upsmp_add_residual(x, up3, 16)
   x = layers.concatenate([down1, x], axis=chan_num)
#   x = upsample_layer_gn(x, 16, strides=1, shape=3, groups=2)
   x = conv_layer_gn(x, 16, strides=1, shape=3, groups=2)

   # Crypt predict
   crypt = layers.Conv2D(1, 3, padding='same')(x)
   crypt = layers.Activation('sigmoid', dtype='float32', name = 'crypt_pred')(crypt)
   
   unet = Model(inputs=inputs, outputs=[crypt, feature_maps], name = 'crypt')
   return unet

def strde_unet_xcept_gn_deep(input_shape=(1024, 1024, 3), chan_num=3):
   inputs = layers.Input(shape=input_shape)
   # 1024
   x = conv_layer_gn(inputs, 16, strides=2, shape=3, groups=2)
   # 512
   down1 = x

   x = conv_layer_gn(x, 32, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 32, strides=1, shape=3, groups=4)
   x = dwnsmp_add_residual(x, down1, 32)
   # 256
   down2 = x
   
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = dwnsmp_add_residual(x, down2, 64)
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
   
   x = conv_layer_gn(x, 384, strides=1, shape=3, groups=24)
   x = conv_layer_gn(x, 384, strides=1, shape=3, groups=24)
   x = dwnsmp_add_residual(x, down5, 384)
   # 16
   down6 = x
   
   x = conv_layer_gn(x, 512, strides=1, shape=3, groups=32)
   x = conv_layer_gn(x, 512, strides=1, shape=3, groups=32)
   x = dwnsmp_add_residual(x, down6, 512)
   # 8
   down7 = x

   ######## Centre
   
   x = upsample_layer_gn(x, 384, strides=1, shape=3, groups=24) #strides=2
   # 64
   x = upsmp_add_residual(x, down7, 384)
   up1 = x  # Set aside next residual
   x = layers.concatenate([down6, x], axis=chan_num)
#   x = upsample_layer_gn(x, 384, strides=1, shape=3, groups=24)
   x = conv_layer_gn(x, 384, strides=1, shape=3, groups=24)

   x = upsample_layer_gn(x, 256, strides=1, shape=3, groups=16) #strides=2
   # 64
   x = upsmp_add_residual(x, up1, 256)
   up2 = x  # Set aside next residual
   x = layers.concatenate([down5, x], axis=chan_num)
#   x = upsample_layer_gn(x, 256, strides=1, shape=3, groups=16)
   x = conv_layer_gn(x, 256, strides=1, shape=3, groups=16)

   x = upsample_layer_gn(x, 128, strides=1, shape=3, groups=8) #strides=2
   # 64
   x = upsmp_add_residual(x, up2, 128)
   up3 = x  # Set aside next residual
   x = layers.concatenate([down4, x], axis=chan_num)
#   x = upsample_layer_gn(x, 128, strides=1, shape=3, groups=8)
   x = conv_layer_gn(x, 128, strides=1, shape=3, groups=8)
   
   x = upsample_layer_gn(x, 64, strides=1, shape=3, groups=4) #strides=2
   # 128
   x = upsmp_add_residual(x, up3, 64)
   up4 = x  # Set aside next residual
   x = layers.concatenate([down3, x], axis=chan_num)
#   x = upsample_layer_gn(x, 64, strides=1, shape=3, groups=4)
   x = conv_layer_gn(x, 64, strides=1, shape=3, groups=4)
   
   x = upsample_layer_gn(x, 32, strides=1, shape=3, groups=4) #strides=2
   # 256
   x = upsmp_add_residual(x, up4, 32)
   up5 = x  # Set aside next residual
   x = layers.concatenate([down2, x], axis=chan_num)
#   feature_maps = upsample_layer_gn(x, 32, strides=1, shape=3, groups=4, name = 'fmaps')
   feature_maps = conv_layer_gn(x, 32, strides=1, shape=3, groups=4, name = 'fmaps')

   x = upsample_layer_gn(feature_maps, 16, strides=1, shape=3, groups=2) #strides=2
   # 256
   x = upsmp_add_residual(x, up5, 16)
   x = layers.concatenate([down1, x], axis=chan_num)
#   x = upsample_layer_gn(x, 16, strides=1, shape=3, groups=2)
   x = conv_layer_gn(x, 16, strides=1, shape=3, groups=2)

   # Crypt predict
   crypt = layers.Conv2D(1, 3, padding='same')(x)
   crypt = layers.Activation('sigmoid', dtype='float32', name = 'crypt_pred')(crypt)
   
   unet = Model(inputs=inputs, outputs=[crypt, feature_maps], name = 'crypt')
   return unet



