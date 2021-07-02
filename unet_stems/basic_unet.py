#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:26:46 2021

@author: edward
"""
from tensorflow.keras.models       import Model
from tensorflow.keras.layers       import Input, concatenate, Conv2D, MaxPooling2D, Activation,\
                                          UpSampling2D
from tensorflow.keras import layers
import tensorflow_addons as tfa

def conv_layer(x_in, filters, gn_num, strides = 1, shape = (3,3), name=None):
    x = Conv2D(filters, shape, strides=strides, padding='same')(x_in)
    x = tfa.layers.GroupNormalization(groups=gn_num, axis=3)(x)
    x = Activation('relu', name = name)(x)
    return x

def unet_basic(input_shape=(1024, 1024, 3), chan_num=3):
    inputs = Input(shape=input_shape)
    # 1024
    down0b = conv_layer(inputs, 16, 2, strides = 2)
    down0b = conv_layer(down0b, 16, 2)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 256

    down0a = conv_layer(down0b_pool, 32, 4)
    down0a = conv_layer(down0a, 32, 4)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 128

    down0 = conv_layer(down0a_pool, 64, 4)
    down0 = conv_layer(down0, 64, 4)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 64

    down1 = conv_layer(down0_pool, 128, 8)
    down1 = conv_layer(down1, 128, 8)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 32

    down2 = conv_layer(down1_pool, 256, 8)
    down2 = conv_layer(down2, 256, 8)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 16

    down3 = conv_layer(down2_pool, 512, 16)
    down3 = conv_layer(down3, 512, 16)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 8

    center = conv_layer(down3_pool, 1024, 16)
    center = conv_layer(center, 1024, 16)
    # center

    up3 = UpSampling2D((2, 2))(center)
    up3 = concatenate([down3, up3], axis=chan_num)
    up3 = conv_layer(up3, 512, 16)
    up3 = conv_layer(up3, 512, 16)
    up3 = conv_layer(up3, 512, 16)
    # 16

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=chan_num)
    up2 = conv_layer(up2, 256, 8)
    up2 = conv_layer(up2, 256, 8)
    up2 = conv_layer(up2, 256, 8)
    # 32

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=chan_num)
    up1 = conv_layer(up1, 128, 8)
    up1 = conv_layer(up1, 128, 8)
    up1 = conv_layer(up1, 128, 8)
    # 64

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=chan_num)
    up0 = conv_layer(up0, 64, 4)
    up0 = conv_layer(up0, 64, 4)
    up0 = conv_layer(up0, 64, 4)
    # 128

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=chan_num)
    up0a = conv_layer(up0a, 32, 4)
    up0a = conv_layer(up0a, 32, 4)
    up0a = conv_layer(up0a, 32, 4, name = 'fmaps')    
    # 256

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=chan_num)
    up0b = conv_layer(up0b, 16, 2)
    up0b = conv_layer(up0b, 16, 2)
    up0b = conv_layer(up0b, 16, 2)
    # 512

    # Crypt predict
    crypt = Conv2D(1, (1, 1))(up0b)
    crypt = layers.Activation('sigmoid', dtype='float32', name='crypt_pred')(crypt)

    # just unet
    just_unet = Model(inputs=inputs, outputs=[crypt, up0a], name = 'crypt')
    return just_unet
