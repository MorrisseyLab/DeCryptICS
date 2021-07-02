#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:04:07 2021

@author: edward
"""

from tensorflow.keras.models       import Model
from tensorflow.keras.layers       import Input, concatenate, Conv2D, MaxPooling2D, Activation,\
                                          UpSampling2D
from tensorflow.keras import layers

def conv_layer(x_in, filters, strides = 1, shape = (3,3)):
    x = Conv2D(filters, shape, strides=strides, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def relu_conv_bn(x_in, filters, strides = 1, shape = (3,3)):
    x = Activation('relu')(x_in)
    x = Conv2D(filters, shape, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def add_res_dwn(x, x_res, filters):
    x_res = MaxPooling2D((2, 2), strides=(2, 2))(x_res)
    residual = layers.Conv2D(filters, 1, strides=1, padding="same")(x_res)
    x = layers.add([x, residual])
    return x

def add_res_up(x, x_res, filters):
    x_res = UpSampling2D((2, 2))(x_res)
    residual = layers.Conv2D(filters, 1, padding="same")(x_res)
    x = layers.add([x, residual])
    return x

def add_res(x, x_res, filters, name = None):
    residual = layers.Conv2D(filters, 1, padding="same")(x_res)
    x = layers.add([x, residual], name = name)
    return x

def res_unet(input_shape=(1024, 1024, 3), chan_num=3):
    inputs = Input(shape=input_shape)
    # 1024
    down0b_pool = conv_layer(inputs, 32, strides = 4, shape= 5)

    # 256

    down0a = relu_conv_bn(down0b_pool, 32)
    down0a = relu_conv_bn(down0a, 32)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    down0a_pool = add_res_dwn(down0a_pool, down0b_pool, 32) # res coonect

    # 128

    down0 = relu_conv_bn(down0a_pool, 64)
    down0 = relu_conv_bn(down0, 64)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down0_pool = add_res_dwn(down0_pool, down0a_pool, 64) # res coonect

    # 64

    down1 = relu_conv_bn(down0_pool, 128)
    down1 = relu_conv_bn(down1, 128)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down1_pool = add_res_dwn(down1_pool, down0_pool, 128) # res coonect

    # 32

    down2 = relu_conv_bn(down1_pool, 256)
    down2 = relu_conv_bn(down2, 256)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down2_pool = add_res_dwn(down2_pool, down1_pool, 256) # res coonect
    # 16

    down3 = relu_conv_bn(down2_pool, 512)
    down3 = relu_conv_bn(down3, 512)
    down3 = add_res(down3, down2_pool, 512)

    up2 = UpSampling2D((2, 2))(down3)
    up2_c = concatenate([down2, up2], axis=chan_num)
    up2 = relu_conv_bn(up2_c, 256)
    up2 = relu_conv_bn(up2, 256)
    up2 = add_res(up2, up2_c, 256)

    # 32

    up1   = UpSampling2D((2, 2))(up2)
    up1_c = concatenate([down1, up1], axis=chan_num)
    up1   = relu_conv_bn(up1_c, 128)
    up1   = relu_conv_bn(up1, 128)
    up1   = add_res(up1, up1_c, 128)

    # 64

    up0   = UpSampling2D((2, 2))(up1)
    up0_c = concatenate([down0, up0], axis=chan_num)
    up0   = relu_conv_bn(up0_c, 64)
    up0   = relu_conv_bn(up0, 64)
    up0   = add_res(up0, up0_c, 64)

    # 128

    up0a   = UpSampling2D((2, 2))(up0)
    up0a_c = concatenate([down0a, up0a], axis=chan_num)
    up0a   = relu_conv_bn(up0a_c, 32)
    up0a   = relu_conv_bn(up0a, 32)
    up0a   = add_res(up0a, up0a_c, 32, name = 'fmaps')

    # 256

    # up0b = UpSampling2D((2, 2))(up0a)
    # up0b_c = concatenate([down0b, up0b], axis=chan_num)
    up0b_c = UpSampling2D((2, 2))(up0a)
    up0b = conv_layer(up0b_c, 8)
    up0b = add_res(up0b, up0b_c, 8)

    # 512

    # Crypt predict
    crypt = Conv2D(1, (1, 1))(up0b)
    crypt = layers.Activation('sigmoid', dtype='float32', name='crypt_pred')(crypt)

    # just unet
    just_unet = Model(inputs=inputs, outputs=[crypt, up0a], name = 'crypt')
    return just_unet

# final_model = res_unet()
# final_model.summary()


