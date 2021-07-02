#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 10:06:52 2021

@author: edward
"""
from tensorflow.keras.models       import Model
from tensorflow.keras.layers       import Input, concatenate, Conv2D, MaxPooling2D, Activation,\
                                          UpSampling2D
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.optimizers   import RMSprop



def conv_layer(x_in, filters, gn_num, do_relu = True, strides = 1, shape = (3,3)):
    x = Conv2D(filters, shape, strides=strides, padding='same')(x_in)
    x = tfa.layers.GroupNormalization(groups=gn_num, axis=3)(x)
    # x = layers.BatchNormalization()(x)
    if do_relu:
        x = Activation('relu')(x)
    return x

def add_res_dwn(x, x_res, filters):
    residual = layers.Conv2D(filters, 1, strides=2, padding="same")(x_res)
    x = layers.add([x, residual])
    return x

def add_res_up(x, x_res, filters):
    x_res = UpSampling2D((2, 2))(x_res)
    residual = layers.Conv2D(filters, 1, padding="same")(x_res)
    x = layers.add([x, residual])
    return x

def add_res(x, x_res, filters):
    residual = layers.Conv2D(filters, 1, padding="same")(x_res)
    x = layers.add([x, residual])
    return x

def res_unet(input_shape=(1024, 1024, 3), chan_num=3):
    inputs = Input(shape=input_shape)
    # 1024
    down0b = conv_layer(inputs, 16, 2, strides = 2)
    down0b = conv_layer(down0b, 16, 2)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # down0b_pool = conv_layer(inputs, 32, 4, strides = 4, shape= 5)

    # 256

    down0a = conv_layer(down0b_pool, 32, 4)
    down0a = conv_layer(down0a, 32, 4)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    down0a_pool = add_res_dwn(down0a_pool, down0b_pool, 32) # res coonect

    # 128

    down0 = conv_layer(down0a_pool, 64, 4)
    down0 = conv_layer(down0, 64, 4)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down0_pool = add_res_dwn(down0_pool, down0a_pool, 64) # res coonect

    # 64

    down1 = conv_layer(down0_pool, 128, 8)
    down1 = conv_layer(down1, 128, 8)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down1_pool = add_res_dwn(down1_pool, down0_pool, 128) # res coonect

    # 32

    down2 = conv_layer(down1_pool, 256, 8)
    down2 = conv_layer(down2, 256, 8)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down2_pool = add_res_dwn(down2_pool, down1_pool, 256) # res coonect
    # 16

    down3 = conv_layer(down2_pool, 512, 16)
    down3 = conv_layer(down3, 512, 16)
    down3 = add_res(down3, down2_pool, 512)

    up2 = UpSampling2D((2, 2))(down3)
    up2_c = concatenate([down2, up2], axis=chan_num)
    up2 = conv_layer(up2_c, 256, 8)
    up2 = conv_layer(up2, 256, 8)
    up2 = add_res(up2, up2_c, 256)

    # 32

    up1   = UpSampling2D((2, 2))(up2)
    up1_c = concatenate([down1, up1], axis=chan_num)
    up1   = conv_layer(up1_c, 128, 8)
    up1   = conv_layer(up1, 128, 8)
    up1   = add_res(up1, up1_c, 128)

    # 64

    up0   = UpSampling2D((2, 2))(up1)
    up0_c = concatenate([down0, up0], axis=chan_num)
    up0   = conv_layer(up0_c, 64, 4)
    up0   = conv_layer(up0, 64, 4)
    up0   = add_res(up0, up0_c, 64)

    # 128

    up0a   = UpSampling2D((2, 2))(up0)
    up0a_c = concatenate([down0a, up0a], axis=chan_num)
    up0a   = conv_layer(up0a_c, 32, 4)
    up0a   = conv_layer(up0a, 32, 4)
    up0a   = add_res(up0a, up0a_c, 32)

    # 256

    # up0b = UpSampling2D((2, 2))(up0a)
    # up0b_c = concatenate([down0b, up0b], axis=chan_num)
    up0b_c = UpSampling2D((2, 2))(up0a)
    up0b = conv_layer(up0b_c, 8, 2)
    up0b = add_res(up0b, up0b_c, 8)

    # 512

    # Crypt predict
    crypt_fufi = Conv2D(1, (1, 1))(up0b)
    crypt_fufi = layers.Activation('sigmoid', dtype='float32', name='crypt')(crypt_fufi)

    # just unet
    just_unet = Model(inputs=inputs, outputs=[crypt_fufi, up0a])
    return just_unet

def res_unet_thmb(input_shape=(1024, 1024, 3), chan_num=3):
    inputs = Input(shape=input_shape)
    # 1024
    down0b = conv_layer(inputs, 16, 2, strides = 2)
    down0b = conv_layer(down0b, 16, 2)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # down0b_pool = conv_layer(inputs, 32, 4, strides = 4, shape= 5)

    # 256

    down0a = conv_layer(down0b_pool, 32, 4)
    down0a = conv_layer(down0a, 32, 4)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    down0a_pool = add_res_dwn(down0a_pool, down0b_pool, 32) # res coonect

    # 128

    down0 = conv_layer(down0a_pool, 64, 4)
    down0 = conv_layer(down0, 64, 4)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down0_pool = add_res_dwn(down0_pool, down0a_pool, 64) # res coonect

    # 64

    down1 = conv_layer(down0_pool, 128, 8)
    down1 = conv_layer(down1, 128, 8)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down1_pool = add_res_dwn(down1_pool, down0_pool, 128) # res coonect

    # 32

    down2 = conv_layer(down1_pool, 256, 8)
    down2 = conv_layer(down2, 256, 8)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down2_pool = add_res_dwn(down2_pool, down1_pool, 256) # res coonect
    # 16

    down3 = conv_layer(down2_pool, 512, 16)
    down3 = conv_layer(down3, 512, 16)
    down3 = add_res(down3, down2_pool, 512)

    up2 = UpSampling2D((2, 2))(down3)
    up2_c = concatenate([down2, up2], axis=chan_num)
    up2 = conv_layer(up2_c, 256, 8)
    up2 = conv_layer(up2, 256, 8)
    up2 = add_res(up2, up2_c, 256)

    # 32

    up1   = UpSampling2D((2, 2))(up2)
    up1_c = concatenate([down1, up1], axis=chan_num)
    up1   = conv_layer(up1_c, 128, 8)
    up1   = conv_layer(up1, 128, 8)
    up1   = add_res(up1, up1_c, 128)

    # 64

    up0   = UpSampling2D((2, 2))(up1)
    up0_c = concatenate([down0, up0], axis=chan_num)
    up0   = conv_layer(up0_c, 64, 4)
    up0   = conv_layer(up0, 64, 4)
    up0   = add_res(up0, up0_c, 64)

    # 128

    up0a   = UpSampling2D((2, 2))(up0)
    up0a_c = concatenate([down0a, up0a], axis=chan_num)
    up0a   = conv_layer(up0a_c, 32, 4)
    up0a   = conv_layer(up0a, 32, 4)
    up0a   = add_res(up0a, up0a_c, 32)

    # 256
    up0a   = UpSampling2D((2, 2))(up0a)
    up0a_c = concatenate([down0b, up0a], axis=chan_num)
    up0a   = conv_layer(up0a_c, 16, 2)
    up0a   = conv_layer(up0a, 16, 2)
    up0a   = add_res(up0a, up0a_c, 16)

    # 512

    # up0b = UpSampling2D((2, 2))(up0a)
    # up0b_c = concatenate([down0b, up0b], axis=chan_num)
    up0b_c = UpSampling2D((2, 2))(up0a)
    up0b = conv_layer(up0b_c, 8, 2)
    up0b = add_res(up0b, up0b_c, 8)
    # 1024

    # Crypt predict
    crypt_fufi = Conv2D(1, (1, 1))(up0b)
    crypt_fufi = layers.Activation('sigmoid', dtype='float32', name='crypt')(crypt_fufi)

    # just unet
    just_unet = Model(inputs=inputs, outputs=crypt_fufi)
    return just_unet

# final_model = res_unet()
# final_model.summary()


