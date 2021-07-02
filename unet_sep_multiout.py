#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:03:24 2021

@author: edward
"""
import tensorflow as tf
from tensorflow.keras.models       import Model
from tensorflow.keras.layers       import Input
from tensorflow.keras              import layers
from tensorflow.keras              import metrics
from tensorflow.keras.optimizers   import RMSprop, SGD
from roi_pool import ROIPoolingLayer, PatchEncoder_w_position
from model_set_parameter_dicts import set_params
params = set_params()

from unet_stems.slim_resunet import res_unet
# from unet_stems.slim_resunet_bnorm import res_unet
from unet_stems.xception_unets_sepconv import strde_sepconv_unet_xcept_gn_shallow, strde_sepconv_unet_xcept_gn_deep
from unet_stems.xception_unets import strde_unet_xcept_gn_shallow, strde_unet_xcept_gn_deep


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def classify_branch(input_shape=(256, 256, 32), roi_pool_size = [10, 10], num_bbox = 400, 
                    chan_num=3, projection_dim = 100, transformer_layers = 4, num_heads = 4, crypt_class=False):
    Input_bbox = Input(shape=(num_bbox, 4))
    fmap       = Input(shape=input_shape)
    
    # Transformer part =========
    pooled_features = ROIPoolingLayer(roi_pool_size[0],roi_pool_size[1])([fmap, Input_bbox])
    c_p_f = PatchEncoder_w_position(num_bbox, projection_dim, 128)([pooled_features, Input_bbox])
    
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
    
    clone = layers.Dense(1)(c_p_f)
    partial = layers.Dense(1)(c_p_f)
    fufi = layers.Dense(1)(c_p_f)
    clone = layers.Activation('sigmoid', dtype='float32', name='clone')(clone)   
    partial = layers.Activation('sigmoid', dtype='float32', name='partial')(partial)
    fufi = layers.Activation('sigmoid', dtype='float32', name='fufi')(fufi)
    if crypt_class:
       crypt = layers.Dense(1)(c_p_f)
       crypt = layers.Activation('sigmoid', dtype='float32', name='crclass')(crypt)
       just_trnsf = Model(inputs=[fmap, Input_bbox], outputs=[clone, partial, fufi, crypt], name = "cpf")
    else:
       just_trnsf = Model(inputs=[fmap, Input_bbox], outputs=[clone, partial, fufi], name = "cpf")
    return just_trnsf

def unet_sep(param, input_shape=(1024, 1024, 3), roi_pool_size = [10, 10], chan_num=3, 
             weight_ccpf = [1,1,1,1,1], projection_dim = 100, transformer_layers = 4, 
             num_heads = 4, is_comp = True):
    num_bbox = param["num_bbox"] 
    
    input_img = layers.Input(shape=input_shape)
    input_bbox = layers.Input(shape=(num_bbox, 4))
    
    ## get unet stem model
    just_unet  = strde_unet_xcept_gn_shallow()
#    just_unet  = strde_unet_xcept_gn_deep()
#    just_unet = strde_sepconv_unet_xcept_gn_shallow()
#    just_unet = strde_sepconv_unet_xcept_gn_deep()
#    just_unet = res_unet()
    
    ## and classifier model
    just_trnsf = classify_branch(num_bbox = num_bbox, crypt_class = param['crypt_class'])
    
    ## crate instances of models
    inst_cr, inst_fm = just_unet(input_img)
    
    if param['crypt_class']:
       inst_cl, inst_pa, inst_fu, inst_crcls = just_trnsf([inst_fm, input_bbox])

       ## combine into final model
       final_model = Model(inputs=[input_img, input_bbox], outputs=[inst_cr, inst_cl, inst_pa, inst_fu, inst_crcls])

       losses = {'crypt': "binary_crossentropy",
                 'cpf': "binary_crossentropy",
                 'cpf_1': "binary_crossentropy",
                 'cpf_2': "binary_crossentropy",
                 'cpf_3': "binary_crossentropy"
       }
       lossWeights = {'crypt': weight_ccpf[0], 
                      'cpf'  : weight_ccpf[1],
                      'cpf_1': weight_ccpf[2],
                      'cpf_2': weight_ccpf[3],
                      'cpf_3': weight_ccpf[4]
       }
       metrics_use = {'crypt': metrics.Accuracy(), 
                      'cpf'  : [metrics.TruePositives(), 
                                metrics.FalseNegatives(), 
                                metrics.FalsePositives(), 
                                metrics.TrueNegatives()],
                      'cpf_1': [metrics.TruePositives(), 
                                metrics.FalseNegatives(), 
                                metrics.FalsePositives(), 
                                metrics.TrueNegatives()],
                      'cpf_2': [metrics.TruePositives(), 
                                metrics.FalseNegatives(), 
                                metrics.FalsePositives(), 
                                metrics.TrueNegatives()],
                      'cpf_3': [metrics.TruePositives(), 
                                metrics.FalseNegatives(), 
                                metrics.FalsePositives(), 
                                metrics.TrueNegatives()]
       }    
    else:
       inst_cl, inst_pa, inst_fu = just_trnsf([inst_fm, input_bbox])

       ## combine into final model
       final_model = Model(inputs=[input_img, input_bbox], outputs=[inst_cr, inst_cl, inst_pa, inst_fu])

       losses = {'crypt': "binary_crossentropy",
                 'cpf': "binary_crossentropy",
                 'cpf_1': "binary_crossentropy",
                 'cpf_2': "binary_crossentropy"
       }
       lossWeights = {'crypt': weight_ccpf[0], 
                      'cpf': weight_ccpf[1],
                      'cpf_1': weight_ccpf[2],
                      'cpf_2': weight_ccpf[3]
       }
       metrics_use = {'crypt': metrics.Accuracy(), 
                      'cpf': [metrics.TruePositives(), 
                              metrics.FalseNegatives(), 
                              metrics.FalsePositives(), 
                              metrics.TrueNegatives()],
                      'cpf_1': [metrics.TruePositives(), 
                                metrics.FalseNegatives(), 
                                metrics.FalsePositives(), 
                                metrics.TrueNegatives()],
                      'cpf_2': [metrics.TruePositives(), 
                                metrics.FalseNegatives(), 
                                metrics.FalsePositives(), 
                                metrics.TrueNegatives()]
       }
    if is_comp: # compile
       final_model.compile(optimizer=RMSprop(lr=0.0001), 
                           loss = losses, loss_weights = lossWeights,
                           metrics = metrics_use)
    return final_model, just_trnsf, just_unet



