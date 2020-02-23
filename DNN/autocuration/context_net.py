from keras.models       import Model, Sequential
from keras.layers       import Input, concatenate, Conv2D, Conv1D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D
from keras.layers       import Softmax, Flatten, Dense, Reshape, Dropout, Lambda, Multiply, Add, Dot, Concatenate, AveragePooling2D, Lambda, Subtract, Dot
#from keras.layers       import Conv3D, MaxPooling3D
from keras.regularizers import l2
from keras.optimizers   import RMSprop, Adam
from keras.losses import binary_crossentropy
from DNN.losses         import *
from DNN.autocuration.layers import ROIPoolingLayer, LayerNormalization
from math import sqrt
import keras.backend as K

#def pick_roi_channel(tensor, ri):
#   return tensor[:,ri,:,:,:]

#def divide_tensor(tensor, d):
#   return tensor * (1./d)

def grab_roi(var):
   shape_y = K.int_shape(var[0])[1] # batch dimension is implict, hence start at 0?
   shape_x = K.int_shape(var[0])[2]
   lwr_x = K.cast(shape_x * var[1][0,0], 'int32')
   lwr_y = K.cast(shape_y * var[1][0,1], 'int32')
   upr_x = K.cast(shape_x * var[1][0,2], 'int32')
   upr_y = K.cast(shape_y * var[1][0,3], 'int32')
   out1 = var[0][lwr_y:upr_y, lwr_x:upr_x, :]
   return out1

def remove_roi_channel(rois):
   return K.squeeze(rois, 1)

def make_a(tensors, sqrtd):
#   Q_exp = K.expand_dims(tensors[0], axis=1)
#   Q_exp = K.expand_dims(tensors[0], axis=2)
   return (tensors[0] * tensors[1]) * (1./sqrtd)
   
def softmax2D_eachchan(a):
   D = K.reshape(a, (K.int_shape(a)[1]*K.int_shape(a)[2],
                     K.int_shape(a)[3]))
   D = K.softmax(a, axis=1)
   return K.reshape(D, (K.int_shape(a)[1], K.int_shape(a)[2], K.int_shape(a)[3]))

def make_A(tensors):
   Axy = tensors[0]*tensors[1]
   Ax = K.sum(Axy, axis=2, keepdims=True)
   return K.sum(Ax, axis=1, keepdims=True)

def test(input_shape=(512, 512, 3), queryroi_shape=(4,), poolsize=(7, 7), chan_num=3):
   ## feed rois as (y, x, h, w)
   inputs = Input(shape=input_shape)
   query  = Input(shape=queryroi_shape)
   d_mod = 128
   sqrt_d_mod = sqrt(128)
   
   # create initial feature map
   down0 = Conv2D(12, (3, 3), padding='same')(inputs)
   down0 = BatchNormalization()(down0)
   down0 = Activation('relu')(down0)
   down0 = Conv2D(12, (3, 3), padding='same')(down0)
   down0 = BatchNormalization()(down0)
   down0 = Activation('relu')(down0)
   down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

   down1 = Conv2D(24, (3, 3), padding='same')(down0_pool)
   down1 = BatchNormalization()(down1)
   down1 = Activation('relu')(down1)
   down1 = Conv2D(24, (3, 3), padding='same')(down1)
   down1 = BatchNormalization()(down1)
   feature_maps = Activation('relu')(down1)  
   
   # extract query crypt features and remove the n_rois dimension
   roi_pool = ROIPoolingLayer(poolsize[1], poolsize[0])([feature_maps, query])
#   roi_pool = Lambda(remove_roi_channel)(roi_pool)

   # process query
#   feat_1aq = Conv2D(24, (1,1))(roi_pool)
#   Qpre1 = Reshape((1, 1, K.int_shape(feat_1aq)[1]*K.int_shape(feat_1aq)[2]*K.int_shape(feat_1aq)[3]))(feat_1aq)
#   Q = Conv2D(d_mod, (1,1))(Qpre1)
   
#   ## Tx block 1, layer 1
#   # get key / feature projections of original feature map
#   keys1   = Conv2D(d_mod, (1,1))(feature_maps)
#   values1 = Conv2D(d_mod, (1,1))(feature_maps)
#   # scaled dot-product attention
#   a1 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q, keys1])
#   a1_sf = Reshape((K.int_shape(a1)[1]*K.int_shape(a1)[2], K.int_shape(a1)[3]))(a1)
#   a1_sf = Softmax(axis=1)(a1_sf)
#   a1_sf = Reshape((K.int_shape(a1)[1], K.int_shape(a1)[2], K.int_shape(a1)[3]))(a1_sf)
#   A1 = Lambda(make_A)([a1_sf, values1])
#   # combine with query
#   A1 = Dropout(0.3)(A1)
#   Qp1 = Add()([Q, A1])
##   Qp1 = LayerNormalization()(Qp1)
#   # FFN
#   Qp1_ff = Conv2D(150, (1,1))(Qp1)
#   Qp1_ff = Activation('relu')(Qp1_ff)
#   Qp1_ff = Conv2D(d_mod, (1,1))(Qp1_ff)
#   Qp1_ff = Dropout(0.3)(Qp1_ff)
#   Qpp1 = Add()([Qp1, Qp1_ff])
##   Qpp1 = LayerNormalization()(Qpp1)
#   
#   ## Tx block 1, layer 2
#   # get key / feature projections of original feature map
#   keys2   = Conv2D(d_mod, (1,1))(feature_maps)
#   values2 = Conv2D(d_mod, (1,1))(feature_maps)
#   # scaled dot-product attention
#   a2 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q, keys2])
#   a2_sf = Reshape((K.int_shape(a2)[1]*K.int_shape(a2)[2], K.int_shape(a2)[3]))(a2)
#   a2_sf = Softmax(axis=1)(a2_sf)
#   a2_sf = Reshape((K.int_shape(a2)[1], K.int_shape(a2)[2], K.int_shape(a2)[3]))(a2_sf)
#   A2 = Lambda(make_A)([a2_sf, values2])
#   # combine with query
#   A2 = Dropout(0.3)(A2)
#   Qp2 = Add()([Q, A2])
##   Qp2 = LayerNormalization()(Qp2)
#   # FFN
#   Qp2_ff = Conv2D(150, (1,1))(Qp2)
#   Qp2_ff = Activation('relu')(Qp2_ff)
#   Qp2_ff = Conv2D(d_mod, (1,1))(Qp2_ff)
#   Qp2_ff = Dropout(0.3)(Qp2_ff)
#   Qpp2 = Add()([Qp2, Qp2_ff])
##   Qpp2 = LayerNormalization()(Qpp2)
#   
#   ## concatenate query outputs
#   Q2 = concatenate([Qpp1, Qpp2], axis=chan_num)

   ## concatenate query outputs
#   Q5 = Conv2D(d_mod, (1,1))(Q)
   Q5 = Flatten()(roi_pool)
      
   # predict on query vector
   prediction = Dense(1, activation='sigmoid')(Q5)
   
   encoder = Model(inputs=[inputs, query],
                   outputs=prediction)   
   encoder.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder

def rcnet3(input_shape=(512, 512, 3), queryroi_shape=(1,4), poolsize=(7, 7), chan_num=3):
   ## feed rois as (y, x, h, w)
   inputs = Input(shape=input_shape)
   query  = Input(shape=queryroi_shape)
   d_mod = 128
   sqrt_d_mod = sqrt(128)
   
   # create initial feature map
   down0 = Conv2D(52, (3, 3), padding='same')(inputs)
   down0 = BatchNormalization()(down0)
   down0 = Activation('relu')(down0)
   down0 = Conv2D(52, (3, 3), padding='same')(down0)
   down0 = BatchNormalization()(down0)
   down0 = Activation('relu')(down0)
   down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

   down1 = Conv2D(100, (3, 3), padding='same')(down0_pool)
   down1 = BatchNormalization()(down1)
   down1 = Activation('relu')(down1)
   down1 = Conv2D(100, (3, 3), padding='same')(down1)
   down1 = BatchNormalization()(down1)
   feature_maps = Activation('relu')(down1)  
   
   # extract query crypt features and remove the n_rois dimension
   roi_pool = ROIPoolingLayer(poolsize[1], poolsize[0])([feature_maps, query])
   roi_pool = Lambda(remove_roi_channel)(roi_pool)

   # process query
   feat_1aq = Conv2D(100, (1,1), padding='same', activation='relu')(roi_pool)
   Qpre1 = Reshape((1, 1, K.int_shape(feat_1aq)[1]*K.int_shape(feat_1aq)[2]*K.int_shape(feat_1aq)[3]))(feat_1aq)
   Q = Conv2D(d_mod, (1,1))(Qpre1)
   
   ## Tx block 1, layer 1
   # get key / feature projections of original feature map
   keys1   = Conv2D(d_mod, (1,1))(feature_maps)
   values1 = Conv2D(d_mod, (1,1))(feature_maps)
   # scaled dot-product attention
   a1 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q, keys1])
   a1_sf = Reshape((K.int_shape(a1)[1]*K.int_shape(a1)[2], K.int_shape(a1)[3]))(a1)
   a1_sf = Softmax(axis=1)(a1_sf)
   a1_sf = Reshape((K.int_shape(a1)[1], K.int_shape(a1)[2], K.int_shape(a1)[3]))(a1_sf)
   A1 = Lambda(make_A)([a1_sf, values1])
   # combine with query
   A1 = Dropout(0.3)(A1)
   Qp1 = Add()([Q, A1])
#   Qp1 = LayerNormalization()(Qp1)
   # FFN
   Qp1_ff = Conv2D(256, (1,1))(Qp1)
   Qp1_ff = Activation('relu')(Qp1_ff)
   Qp1_ff = Conv2D(d_mod, (1,1))(Qp1_ff)
   Qp1_ff = Dropout(0.3)(Qp1_ff)
   Qpp1 = Add()([Qp1, Qp1_ff])
#   Qpp1 = LayerNormalization()(Qpp1)
   
   ## Tx block 1, layer 2
   # get key / feature projections of original feature map
   keys2   = Conv2D(d_mod, (1,1))(feature_maps)
   values2 = Conv2D(d_mod, (1,1))(feature_maps)
   # scaled dot-product attention
   a2 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q, keys2])
   a2_sf = Reshape((K.int_shape(a2)[1]*K.int_shape(a2)[2], K.int_shape(a2)[3]))(a2)
   a2_sf = Softmax(axis=1)(a2_sf)
   a2_sf = Reshape((K.int_shape(a2)[1], K.int_shape(a2)[2], K.int_shape(a2)[3]))(a2_sf)
   A2 = Lambda(make_A)([a2_sf, values2])
   # combine with query
   A2 = Dropout(0.3)(A2)
   Qp2 = Add()([Q, A2])
#   Qp2 = LayerNormalization()(Qp2)
   # FFN
   Qp2_ff = Conv2D(256, (1,1))(Qp2)
   Qp2_ff = Activation('relu')(Qp2_ff)
   Qp2_ff = Conv2D(d_mod, (1,1))(Qp2_ff)
   Qp2_ff = Dropout(0.3)(Qp2_ff)
   Qpp2 = Add()([Qp2, Qp2_ff])
#   Qpp2 = LayerNormalization()(Qpp2)
   
   ## concatenate query outputs
   Q2 = concatenate([Qpp1, Qpp2], axis=chan_num)
   # process query
   feat_2aq = Conv2D(100, (1,1), padding='same', activation='relu')(Q2)
   Qpre2 = Reshape((1, 1, K.int_shape(feat_2aq)[1]*K.int_shape(feat_2aq)[2]*K.int_shape(feat_2aq)[3]))(feat_2aq)
   Q2 = Conv2D(d_mod, (1,1))(Qpre2)
   
   ## Tx block 2, layer 1
   # get key / feature projections of original feature map
   keys3   = Conv2D(d_mod, (1,1))(feature_maps)
   values3 = Conv2D(d_mod, (1,1))(feature_maps)
   # scaled dot-product attention
   a3 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q2, keys3])
   a3_sf = Reshape((K.int_shape(a3)[1]*K.int_shape(a3)[2], K.int_shape(a3)[3]))(a3)
   a3_sf = Softmax(axis=1)(a3_sf)
   a3_sf = Reshape((K.int_shape(a3)[1], K.int_shape(a3)[2], K.int_shape(a3)[3]))(a3_sf)
   A3 = Lambda(make_A)([a3_sf, values3])
   # combine with query
   A3 = Dropout(0.3)(A3)
   Qp3 = Add()([Q2, A3])
#   Qp3 = LayerNormalization()(Qp3)
   # FFN
   Qp3_ff = Conv2D(256, (1,1))(Qp3)
   Qp3_ff = Activation('relu')(Qp3_ff)
   Qp3_ff = Conv2D(d_mod, (1,1))(Qp3_ff)
   Qp3_ff = Dropout(0.3)(Qp3_ff)
   Qpp3 = Add()([Qp3, Qp3_ff])
#   Qpp3 = LayerNormalization()(Qpp3)
   
   ## Tx block 2, layer 2
   # get key / feature projections of original feature map
   keys4   = Conv2D(d_mod, (1,1))(feature_maps)
   values4 = Conv2D(d_mod, (1,1))(feature_maps)
   # scaled dot-product attention
   a4 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q2, keys4])
   a4_sf = Reshape((K.int_shape(a4)[1]*K.int_shape(a4)[2], K.int_shape(a4)[3]))(a4)
   a4_sf = Softmax(axis=1)(a4_sf)
   a4_sf = Reshape((K.int_shape(a4)[1], K.int_shape(a4)[2], K.int_shape(a4)[3]))(a4_sf)
   A4 = Lambda(make_A)([a4_sf, values4])
   # combine with query
   A4 = Dropout(0.3)(A4)
   Qp4 = Add()([Q2, A4])
#   Qp4 = LayerNormalization()(Qp4)
   # FFN
   Qp4_ff = Conv2D(256, (1,1))(Qp4)
   Qp4_ff = Activation('relu')(Qp4_ff)
   Qp4_ff = Conv2D(d_mod, (1,1))(Qp4_ff)
   Qp4_ff = Dropout(0.3)(Qp4_ff)
   Qpp4 = Add()([Qp4, Qp4_ff])
#   Qpp4 = LayerNormalization()(Qpp4)

   ## concatenate query outputs
   Q3 = concatenate([Qpp3, Qpp4], axis=chan_num)
#   # process query
#   feat_3aq = Conv2D(100, (1,1), padding='same', activation='relu')(Q3)
#   Qpre3 = Reshape((1, 1, K.int_shape(feat_3aq)[1]*K.int_shape(feat_3aq)[2]*K.int_shape(feat_3aq)[3]))(feat_3aq)
#   Q3 = Conv2D(d_mod, (1,1))(Qpre3)

#   ## Tx block 3, layer 1
#   # get key / feature projections of original feature map
#   keys5   = Conv2D(d_mod, (1,1))(feature_maps)
#   values5 = Conv2D(d_mod, (1,1))(feature_maps)
#   # scaled dot-product attention
#   a5 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q3, keys5])
#   a5_sf = Reshape((K.int_shape(a5)[1]*K.int_shape(a5)[2], K.int_shape(a5)[3]))(a5)
#   a5_sf = Softmax(axis=1)(a5_sf)
#   a5_sf = Reshape((K.int_shape(a5)[1], K.int_shape(a5)[2], K.int_shape(a5)[3]))(a5_sf)
#   A5 = Lambda(make_A)([a5_sf, values5])
#   # combine with query
#   A5 = Dropout(0.3)(A5)
#   Qp5 = Add()([Q3, A5])
##   Qp5 = LayerNormalization()(Qp5)
#   # FFN
#   Qp5_ff = Conv2D(256, (1,1))(Qp5)
#   Qp5_ff = Activation('relu')(Qp5_ff)
#   Qp5_ff = Conv2D(d_mod, (1,1))(Qp5_ff)
#   Qp5_ff = Dropout(0.3)(Qp5_ff)
#   Qpp5 = Add()([Qp5, Qp5_ff])
##   Qpp5 = LayerNormalization()(Qpp5)
#   
#   ## Tx block 3, layer 2
#   # get key / feature projections of original feature map
#   keys6   = Conv2D(d_mod, (1,1))(feature_maps)
#   values6 = Conv2D(d_mod, (1,1))(feature_maps)
#   # scaled dot-product attention
#   a6 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q3, keys6])
#   a6_sf = Reshape((K.int_shape(a6)[1]*K.int_shape(a6)[2], K.int_shape(a6)[3]))(a6)
#   a6_sf = Softmax(axis=1)(a6_sf)
#   a6_sf = Reshape((K.int_shape(a6)[1], K.int_shape(a6)[2], K.int_shape(a6)[3]))(a6_sf)
#   A6 = Lambda(make_A)([a6_sf, values6])
#   # combine with query
#   A6 = Dropout(0.3)(A6)
#   Qp6 = Add()([Q3, A6])
##   Qp6 = LayerNormalization()(Qp6)
#   # FFN
#   Qp6_ff = Conv2D(256, (1,1))(Qp6)
#   Qp6_ff = Activation('relu')(Qp6_ff)
#   Qp6_ff = Conv2D(d_mod, (1,1))(Qp6_ff)
#   Qp6_ff = Dropout(0.3)(Qp6_ff)
#   Qpp6 = Add()([Qp6, Qp6_ff])
##   Qpp6 = LayerNormalization()(Qpp6)

#   ## concatenate query outputs
#   Q4 = concatenate([Qpp5, Qpp6], axis=chan_num)
#   Q5 = Conv2D(d_mod, (1,1))(Q4)
#   Q5 = Flatten()(Q5)

   ## concatenate query outputs
   Q5 = Conv2D(d_mod, (1,1))(Q3)
   Q5 = Flatten()(Q5)
      
   # predict on query vector
   prediction = Dense(1, activation='sigmoid')(Q5)
   
   encoder = Model(inputs=[inputs, query],
                   outputs=prediction)   
   encoder.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder


#def rcnet2(input_shape=(512, 512, 3), queryroi_shape=(1,4), poolsize=(7, 7), chan_num=3):
#   ## feed rois as (y, x, h, w)
#   inputs = Input(shape=input_shape)
#   query  = Input(shape=queryroi_shape)
#   
#   # create initial feature map
#   down0 = Conv2D(64, (3, 3), padding='same')(inputs)
#   down0 = BatchNormalization()(down0)
#   down0 = Activation('relu')(down0)
#   down0 = Conv2D(64, (3, 3), padding='same')(down0)
#   down0 = BatchNormalization()(down0)
#   down0 = Activation('relu')(down0)
#   down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

#   down1 = Conv2D(128, (3, 3), padding='same')(down0_pool)
#   down1 = BatchNormalization()(down1)
#   down1 = Activation('relu')(down1)
#   down1 = Conv2D(128, (3, 3), padding='same')(down1)
#   down1 = BatchNormalization()(down1)
#   feature_maps = Activation('relu')(down1)  
#   
#   # extract query crypt features and remove the n_rois dimension
#   roi_pool = ROIPoolingLayer(poolsize[1], poolsize[0])([feature_maps, query])
#   ch = 0
#   roi_pool = Lambda(pick_roi_channel, arguments={'ri':ch})(roi_pool)

#   # process query crypt using cutdown inception block
#   feat_1aq = Conv2D(512, (1,1), padding='same', activation='relu')(roi_pool)
#   feat_3aq = MaxPooling2D((3,3), strides=(1,1), padding='same')(roi_pool)
#   feat_3aq = Conv2D(512, (1,1), padding='same', activation='relu')(feat_3aq)
#   query_feature_block = concatenate([feat_1aq, feat_3aq], axis = chan_num)
#   d_mod = 128
#   query_feature_block = Conv2D(d_mod, (1,1))(query_feature_block)
#   
#   # get key / feature projections of original feature map
#   keys   = Conv2D(d_mod, (1,1))(feature_maps)
#   values = Conv2D(d_mod, (1,1))(feature_maps)
#   
#   # scaled dot-product attention
#   a_xy = Dot(axes=chan_num)([query_feature_block, keys]) # surely these need to have an axis that matches?
#   a_xy = Lambda(divide_tensor, arguments={'d':sqrt(d_mod)})(roi_pool)
#   a_xy = Activation('softmax', axis=chan_num)(a_xy)
#   Att_xy = Dot(axes=chan_num)([a_xy, values])
#   
#   # feedforward
#   Att_ff = Conv2D(2048, (1,1))(Att_xy)
#   Att_ff = Activation('relu')(Att_ff)
#   Att_out = Conv2D(512, (1,1))(Att_ff)
#   Att_out = BatchNormalization()(Att_out)
#   
#      
#   # predict on query vector
#   prediction = Dense(1, activation='sigmoid')(Att_out)
#   
#   encoder = Model(inputs=[inputs, rois, query],
#                   outputs=prediction)   
#   encoder.compile(optimizer=RMSprop(lr=0.0001),
#                  loss='binary_crossentropy', 
#                  metrics=['accuracy'])
#   return encoder

#def rcnet(input_shape=(512, 512, 3), queryroi_shape=(4,), rois_shape=(20, 4), poolsize=(7, 7), chan_num=3):
#   ## feed rois as (y, x, h, w)
#   inputs = Input(shape=input_shape)
#   rois   = Input(shape=rois_shape)
#   query  = Input(shape=queryroi_shape)
#   
#   # create initial feature map
#   down0 = Conv2D(64, (3, 3), padding='same')(inputs)
#   down0 = BatchNormalization()(down0)
#   down0 = Activation('relu')(down0)
#   down0 = Conv2D(64, (3, 3), padding='same')(down0)
#   down0 = BatchNormalization()(down0)
#   down0 = Activation('relu')(down0)
#   down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

#   down1 = Conv2D(128, (3, 3), padding='same')(down0_pool)
#   down1 = BatchNormalization()(down1)
#   down1 = Activation('relu')(down1)
#   down1 = Conv2D(128, (3, 3), padding='same')(down1)
#   down1 = BatchNormalization()(down1)
#   feature_maps = Activation('relu')(down1)
#   
#   # extract query crypt features
#   query_feature_map = Lambda(grab_roi)([feature_maps, query])

#   # process query crypt using cutdown inception block
#   feat_1aq = Conv2D(24, (1,1), padding='same', activation='relu')(query_feature_map)
#   feat_3aq = MaxPooling2D((3,3), strides=(1,1), padding='same')(query_feature_map)
#   feat_3aq = Conv2D(24, (1,1), padding='same', activation='relu')(feat_3aq)
#   query_feature_vec = concatenate([feat_1aq, feat_3aq], axis = chan_num)
#   ch_q = 128
#   query_feature_vec = Dense(ch_q)(query_feature_vec)
#   
#   # pool rois of the feature map 
#   roi_pool = ROIPoolingLayer(poolsize[1], poolsize[0])([feature_maps, rois]) # shape (batch_size, n_rois, self.pooled_width, self.pooled_height, n_channels)
#   
#   for ri in range(rois_shape[0]):   
#      # process roi using cut-down inception block
#      this_roi = Lambda(pick_roi_channel, arguments={"ri":ri})(roi_pool)
#      feat_1a = Conv2D(24, (1,1), padding='same', activation='relu')(this_roi)
#      feat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(this_roi)
#      feat_3a = Conv2D(24, (1,1), padding='same', activation='relu')(feat_3a)
#      roi_feature_vec = concatenate([feat_1a, feat_3a], axis = chan_num)
#      # squeeze excite and update query
#      xb = GlobalAveragePooling2D()(roi_feature_vec)      
#      xb = Dense(ch_q//16, activation='relu')(xb)
#      xb = Dense(ch_q, activation='sigmoid')(xb)
#      query_feature_vec = Multiply()([query_feature_vec, xb])
#      
#   # predict on query vector
#   prediction = Dense(1, activation='sigmoid')(query_feature_vec)
#   
#   encoder = Model(inputs=[inputs, rois, query],
#                   outputs=prediction)   
#   encoder.compile(optimizer=RMSprop(lr=0.0001),
#                  loss='binary_crossentropy', 
#                  metrics=['accuracy'])
#   return encoder

      
