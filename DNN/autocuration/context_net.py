import tensorflow as tf
from tensorflow.keras.models       import Model, Sequential
from tensorflow.keras.layers       import Input, concatenate, Conv2D, Conv1D, MaxPooling2D, Activation,\
                                          UpSampling2D, BatchNormalization, GlobalAveragePooling2D,\
                                          SeparableConv2D, Softmax, Flatten, Dense, Reshape, Dropout,\
                                          Lambda, Multiply, Add, Dot, Concatenate, AveragePooling2D,\
                                          Subtract, LayerNormalization
#from tensorflow.keras.layers       import Conv3D, MaxPooling3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers   import RMSprop, Adam
from tensorflow.keras.losses import binary_crossentropy
from DNN.losses         import *
#from DNN.autocuration.layers import ROIPoolingLayer#, LayerNormalization
from DNN.autocuration.layers import AttentionROI #AttentionAugmentation2D
from math import sqrt
import tensorflow.keras.backend as K
#from DNN.autocuration.video_action_transformer_net_edit import *

def embed_distance_batch(x):
   def curried_embed(x): 
      return embed_distance(x)

   embedded_dists = tf.map_fn(curried_embed, x, dtype=tf.float32)
   return embedded_dists

def embed_distance(tensor):
   H, W, D = K.int_shape(tensor)
   mi, mj = tf.meshgrid(range(H), range(W), indexing='ij')
   mi = tf.cast(K.reshape(mi, (H,W,1)), tf.float32)
   mj = tf.cast(K.reshape(mj, (H,W,1)), tf.float32)
   Hc = K.constant(float(H//2))
   Wc = K.constant(float(W//2))
   distance_embedding = tf.sqrt(tf.add(tf.square(tf.subtract(mi,Hc)), tf.square(tf.subtract(mj, Wc))))
   tensor = concatenate([tensor, distance_embedding], axis=-1)
   return tensor

def att_roi_net2(input_shape1=(512,512,3), input_shape2=(384,384,3), 
                 d_model=128, depth_k=8, depth_v=12, num_heads=2, dff=256, dropout_rate=0.3):
   inputs1 = Input(shape=input_shape1)
   inputs2 = Input(shape=input_shape2)
   
   ## create zoomed-out tile feature maps
   cc1 = Conv2D(24, (3,3), padding='same')(inputs1)   
   cc1 = Activation('relu')(cc1)
   cc1 = BatchNormalization()(cc1)
   cc1 = Conv2D(24, (3,3), padding='same')(cc1)   
   cc1 = Activation('relu')(cc1)
   cc1 = BatchNormalization()(cc1)
   cc1_pool = MaxPooling2D((2, 2), strides=(2, 2))(cc1)
   
   cc2 = Conv2D(48, (3,3), padding='same')(cc1_pool)
   cc2 = Activation('relu')(cc2)
   cc2 = BatchNormalization()(cc2)   
   cc2 = Conv2D(48, (3,3), padding='same')(cc2)   
   cc2 = Activation('relu')(cc2)
   feature_maps = BatchNormalization()(cc2)
   ## add positional embedding: distance from centre
   feature_maps = Lambda(embed_distance_batch)(feature_maps)

   ## create zoomed-in crypt feature maps
   cc3 = Conv2D(24, (3,3), padding='same')(inputs2)
   cc3 = Activation('relu')(cc3)
   cc3 = BatchNormalization()(cc3)   
   cc3 = Conv2D(24, (3,3), padding='same')(cc3)
   cc3 = Activation('relu')(cc3)
   cc3 = BatchNormalization()(cc3)   
   cc3_pool = MaxPooling2D((2, 2), strides=(2, 2))(cc3)
   
   cc4 = Conv2D(48, (3,3), padding='same')(cc3_pool)
   cc4 = Activation('relu')(cc4)
   cc4 = BatchNormalization()(cc4)   
   cc4 = Conv2D(48, (3,3), padding='same')(cc4)
   cc4 = Activation('relu')(cc4)
   cc4 = BatchNormalization()(cc4)
   cc4_pool = MaxPooling2D((2, 2), strides=(2, 2))(cc4)
   
   cc5 = Conv2D(64, (3,3), padding='same')(cc4_pool)
   cc5 = Activation('relu')(cc5)
   cc5 = BatchNormalization()(cc5)   
   cc5 = Conv2D(64, (3,3), padding='same')(cc5)
   cc5 = Activation('relu')(cc5)
   cc5 = BatchNormalization()(cc5)
   cc5_pool = MaxPooling2D((2, 2), strides=(2, 2))(cc5)

   cc6 = Conv2D(80, (3,3), padding='same')(cc5_pool)
   cc6 = Activation('relu')(cc6)
   cc6 = BatchNormalization()(cc6)
   cc6 = Conv2D(80, (3,3), padding='same')(cc6)
   cc6 = Activation('relu')(cc6)
   cc6 = BatchNormalization()(cc6)
   feat_q = MaxPooling2D((2, 2), strides=(2, 2))(cc6)
   
   # squeeze excite and update query
   ch_q = feat_q.shape[3]
   xb = GlobalAveragePooling2D()(feat_q)      
   xb = Dense(ch_q//16, activation='relu')(xb)
   xb = Dense(ch_q, activation='sigmoid')(xb)
   feat_q = Multiply()([feat_q, xb])
   feat_q = Conv2D(16, (1,1), padding='same')(feat_q)
   feat_q = Activation('relu')(feat_q)
   feat_q = BatchNormalization()(feat_q)
   
   # process query
   Qpre = Reshape((1, 1, K.int_shape(feat_q)[1]*K.int_shape(feat_q)[2]*K.int_shape(feat_q)[3]))(feat_q)
   Q = Conv2D(d_model, (1,1))(Qpre)
   
   ## use crypt feature maps as query and zoomed_out feature maps as keys, values
   Q1 = AttentionROI(d_model, depth_k, dff, num_heads, dropout_rate, relative=False)([Q, feature_maps])
   Q2 = AttentionROI(d_model, depth_k, dff, num_heads, dropout_rate, relative=False)([Q1, feature_maps])
   Q3 = AttentionROI(d_model, depth_k, dff, num_heads, dropout_rate, relative=False)([Q2, feature_maps])
   # add processing of Q between each attention layer above?

   dense_Q = Dense(d_model)(Q3)
   dense_Q = Flatten()(dense_Q)
   prediction = Dense(1, activation='sigmoid')(dense_Q)

   clone_pred = Model(inputs=[inputs1, inputs2],
                      outputs=prediction)   
   clone_pred.compile(optimizer=RMSprop(lr=0.0001), #, clipnorm=1.
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
   return clone_pred

def att_roi_net(input_shape1=(512,512,3), input_shape2=(384,384,3), query_shape=(4,), poolsize=(25,25),
                d_model=128, depth_k=8, depth_v=12, num_heads=2, dff=256, dropout_rate=0.3):
   inputs1 = Input(shape=input_shape1)
   inputs2 = Input(shape=input_shape2)
   query = Input(shape=query_shape)
   
   ## create zoomed-out tile feature maps
   cc1 = AttentionAugmentation2D(24, 3, depth_k, depth_v, num_heads, relative=False)(inputs1)
   cc1 = BatchNormalization()(cc1)
   cc1 = Activation('relu')(cc1)
   feature_maps_cr = Activation('relu')(cc1)
   cc1 = AttentionAugmentation2D(24, 3, depth_k, depth_v, num_heads, relative=False)(cc1)
   cc1 = BatchNormalization()(cc1)
   cc1 = Activation('relu')(cc1)
   cc1_pool = MaxPooling2D((2, 2), strides=(2, 2))(cc1)
   
   cc2 = AttentionAugmentation2D(48, 3, depth_k, depth_v, num_heads, relative=False)(cc1_pool)
   cc2 = BatchNormalization()(cc2)
   cc2 = Activation('relu')(cc2)
   cc2 = AttentionAugmentation2D(48, 3, depth_k, depth_v, num_heads, relative=False)(cc2)
   cc2 = BatchNormalization()(cc2)
   feature_maps = Activation('relu')(cc2)

   ## create zoomed-in crypt feature maps
   cc3 = AttentionAugmentation2D(24, 3, depth_k, depth_v, num_heads, relative=False)(inputs2)
   cc3 = BatchNormalization()(cc3)
   cc3 = Activation('relu')(cc3)
   cc3 = AttentionAugmentation2D(24, 3, depth_k, depth_v, num_heads, relative=False)(cc3)
   cc3 = BatchNormalization()(cc3)
   cc3 = Activation('relu')(cc3)
   cc3_pool = MaxPooling2D((2, 2), strides=(2, 2))(cc3)
   
   cc4 = AttentionAugmentation2D(48, 3, depth_k, depth_v, num_heads, relative=False)(cc3_pool)
   cc4 = BatchNormalization()(cc4)
   cc4 = Activation('relu')(cc4)
   cc4 = AttentionAugmentation2D(48, 3, depth_k, depth_v, num_heads, relative=False)(cc4)
   cc4 = BatchNormalization()(cc4)
   cc4_pool = MaxPooling2D((2, 2), strides=(2, 2))(cc4)
   
   cc5 = AttentionAugmentation2D(64, 3, depth_k, depth_v, num_heads, relative=False)(cc4_pool)
   cc5 = BatchNormalization()(cc5)
   cc5 = Activation('relu')(cc5)
   cc5 = AttentionAugmentation2D(64, 3, depth_k, depth_v, num_heads, relative=False)(cc5)
   cc5 = BatchNormalization()(cc5)
   feature_maps_cr = Activation('relu')(cc5)
 
   # extract query crypt features
   roi_pool = Lambda(pool_one_roi, arguments={'pooled_width':poolsize[0], 'pooled_height':poolsize[1]})([feature_maps_cr, query])

   # process query
   feat_q = Conv2D(8, (1,1))(roi_pool)
   Qpre = Reshape((1, 1, K.int_shape(feat_q)[1]*K.int_shape(feat_q)[2]*K.int_shape(feat_q)[3]))(feat_q)
   Q = Conv2D(d_model, (1,1))(Qpre)
   
   ## use crypt feature maps as query and zoomed_out feature maps as keys, values
#   Q1 = AttentionROI(d_model, 3, depth_k, dff, num_heads, dropout_rate, relative=False)([Q, feature_maps])
#   Q2 = AttentionROI(d_model, 3, depth_k, dff, num_heads, dropout_rate, relative=False)([Q1, feature_maps])
#   Q3 = AttentionROI(d_model, 3, depth_k, dff, num_heads, dropout_rate, relative=False)([Q2, feature_maps])
   # add processing of Q between each attention layer above?

   dense_Q = Dense(d_model)(Q) #Q3
   dense_Q = Flatten()(dense_Q)
   prediction = Dense(1, activation='sigmoid')(dense_Q)

   clone_pred = Model(inputs=[inputs1, inputs2, query],
                      outputs=prediction)   
   clone_pred.compile(optimizer=RMSprop(lr=0.0001),
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
   return clone_pred

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

def pool_one_roi(x, pooled_width, pooled_height):
   def curried_pool_rois(x): 
      return _pool_roi(x[0], x[1], pooled_width, pooled_height)

   pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)
   return pooled_areas

def _pool_roi(feature_map, roi, pooled_width, pooled_height):
   """ Applies ROI pooling to a single image and a single region of interest
   """
   # Compute the region of interest        
   feature_map_height = int(feature_map.shape[0])
   feature_map_width  = int(feature_map.shape[1])

   x_start = tf.stop_gradient(tf.cast(feature_map_width  * roi[0], 'int32'))
   y_start = tf.stop_gradient(tf.cast(feature_map_height * roi[1], 'int32'))
   x_end   = tf.stop_gradient(tf.cast(feature_map_width  * roi[2], 'int32'))
   y_end   = tf.stop_gradient(tf.cast(feature_map_height * roi[3], 'int32'))

   region = feature_map[y_start:y_end, x_start:x_end, :]

   # Divide the region into non overlapping areas
   region_height = y_end - y_start
   region_width  = x_end - x_start
   y_step = tf.cast( region_height / pooled_height, 'int32')
   x_step = tf.cast( region_width  / pooled_width , 'int32')

   # this just puts all remainder at edge, get better binning!
   areas = [[(
              i*y_step,
              j*x_step,
              (i+1)*y_step if i+1 < pooled_height else region_height, 
              (j+1)*x_step if j+1 < pooled_width else region_width
             ) 
             for j in range(pooled_width)] 
            for i in range(pooled_height)]

   # take the maximum of each area and stack the result
   def pool_area(x): 
      return K.max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])

   pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
   return pooled_features


def test(input_shape=(512, 512, 3), queryroi_shape=(4,), poolsize=(15, 15), chan_num=3):
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
#   down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

#   down1 = Conv2D(24, (3, 3), padding='same')(down0_pool)
#   down1 = BatchNormalization()(down1)
#   down1 = Activation('relu')(down1)
#   down1 = Conv2D(24, (3, 3), padding='same')(down1)
#   down1 = BatchNormalization()(down1)
#   feature_maps = Activation('relu')(down1)  
   feature_maps = down0
   
   # extract query crypt features and remove the n_rois dimension
   roi_pool = Lambda(pool_one_roi, arguments={'pooled_width':poolsize[0], 'pooled_height':poolsize[1]})([feature_maps, query])
#   roi_pool = ROIPoolingLayer(poolsize[1], poolsize[0])([down0, query])
#   roi_pool = Lambda(remove_roi_channel)(roi_pool)
#   roi_pool = down0

   # process query
   feat_1aq = Conv2D(24, (1,1))(roi_pool)
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
   Qp1 = LayerNormalization()(Qp1)
   # FFN
   Qp1_ff = Conv2D(150, (1,1))(Qp1)
   Qp1_ff = Activation('relu')(Qp1_ff)
   Qp1_ff = Conv2D(d_mod, (1,1))(Qp1_ff)
   Qp1_ff = Dropout(0.3)(Qp1_ff)
   Qpp1 = Add()([Qp1, Qp1_ff])
   Qpp1 = LayerNormalization()(Qpp1)
   
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
   Qp2 = LayerNormalization()(Qp2)
   # FFN
   Qp2_ff = Conv2D(150, (1,1))(Qp2)
   Qp2_ff = Activation('relu')(Qp2_ff)
   Qp2_ff = Conv2D(d_mod, (1,1))(Qp2_ff)
   Qp2_ff = Dropout(0.3)(Qp2_ff)
   Qpp2 = Add()([Qp2, Qp2_ff])
   Qpp2 = LayerNormalization()(Qpp2)
   
   ## concatenate query outputs
   Q2 = concatenate([Qpp1, Qpp2], axis=chan_num)

   ## concatenate query outputs
#   Q5 = Conv2D(d_mod, (1,1))(Q)
   Q5 = Flatten()(Q2)
      
   # predict on query vector
   prediction = Dense(1, activation='sigmoid')(Q5)
   
   encoder = Model(inputs=[inputs, query],
                   outputs=prediction)   
   encoder.compile(optimizer=RMSprop(lr=0.000001), #lr=0.0001
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder

def rcnet_reducedtest(input_shape=(512, 512, 3), queryroi_shape=(4), poolsize=(15, 15), chan_num=3):
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
   
   # extract query crypt features
   roi_pool = Lambda(pool_one_roi, arguments={'pooled_width':poolsize[0], 'pooled_height':poolsize[1]})([feature_maps, query])
   feat_1aq = Conv2D(24, (1,1))(roi_pool)
   Qpre1 = Reshape((1, 1, K.int_shape(feat_1aq)[1]*K.int_shape(feat_1aq)[2]*K.int_shape(feat_1aq)[3]))(feat_1aq)
   Q = Conv2D(d_mod, (1,1))(Qpre1)

   ## concatenate query outputs
   Q5 = Flatten()(Q)
     
   # predict on query vector
   prediction = Dense(1, activation='sigmoid')(Q5)
   
   encoder = Model(inputs=[inputs, query],
                   outputs=prediction)   
   encoder.compile(optimizer=RMSprop(lr=0.00005),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder

def rcnet3(input_shape=(512, 512, 3), queryroi_shape=(4), poolsize=(15, 15), chan_num=3):
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
   
   # extract query crypt features
   roi_pool = Lambda(pool_one_roi, arguments={'pooled_width':poolsize[0], 'pooled_height':poolsize[1]})([feature_maps, query])

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
   Qp1 = LayerNormalization()(Qp1)
   # FFN
   Qp1_ff = Conv2D(256, (1,1))(Qp1)
   Qp1_ff = Activation('relu')(Qp1_ff)
   Qp1_ff = Conv2D(d_mod, (1,1))(Qp1_ff)
   Qp1_ff = Dropout(0.3)(Qp1_ff)
   Qpp1 = Add()([Qp1, Qp1_ff])
   Qpp1 = LayerNormalization()(Qpp1)
   
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
   Qp2 = LayerNormalization()(Qp2)
   # FFN
   Qp2_ff = Conv2D(256, (1,1))(Qp2)
   Qp2_ff = Activation('relu')(Qp2_ff)
   Qp2_ff = Conv2D(d_mod, (1,1))(Qp2_ff)
   Qp2_ff = Dropout(0.3)(Qp2_ff)
   Qpp2 = Add()([Qp2, Qp2_ff])
   Qpp2 = LayerNormalization()(Qpp2)
   
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
   Qp3 = LayerNormalization()(Qp3)
   # FFN
   Qp3_ff = Conv2D(256, (1,1))(Qp3)
   Qp3_ff = Activation('relu')(Qp3_ff)
   Qp3_ff = Conv2D(d_mod, (1,1))(Qp3_ff)
   Qp3_ff = Dropout(0.3)(Qp3_ff)
   Qpp3 = Add()([Qp3, Qp3_ff])
   Qpp3 = LayerNormalization()(Qpp3)
   
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
   Qp4 = LayerNormalization()(Qp4)
   # FFN
   Qp4_ff = Conv2D(256, (1,1))(Qp4)
   Qp4_ff = Activation('relu')(Qp4_ff)
   Qp4_ff = Conv2D(d_mod, (1,1))(Qp4_ff)
   Qp4_ff = Dropout(0.3)(Qp4_ff)
   Qpp4 = Add()([Qp4, Qp4_ff])
   Qpp4 = LayerNormalization()(Qpp4)

   ## concatenate query outputs
   Q3 = concatenate([Qpp3, Qpp4], axis=chan_num)
   # process query
   feat_3aq = Conv2D(100, (1,1), padding='same', activation='relu')(Q3)
   Qpre3 = Reshape((1, 1, K.int_shape(feat_3aq)[1]*K.int_shape(feat_3aq)[2]*K.int_shape(feat_3aq)[3]))(feat_3aq)
   Q3 = Conv2D(d_mod, (1,1))(Qpre3)

   ## Tx block 3, layer 1
   # get key / feature projections of original feature map
   keys5   = Conv2D(d_mod, (1,1))(feature_maps)
   values5 = Conv2D(d_mod, (1,1))(feature_maps)
   # scaled dot-product attention
   a5 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q3, keys5])
   a5_sf = Reshape((K.int_shape(a5)[1]*K.int_shape(a5)[2], K.int_shape(a5)[3]))(a5)
   a5_sf = Softmax(axis=1)(a5_sf)
   a5_sf = Reshape((K.int_shape(a5)[1], K.int_shape(a5)[2], K.int_shape(a5)[3]))(a5_sf)
   A5 = Lambda(make_A)([a5_sf, values5])
   # combine with query
   A5 = Dropout(0.3)(A5)
   Qp5 = Add()([Q3, A5])
   Qp5 = LayerNormalization()(Qp5)
   # FFN
   Qp5_ff = Conv2D(256, (1,1))(Qp5)
   Qp5_ff = Activation('relu')(Qp5_ff)
   Qp5_ff = Conv2D(d_mod, (1,1))(Qp5_ff)
   Qp5_ff = Dropout(0.3)(Qp5_ff)
   Qpp5 = Add()([Qp5, Qp5_ff])
   Qpp5 = LayerNormalization()(Qpp5)
   
   ## Tx block 3, layer 2
   # get key / feature projections of original feature map
   keys6   = Conv2D(d_mod, (1,1))(feature_maps)
   values6 = Conv2D(d_mod, (1,1))(feature_maps)
   # scaled dot-product attention
   a6 = Lambda(make_a, arguments={'sqrtd':sqrt_d_mod})([Q3, keys6])
   a6_sf = Reshape((K.int_shape(a6)[1]*K.int_shape(a6)[2], K.int_shape(a6)[3]))(a6)
   a6_sf = Softmax(axis=1)(a6_sf)
   a6_sf = Reshape((K.int_shape(a6)[1], K.int_shape(a6)[2], K.int_shape(a6)[3]))(a6_sf)
   A6 = Lambda(make_A)([a6_sf, values6])
   # combine with query
   A6 = Dropout(0.3)(A6)
   Qp6 = Add()([Q3, A6])
   Qp6 = LayerNormalization()(Qp6)
   # FFN
   Qp6_ff = Conv2D(256, (1,1))(Qp6)
   Qp6_ff = Activation('relu')(Qp6_ff)
   Qp6_ff = Conv2D(d_mod, (1,1))(Qp6_ff)
   Qp6_ff = Dropout(0.3)(Qp6_ff)
   Qpp6 = Add()([Qp6, Qp6_ff])
   Qpp6 = LayerNormalization()(Qpp6)

   ## concatenate query outputs
   Q4 = concatenate([Qpp5, Qpp6], axis=chan_num)
   Q5 = Conv2D(d_mod, (1,1))(Q4)
   Q5 = Flatten()(Q5)

#   ## concatenate query outputs
#   Q5 = Conv2D(d_mod, (1,1))(Q3)
#   Q5 = Flatten()(Q5)
      
   # predict on query vector
   prediction = Dense(1, activation='sigmoid')(Q5)
   
   encoder = Model(inputs=[inputs, query],
                   outputs=prediction)   
   encoder.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder


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

      
