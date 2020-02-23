from keras.models       import Model, Sequential
from keras.layers       import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D
from keras.layers       import Softmax, Flatten, Dense, Reshape, Dropout, Lambda, Multiply, Add, Dot, Concatenate, AveragePooling2D, Lambda, Subtract
from keras.regularizers import l2
from keras.optimizers   import RMSprop, Adam
from keras.losses import binary_crossentropy
from DNN.losses         import *
import keras.backend as K

def inet5(input_shape1=(50, 50, 90), input_shape2=(384, 384, 3), chan_num=3, modtype=1):
   inputs_out_left = Input(shape=input_shape1)
   inputs_cr_left  = Input(shape=input_shape2)
   ## crypt
   crypt_map = Conv2D(48, (3, 3), padding='same', activation='relu')(inputs_cr_left)
   # squeeze excite block
   x = GlobalAveragePooling2D()(crypt_map)
   ch = K.int_shape(crypt_map)[chan_num]
   x = Dense(ch//8, activation='relu')(x)
   x = Dense(ch, activation='sigmoid')(x)
   crypt_map = Multiply()([crypt_map, x])
   crypt_map = MaxPooling2D((2, 2), strides=(2, 2))(crypt_map)
   # generate feature map using inception block   
   feat_1a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_1a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_1a)
   feat_2a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(crypt_map)
   feat_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feat_3a)
   feature_mapa = concatenate([feat_1a, feat_2a, feat_3a], axis = chan_num)
   # squeeze excite block
   xa = GlobalAveragePooling2D()(feature_mapa)
   cha = K.int_shape(feature_mapa)[chan_num]
   xa = Dense(cha//4, activation='relu')(xa)
   xa = Dense(cha, activation='sigmoid')(xa)
   feature_mapa = Multiply()([feature_mapa, xa])   
   # reduction layer
   red_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapa)
   red_2a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (3,3), padding='same', activation='relu')(red_3a)
   red_3a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3a)
   feature_map2a = concatenate([red_1a, red_2a, red_3a], axis = chan_num)
   # inception block
   feat_1b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_1b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_1b)
   feat_2b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_3b = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2a)
   feat_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feat_3b)
   feature_mapb = concatenate([feat_1b, feat_2b, feat_3b], axis = chan_num)
   # squeeze excite block
   xb = GlobalAveragePooling2D()(feature_mapb)
   chb = K.int_shape(feature_mapb)[chan_num]
   xb = Dense(chb//4, activation='relu')(xb)
   xb = Dense(chb, activation='sigmoid')(xb)
   feature_mapb = Multiply()([feature_mapb, xb])   
   # reduction layer
   red_1b = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapb)
   red_2b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (3,3), padding='same', activation='relu')(red_3b)
   red_3b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3b)
   feature_map2b = concatenate([red_1b, red_2b, red_3b], axis = chan_num)
   # inception block
   feat_1c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_map2b)
   feat_1c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_1c)
   feat_2c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_map2b)
   feat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_2c)
   feat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_2c)
   feat_3c = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2b)
   feat_3c = Conv2D(24, (1,1), padding='same', activation='relu')(feat_3c)
   feature_mapc = concatenate([feat_1c, feat_2c, feat_3c], axis = chan_num)
   if modtype==2:
      encoded_1 = Flatten()(feature_mapc)
   else:
      encoded_1 = GlobalAveragePooling2D()(feature_mapc)      
     
   ## context outer
   bigcontext_map = SeparableConv2D(90, (3, 3), padding='same', activation='relu')(inputs_out_left)
   # squeeze excite block
   x1 = GlobalAveragePooling2D()(bigcontext_map)
   ch1 = K.int_shape(bigcontext_map)[chan_num]
   x1 = Dense(ch1//8, activation='relu')(x1)
   x1 = Dense(ch1, activation='sigmoid')(x1)
   bigcontext_map = Multiply()([bigcontext_map, x1])
   # generate feature map using inception block   
   dfeat_1a = Conv2D(48, (1,1), padding='same', activation='relu')(bigcontext_map)
   dfeat_1a = Conv2D(48, (3,3), padding='same', activation='relu')(dfeat_1a)
   dfeat_2a = Conv2D(48, (1,1), padding='same', activation='relu')(bigcontext_map)
   dfeat_2a = Conv2D(48, (3,3), padding='same', activation='relu')(dfeat_2a)
   dfeat_2a = Conv2D(48, (3,3), padding='same', activation='relu')(dfeat_2a)
   dfeat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(bigcontext_map)
   dfeat_3a = Conv2D(48, (1,1), padding='same', activation='relu')(dfeat_3a)
   dfeature_mapa = concatenate([dfeat_1a, dfeat_2a, dfeat_3a], axis = chan_num)
   # squeeze excite block
   x1a = GlobalAveragePooling2D()(dfeature_mapa)
   ch1a = K.int_shape(dfeature_mapa)[chan_num]
   x1a = Dense(ch1a//4, activation='relu')(x1a)
   x1a = Dense(ch1a, activation='sigmoid')(x1a)
   dfeature_mapa = Multiply()([dfeature_mapa, x1a])
   # reduction layer
   dred_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapa)
   dred_2a = Conv2D(48, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapa)
   dred_3a = Conv2D(48, (1,1), padding='same', activation='relu')(dfeature_mapa)
   dred_3a = Conv2D(48, (3,3), padding='same', activation='relu')(dred_3a)
   dred_3a = Conv2D(48, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3a)
   dfeature_map2a = concatenate([dred_1a, dred_2a, dred_3a], axis = chan_num)
   if modtype==2:
      encoded_2 = Flatten()(dfeature_map2a)
   else:
      encoded_2 = GlobalAveragePooling2D()(dfeature_map2a)      
   
   ## subtract the two streams
   encoded_2 = Dense(K.int_shape(encoded_1)[1], activation='relu')(encoded_2)
   encoded = Subtract()([encoded_1, encoded_2])
   
   # Add a dense layer to classify as either WT or Clone
   prediction = Dense(1, activation='sigmoid')(encoded)
   
   encoder = Model(inputs=[inputs_out_left, inputs_cr_left],
                   outputs=prediction)   
   encoder.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder

def inet4(input_shape1=(50, 50, 90), input_shape2=(384, 384, 3), chan_num=3, modtype=1):
   inputs_out_left = Input(shape=input_shape1)
   inputs_cr_left  = Input(shape=input_shape2)
   ## crypt
   crypt_map = Conv2D(24, (3, 3), padding='same', activation='relu')(inputs_cr_left)
   crypt_map = MaxPooling2D((2, 2), strides=(2, 2))(crypt_map)
   # generate feature map using inception block   
   feat_1a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_1a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_1a)
   feat_2a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(crypt_map)
   feat_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feat_3a)
   feature_mapa = concatenate([feat_1a, feat_2a, feat_3a], axis = chan_num)
   # reduction layer
   red_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapa)
   red_2a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (3,3), padding='same', activation='relu')(red_3a)
   red_3a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3a)
   feature_map2a = concatenate([red_1a, red_2a, red_3a], axis = chan_num)
   # inception block
   feat_1b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_1b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_1b)
   feat_2b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_3b = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2a)
   feat_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feat_3b)
   feature_mapb = concatenate([feat_1b, feat_2b, feat_3b], axis = chan_num)
   # reduction layer
   red_1b = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapb)
   red_2b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (3,3), padding='same', activation='relu')(red_3b)
   red_3b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3b)
   feature_map2b = concatenate([red_1b, red_2b, red_3b], axis = chan_num)
   # inception block
   feat_1c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_map2b)
   feat_1c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_1c)
   feat_2c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_map2b)
   feat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_2c)
   feat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_2c)
   feat_3c = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2b)
   feat_3c = Conv2D(24, (1,1), padding='same', activation='relu')(feat_3c)
   feature_mapc = concatenate([feat_1c, feat_2c, feat_3c], axis = chan_num)
   if modtype==2:
      encoded_1 = Flatten()(feature_mapc)
   else:
      encoded_1 = GlobalAveragePooling2D()(feature_mapc)      
     
   ## context outer
   bigcontext_map = Conv2D(90, (3, 3), padding='same', activation='relu')(inputs_out_left)
   # generate feature map using inception block   
   dfeat_1a = Conv2D(48, (1,1), padding='same', activation='relu')(bigcontext_map)
   dfeat_1a = Conv2D(48, (3,3), padding='same', activation='relu')(dfeat_1a)
   dfeat_2a = Conv2D(48, (1,1), padding='same', activation='relu')(bigcontext_map)
   dfeat_2a = Conv2D(48, (3,3), padding='same', activation='relu')(dfeat_2a)
   dfeat_2a = Conv2D(48, (3,3), padding='same', activation='relu')(dfeat_2a)
   dfeat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(bigcontext_map)
   dfeat_3a = Conv2D(48, (1,1), padding='same', activation='relu')(dfeat_3a)
   dfeature_mapa = concatenate([dfeat_1a, dfeat_2a, dfeat_3a], axis = chan_num)
   # reduction layer
   dred_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapa)
   dred_2a = Conv2D(48, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapa)
   dred_3a = Conv2D(48, (1,1), padding='same', activation='relu')(dfeature_mapa)
   dred_3a = Conv2D(48, (3,3), padding='same', activation='relu')(dred_3a)
   dred_3a = Conv2D(48, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3a)
   dfeature_map2a = concatenate([dred_1a, dred_2a, dred_3a], axis = chan_num)
   if modtype==2:
      encoded_2 = Flatten()(dfeature_map2a)
   else:
      encoded_2 = GlobalAveragePooling2D()(dfeature_map2a)      
   
   ## subtract the two streams
   encoded_2 = Dense(K.int_shape(encoded_1)[1], activation='sigmoid')(encoded_2)
   encoded = Subtract()([encoded_1, encoded_2])
   
   # Add a dense layer to classify as either WT or Clone
   prediction = Dense(1, activation='sigmoid')(encoded)
   
   encoder = Model(inputs=[inputs_out_left, inputs_cr_left],
                   outputs=prediction)
   
   encoder.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder


def subset_tensor(var, csize):
   shape_x = K.int_shape(var)[1]
   lwr = (shape_x//2-csize//2)
   upr = (shape_x//2-csize//2 + csize)
   out1 = var[:, lwr:upr, lwr:upr, :]
   return out1

def inet3(input_shape1=(512, 512, 3), input_shape2=(384, 384, 3), chan_num=3, modtype=1):
   inputs_out_left = Input(shape=input_shape1)
   inputs_cr_left  = Input(shape=input_shape2)
   ## crypt
   crypt_map = Conv2D(24, (3, 3), padding='same', activation='relu')(inputs_cr_left)
   crypt_map = MaxPooling2D((2, 2), strides=(2, 2))(crypt_map)
   # generate feature map using inception block   
   feat_1a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_1a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_1a)
   feat_2a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(crypt_map)
   feat_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feat_3a)
   feature_mapa = concatenate([feat_1a, feat_2a, feat_3a], axis = chan_num)
   # reduction layer
   red_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapa)
   red_2a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (3,3), padding='same', activation='relu')(red_3a)
   red_3a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3a)
   feature_map2a = concatenate([red_1a, red_2a, red_3a], axis = chan_num)
   # inception block
   feat_1b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_1b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_1b)
   feat_2b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_3b = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2a)
   feat_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feat_3b)
   feature_mapb = concatenate([feat_1b, feat_2b, feat_3b], axis = chan_num)
   # reduction layer
   red_1b = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapb)
   red_2b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (3,3), padding='same', activation='relu')(red_3b)
   red_3b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3b)
   feature_map2b = concatenate([red_1b, red_2b, red_3b], axis = chan_num)
   # inception block
   feat_1c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_map2b)
   feat_1c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_1c)
   feat_2c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_map2b)
   feat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_2c)
   feat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_2c)
   feat_3c = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2b)
   feat_3c = Conv2D(24, (1,1), padding='same', activation='relu')(feat_3c)
   feature_mapc = concatenate([feat_1c, feat_2c, feat_3c], axis = chan_num)
     
   ## context outer
   bigcontext_map = Conv2D(12, (5, 5), padding='valid', activation='relu')(inputs_out_left)
   bigcontext_map = Conv2D(18, (5, 5), padding='valid', activation='relu', dilation_rate=2)(bigcontext_map)
   bigcontext_map = Conv2D(24, (5, 5), padding='valid', activation='relu', dilation_rate=2)(bigcontext_map)
   bigcontext_map = Conv2D(30, (5, 5), padding='valid', activation='relu', dilation_rate=2)(bigcontext_map)
   bigcontext_map = Conv2D(36, (5, 5), padding='valid', activation='relu', dilation_rate=2)(bigcontext_map)
   bigcontext_map = Conv2D(42, (5, 5), padding='valid', activation='relu', dilation_rate=2)(bigcontext_map)
   # join to crypt image
   cr_size = K.int_shape(feature_mapc)[1]
   subsetcontext_map = Lambda(subset_tensor,  arguments={'csize':cr_size})(bigcontext_map)   
   feature_map_j = concatenate([feature_mapc, subsetcontext_map], axis = chan_num)
   
   feature_map_j = Conv2D(36, (3, 3), padding='same', activation='relu')(feature_map_j)
   encoded = GlobalAveragePooling2D()(feature_map_j)
   
   # Add a dense layer to classify as either WT or Clone
   prediction = Dense(1, activation='sigmoid')(encoded)
   
   encoder = Model(inputs=[inputs_out_left, inputs_cr_left],
                   outputs=prediction)
   
   encoder.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder

def inet2(input_shape1=(512, 512, 3), input_shape2=(384, 384, 3), chan_num=3, modtype=1):
   inputs_out_left = Input(shape=input_shape1)
   inputs_cr_left  = Input(shape=input_shape2)
   ## crypt
   crypt_map = Conv2D(24, (3, 3), padding='same', activation='relu')(inputs_cr_left)
   crypt_map = MaxPooling2D((2, 2), strides=(2, 2))(crypt_map)
   # generate feature map using inception block   
   feat_1a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_1a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_1a)
   feat_2a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(crypt_map)
   feat_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feat_3a)
   feature_mapa = concatenate([feat_1a, feat_2a, feat_3a], axis = chan_num)
   # reduction layer
   red_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapa)
   red_2a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (3,3), padding='same', activation='relu')(red_3a)
   red_3a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3a)
   feature_map2a = concatenate([red_1a, red_2a, red_3a], axis = chan_num)
   # inception block
   feat_1b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_1b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_1b)
   feat_2b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_3b = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2a)
   feat_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feat_3b)
   feature_mapb = concatenate([feat_1b, feat_2b, feat_3b], axis = chan_num)
   # reduction layer
   red_1b = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapb)
   red_2b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (3,3), padding='same', activation='relu')(red_3b)
   red_3b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3b)
   feature_map2b = concatenate([red_1b, red_2b, red_3b], axis = chan_num)
   if modtype==2:
      encoded_1 = Flatten()(feature_map2b)
   else:
      encoded_1 = GlobalAveragePooling2D()(feature_map2b)

   ## context outer
   bigcontext_map = Conv2D(24, (3, 3), padding='same', activation='relu')(inputs_out_left)
   bigcontext_map = MaxPooling2D((2, 2), strides=(2, 2))(bigcontext_map)
   # generate feature map using inception block   
   dfeat_1a = Conv2D(8, (1,1), padding='same', activation='relu')(bigcontext_map)
   dfeat_1a = Conv2D(8, (3,3), padding='same', activation='relu')(dfeat_1a)
   dfeat_2a = Conv2D(8, (1,1), padding='same', activation='relu')(bigcontext_map)
   dfeat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(dfeat_2a)
   dfeat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(dfeat_2a)
   dfeat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(bigcontext_map)
   dfeat_3a = Conv2D(8, (1,1), padding='same', activation='relu')(dfeat_3a)
   dfeature_mapa = concatenate([dfeat_1a, dfeat_2a, dfeat_3a], axis = chan_num)
   # reduction layer
   dred_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapa)
   dred_2a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapa)
   dred_3a = Conv2D(8, (1,1), padding='same', activation='relu')(dfeature_mapa)
   dred_3a = Conv2D(8, (3,3), padding='same', activation='relu')(dred_3a)
   dred_3a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3a)
   dfeature_map2a = concatenate([dred_1a, dred_2a, dred_3a], axis = chan_num)
   # inception block
   dfeat_1b = Conv2D(16, (1,1), padding='same', activation='relu')(dfeature_map2a)
   dfeat_1b = Conv2D(16, (3,3), padding='same', activation='relu')(dfeat_1b)
   dfeat_2b = Conv2D(16, (1,1), padding='same', activation='relu')(dfeature_map2a)
   dfeat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(dfeat_2b)
   dfeat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(dfeat_2b)
   dfeat_3b = MaxPooling2D((3,3), strides=(1,1), padding='same')(dfeature_map2a)
   dfeat_3b = Conv2D(16, (1,1), padding='same', activation='relu')(dfeat_3b)
   dfeature_mapb = concatenate([dfeat_1b, dfeat_2b, dfeat_3b], axis = chan_num)
   # reduction layer
   dred_1b = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapb)
   dred_2b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapb)
   dred_3b = Conv2D(16, (1,1), padding='same', activation='relu')(dfeature_mapb)
   dred_3b = Conv2D(16, (3,3), padding='same', activation='relu')(dred_3b)
   dred_3b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3b)
   dfeature_map2b = concatenate([dred_1b, dred_2b, dred_3b], axis = chan_num)
   # inception block
   dfeat_1c = Conv2D(24, (1,1), padding='same', activation='relu')(dfeature_map2b)
   dfeat_1c = Conv2D(24, (3,3), padding='same', activation='relu')(dfeat_1c)
   dfeat_2c = Conv2D(24, (1,1), padding='same', activation='relu')(dfeature_map2b)
   dfeat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(dfeat_2c)
   dfeat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(dfeat_2c)
   dfeat_3c = MaxPooling2D((3,3), strides=(1,1), padding='same')(dfeature_map2b)
   dfeat_3c = Conv2D(24, (1,1), padding='same', activation='relu')(dfeat_3c)
   dfeature_mapc = concatenate([dfeat_1c, dfeat_2c, dfeat_3c], axis = chan_num)
   # reduction layer
   dred_1c = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapc)
   dred_2c = Conv2D(24, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapc)
   dred_3c = Conv2D(24, (1,1), padding='same', activation='relu')(dfeature_mapc)
   dred_3c = Conv2D(24, (3,3), padding='same', activation='relu')(dred_3c)
   dred_3c = Conv2D(24, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3c)
   dfeature_map2c = concatenate([dred_1c, dred_2c, dred_3c], axis = chan_num)
   if modtype==2:
      encoded_2 = Flatten()(dfeature_map2c)
   else:
      encoded_2 = GlobalAveragePooling2D()(dfeature_map2c)      
   
   if modtype==3:
         encoded_2 = Dense(K.int_shape(encoded_1)[1], activation='sigmoid')(encoded_2)
         encoded = Lambda(lambda x: x[0]-x[1])([encoded_1, encoded_2])
   else:
      encoded = concatenate([encoded_1, encoded_2], axis=-1)
   
   # Add a dense layer to classify as either WT or Clone
   prediction = Dense(1, activation='sigmoid')(encoded)
   
   encoder = Model(inputs=[inputs_out_left, inputs_cr_left],
                   outputs=prediction)
   
   encoder.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
   return encoder

def inet(input_shape1=(512, 512, 3), input_shape2=(384, 384, 3), chan_num=3, modtype=1):
   inputs_out_left = Input(shape=input_shape1)
   inputs_cr_left  = Input(shape=input_shape2)
   ## crypt
   crypt_map = Conv2D(24, (3, 3), padding='same', activation='relu')(inputs_cr_left)
   crypt_map = MaxPooling2D((2, 2), strides=(2, 2))(crypt_map)

   # generate feature map using inception block   
   feat_1a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_1a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_1a)
   feat_2a = Conv2D(8, (1,1), padding='same', activation='relu')(crypt_map)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(feat_2a)
   feat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(crypt_map)
   feat_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feat_3a)
   feature_mapa = concatenate([feat_1a, feat_2a, feat_3a], axis = chan_num)
   # reduction layer
   red_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapa)
   red_2a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (1,1), padding='same', activation='relu')(feature_mapa)
   red_3a = Conv2D(8, (3,3), padding='same', activation='relu')(red_3a)
   red_3a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3a)
   feature_map2a = concatenate([red_1a, red_2a, red_3a], axis = chan_num)
   # inception block
   feat_1b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_1b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_1b)
   feat_2b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_map2a)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(feat_2b)
   feat_3b = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2a)
   feat_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feat_3b)
   feature_mapb = concatenate([feat_1b, feat_2b, feat_3b], axis = chan_num)
   # reduction layer
   red_1b = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapb)
   red_2b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (1,1), padding='same', activation='relu')(feature_mapb)
   red_3b = Conv2D(16, (3,3), padding='same', activation='relu')(red_3b)
   red_3b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3b)
   feature_map2b = concatenate([red_1b, red_2b, red_3b], axis = chan_num)
   # inception block
   feat_1c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_map2b)
   feat_1c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_1c)
   feat_2c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_map2b)
   feat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_2c)
   feat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(feat_2c)
   feat_3c = MaxPooling2D((3,3), strides=(1,1), padding='same')(feature_map2b)
   feat_3c = Conv2D(24, (1,1), padding='same', activation='relu')(feat_3c)
   feature_mapc = concatenate([feat_1c, feat_2c, feat_3c], axis = chan_num)
   # reduction layer
   red_1c = MaxPooling2D((3,3), strides=(2,2), padding='valid')(feature_mapc)
   red_2c = Conv2D(24, (3,3), strides=(2,2), padding='valid', activation='relu')(feature_mapc)
   red_3c = Conv2D(24, (1,1), padding='same', activation='relu')(feature_mapc)
   red_3c = Conv2D(24, (3,3), padding='same', activation='relu')(red_3c)
   red_3c = Conv2D(24, (3,3), strides=(2,2), padding='valid', activation='relu')(red_3c)
   feature_map2c = concatenate([red_1c, red_2c, red_3c], axis = chan_num)
   if modtype==2:
      encoded_1 = Flatten()(feature_map2c)
   else:
      encoded_1 = GlobalAveragePooling2D()(feature_map2c)

  
   ## context outer
   bigcontext_map = Conv2D(24, (3, 3), padding='same', activation='relu')(inputs_out_left)
   bigcontext_map = MaxPooling2D((2, 2), strides=(2, 2))(bigcontext_map)

   # generate feature map using inception block   
   dfeat_1a = Conv2D(8, (1,1), padding='same', activation='relu')(bigcontext_map)
   dfeat_1a = Conv2D(8, (3,3), padding='same', activation='relu')(dfeat_1a)
   dfeat_2a = Conv2D(8, (1,1), padding='same', activation='relu')(bigcontext_map)
   dfeat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(dfeat_2a)
   dfeat_2a = Conv2D(8, (3,3), padding='same', activation='relu')(dfeat_2a)
   dfeat_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')(bigcontext_map)
   dfeat_3a = Conv2D(8, (1,1), padding='same', activation='relu')(dfeat_3a)
   dfeature_mapa = concatenate([dfeat_1a, dfeat_2a, dfeat_3a], axis = chan_num)
   # reduction layer
   dred_1a = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapa)
   dred_2a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapa)
   dred_3a = Conv2D(8, (1,1), padding='same', activation='relu')(dfeature_mapa)
   dred_3a = Conv2D(8, (3,3), padding='same', activation='relu')(dred_3a)
   dred_3a = Conv2D(8, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3a)
   dfeature_map2a = concatenate([dred_1a, dred_2a, dred_3a], axis = chan_num)
   # inception block
   dfeat_1b = Conv2D(16, (1,1), padding='same', activation='relu')(dfeature_map2a)
   dfeat_1b = Conv2D(16, (3,3), padding='same', activation='relu')(dfeat_1b)
   dfeat_2b = Conv2D(16, (1,1), padding='same', activation='relu')(dfeature_map2a)
   dfeat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(dfeat_2b)
   dfeat_2b = Conv2D(16, (3,3), padding='same', activation='relu')(dfeat_2b)
   dfeat_3b = MaxPooling2D((3,3), strides=(1,1), padding='same')(dfeature_map2a)
   dfeat_3b = Conv2D(16, (1,1), padding='same', activation='relu')(dfeat_3b)
   dfeature_mapb = concatenate([dfeat_1b, dfeat_2b, dfeat_3b], axis = chan_num)
   # reduction layer
   dred_1b = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapb)
   dred_2b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapb)
   dred_3b = Conv2D(16, (1,1), padding='same', activation='relu')(dfeature_mapb)
   dred_3b = Conv2D(16, (3,3), padding='same', activation='relu')(dred_3b)
   dred_3b = Conv2D(16, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3b)
   dfeature_map2b = concatenate([dred_1b, dred_2b, dred_3b], axis = chan_num)
   # inception block
   dfeat_1c = Conv2D(24, (1,1), padding='same', activation='relu')(dfeature_map2b)
   dfeat_1c = Conv2D(24, (3,3), padding='same', activation='relu')(dfeat_1c)
   dfeat_2c = Conv2D(24, (1,1), padding='same', activation='relu')(dfeature_map2b)
   dfeat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(dfeat_2c)
   dfeat_2c = Conv2D(24, (3,3), padding='same', activation='relu')(dfeat_2c)
   dfeat_3c = MaxPooling2D((3,3), strides=(1,1), padding='same')(dfeature_map2b)
   dfeat_3c = Conv2D(24, (1,1), padding='same', activation='relu')(dfeat_3c)
   dfeature_mapc = concatenate([dfeat_1c, dfeat_2c, dfeat_3c], axis = chan_num)
   # reduction layer
   dred_1c = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapc)
   dred_2c = Conv2D(24, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapc)
   dred_3c = Conv2D(24, (1,1), padding='same', activation='relu')(dfeature_mapc)
   dred_3c = Conv2D(24, (3,3), padding='same', activation='relu')(dred_3c)
   dred_3c = Conv2D(24, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3c)
   dfeature_map2c = concatenate([dred_1c, dred_2c, dred_3c], axis = chan_num)
   # inception block
   dfeat_1d = Conv2D(20, (1,1), padding='same', activation='relu')(dfeature_map2c)
   dfeat_1d = Conv2D(20, (3,3), padding='same', activation='relu')(dfeat_1d)
   dfeat_2d = Conv2D(20, (1,1), padding='same', activation='relu')(dfeature_map2c)
   dfeat_2d = Conv2D(20, (3,3), padding='same', activation='relu')(dfeat_2d)
   dfeat_2d = Conv2D(20, (3,3), padding='same', activation='relu')(dfeat_2d)
   dfeat_3d = MaxPooling2D((3,3), strides=(1,1), padding='same')(dfeature_map2c)
   dfeat_3d = Conv2D(20, (1,1), padding='same', activation='relu')(dfeat_3d)
   dfeature_mapd = concatenate([dfeat_1d, dfeat_2d, dfeat_3d], axis = chan_num)
   # reduction layer
   dred_1d = MaxPooling2D((3,3), strides=(2,2), padding='valid')(dfeature_mapd)
   dred_2d = Conv2D(20, (3,3), strides=(2,2), padding='valid', activation='relu')(dfeature_mapd)
   dred_3d = Conv2D(20, (1,1), padding='same', activation='relu')(dfeature_mapd)
   dred_3d = Conv2D(20, (3,3), padding='same', activation='relu')(dred_3d)
   dred_3d = Conv2D(20, (3,3), strides=(2,2), padding='valid', activation='relu')(dred_3d)
   dfeature_map2d = concatenate([dred_1d, dred_2d, dred_3d], axis = chan_num)
   if modtype==2:
      encoded_2 = Flatten()(dfeature_map2d)
   else:
      encoded_2 = GlobalAveragePooling2D()(dfeature_map2d)      
   
   if modtype==3:
         encoded_2 = Dense(K.int_shape(encoded_1)[1], activation='sigmoid')(encoded)
         encoded = encoded_1 - encoded_2
   else:
      encoded = concatenate([encoded_1, encoded_2], axis=-1)
   
   # Add a dense layer to classify as either WT or Clone
   prediction = Dense(1, activation='sigmoid')(encoded)
   
   encoder = Model(inputs=[inputs_out_left, inputs_cr_left],
                   outputs=prediction)
   
   encoder.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

   return encoder
