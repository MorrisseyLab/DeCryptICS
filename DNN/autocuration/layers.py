import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Conv2D, concatenate, SeparableConv2D, Add,\
                                    LayerNormalization, Dropout, Activation                                    
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.ops import array_ops, nn
from tensorflow.keras.initializers import Ones, Zeros

class AttentionAugmentation2D(tf.keras.models.Model):
    def __init__(self, Fout, k, dk, dv, Nh, relative=False):
        super(AttentionAugmentation2D,self).__init__()
        self.output_filters = Fout
        self.depth_k = dk
        self.depth_v = dv
        self.num_heads = Nh
        self.relative = relative

        self.conv_out = SeparableConv2D(filters = Fout - dv, kernel_size = k, padding="same")
        self.qkv = Conv2D(filters = 2*dk + dv, kernel_size= 1)
        self.attn_out_conv = Conv2D(filters = dv, kernel_size = 1)

    def call(self,inputs):
        out = self.conv_out(inputs)
        shape = K.int_shape(inputs)
        if None in shape:
            shape = [-1 if type(v)==type(None) else v for v in shape]
        Batch,H,W,_ = shape

        flat_q, flat_k, flat_v = self.compute_flat_qkv(inputs)
        dkh = self.depth_k // self.num_heads
        # might be able to calculate this differently as query will be "1x1xdepth"
        # so same all over the spatial dimensions? (reduce size of problem)
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(self.q, H, W, self.num_heads,dkh)
            logits += h_rel_logits
            logits += w_rel_logits
        
        weights = K.softmax(logits, axis=-1)
        attn_out = tf.matmul(weights, flat_v)
        attn_out = K.reshape(attn_out, [Batch, self.num_heads, H, W, self.depth_v // self.num_heads])

        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out_conv(attn_out)
        output =  concatenate([out,attn_out],axis=3)
        return output

    def combine_heads_2d(self,x):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = K.permute_dimensions(x, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = K.int_shape(transposed)
        if None in shape:
            shape = [-1 if type(v)==type(None) else v for v in shape]
        batch, h , w, a , b = shape 
        ret_shape = [batch, h ,w, a*b]
        # [batch, height, width, depth_v]
        return K.reshape(transposed, ret_shape)

    def rel_to_abs(self, x):
        shape = K.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L = shape
        col_pad = K.zeros((B, Nh, L, 1), name="zero1")
        x = K.concatenate([x, col_pad], axis=3)
        flat_x = K.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = K.zeros((B, Nh, L-1), name="zero2")
        flat_x_padded = K.concatenate([flat_x, flat_pad], axis=2)
        final_x = K.reshape(flat_x_padded, [B, Nh, L+1, 2*L-1])
        final_x = final_x[:, :, :L, L-1:]
        return final_x

    def relative_logits_1d(self, q, rel_k, H, W, Nh, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = K.reshape(rel_logits, [-1, Nh*H, W, 2*W-1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = K.reshape(rel_logits, [-1, Nh, H, W, W])
        rel_logits = K.expand_dims(rel_logits, axis=3)
        rel_logits = K.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = K.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = K.reshape(rel_logits, [-1, Nh, H*W, H*W])
        return rel_logits

    def relative_logits(self, q, H, W, Nh, dkh):
        key_rel_w  = K.random_normal(shape = (int(2 * W - 1), dkh))
        key_rel_h  = K.random_normal(shape = (int(2 * H - 1), dkh))

        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, [0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(K.permute_dimensions(q, [0, 1, 3, 2, 4]), 
                                               key_rel_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
        return rel_logits_h , rel_logits_w

    def split_heads_2d(self,q,Nh):
        batch, height,width,channels = K.int_shape(q)
        ret_shape = [-1,height,width,Nh,channels//Nh]
        split = K.reshape(q,ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = K.permute_dimensions(split, transpose_axes)
        return split

    def compute_flat_qkv(self,inputs):
        qkv = self.qkv(inputs)
        B,H,W,_ = K.int_shape(inputs)
        q,k,v = tf.split(qkv,[self.depth_k,self.depth_k,self.depth_v],axis=3)

        dkh = self.depth_k // self.num_heads
        dvh = self.depth_v // self.num_heads
        q *= dkh ** -0.5

        self.q = self.split_heads_2d(q, self.num_heads)
        self.k = self.split_heads_2d(k, self.num_heads)
        self.v = self.split_heads_2d(v, self.num_heads)

        flat_q = K.reshape(self.q, [-1, self.num_heads, H * W, dkh])
        flat_k = K.reshape(self.k, [-1, self.num_heads, H * W, dkh])
        flat_v = K.reshape(self.v, [-1, self.num_heads, H * W, dvh])

        return flat_q, flat_k, flat_v
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_filters
        return tuple(output_shape)

class AttentionROI(tf.keras.models.Model):
    def __init__(self, Fout, dk, dff, Nh, dr, relative=False):
        super(AttentionROI,self).__init__()
        self.output_filters = Fout
        self.depth_k = dk
        self.depth_v = Fout
        self.num_heads = Nh
        self.relative = relative

        # define sub layers
        self.kv = Conv2D(filters = dk + Fout, kernel_size = 1)
        self.q_init = Conv2D(filters = dk, kernel_size = 1)
        self.attn_out_conv = Conv2D(filters = Fout, kernel_size = 1)
        self.c_ffn_1 = Conv2D(filters = dff, kernel_size = 1)
        self.c_ffn_2 = Conv2D(filters = Fout, kernel_size = 1)
        self.dropout = Dropout(rate = dr)
        self.activ = Activation('relu')
        self.laynorm = LayerNormalization()
        self.add = Add()

    def call(self, inputs):
        shape = K.int_shape(inputs[1])        
        if None in shape:
            shape = [-1 if type(v)==type(None) else v for v in shape]
        Batch, H, W, _ = shape
        tileby = tf.constant([1, H, W, 1], tf.int32)
               
        flat_q, flat_k, flat_v = self.compute_flat_qkv([inputs[0], inputs[1]])
        
        logits_red = tf.matmul(flat_q, flat_k, transpose_b=True)
        weights_red = K.softmax(logits_red, axis=-1)
        
        attn_out_red = tf.matmul(weights_red, flat_v)
        attn_out_red = K.reshape(attn_out_red, [Batch, self.num_heads, 1, 1, self.depth_v // self.num_heads])
        attn_out_red = self.combine_heads_2d(attn_out_red)
        attn_out_red = tf.tile(attn_out_red, tileby)   
        
        A = self.attn_out_conv(attn_out_red)
        A = K.sum(A, axis=2, keepdims=True)
        A = K.sum(A, axis=1, keepdims=True)        
        # combine with query
        A = self.dropout(A)
        Qp = self.add([inputs[0], A])
        Qp = self.laynorm(Qp)
        # FFN
        Qp_ff = self.c_ffn_1(Qp)
        Qp_ff = self.activ(Qp_ff)
        Qp_ff = self.c_ffn_2(Qp_ff)
        Qp_ff = self.dropout(Qp_ff)
        Qpp = self.add([Qp, Qp_ff])
        Qpp = self.laynorm(Qpp)        
        return Qpp

    def combine_heads_2d(self, x):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = K.permute_dimensions(x, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = K.int_shape(transposed)
        if None in shape:
            shape = [-1 if type(v)==type(None) else v for v in shape]
        batch, h, w, a, b = shape 
        ret_shape = [batch, h ,w, a*b]
        # [batch, height, width, depth_v]
        return K.reshape(transposed, ret_shape)

    def split_heads_2d(self,q,Nh):
        batch, height, width, channels = K.int_shape(q)
        ret_shape = [-1, height, width, Nh, channels//Nh]
        split = K.reshape(q,ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = K.permute_dimensions(split, transpose_axes)
        return split

    def compute_flat_qkv(self, inputs_a):
        kv = self.kv(inputs_a[1])
        q = self.q_init(inputs_a[0])
        B, H, W, _ = K.int_shape(inputs_a[1])
        k, v = tf.split(kv, [self.depth_k, self.depth_v], axis=3)

        dkh = self.depth_k // self.num_heads
        dvh = self.depth_v // self.num_heads
        q *= dkh ** -0.5

        self.q = self.split_heads_2d(q, self.num_heads)
        self.k = self.split_heads_2d(k, self.num_heads)
        self.v = self.split_heads_2d(v, self.num_heads)

        flat_q = K.reshape(self.q, [-1, self.num_heads, 1    , dkh])
        flat_k = K.reshape(self.k, [-1, self.num_heads, H * W, dkh])
        flat_v = K.reshape(self.v, [-1, self.num_heads, H * W, dvh])

        return flat_q, flat_k, flat_v
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape[-1] = self.output_filters
        return tuple(output_shape)

class ROIPoolingLayer(Layer):
   """ Implements Region Of Interest Max Pooling 
       for channel-last images and relative bounding box coordinates

       # Constructor parameters
          pooled_width, pooled_height (int) -- 
          specify width and height of layer outputs

       Shape of inputs
          [(batch_size, pooled_width, pooled_height, n_channels),
           (batch_size, 4)]
        
       Shape of output
          (batch_size, pooled_width, pooled_height, n_channels)

   """
   def __init__(self, pooled_width, pooled_height, **kwargs):
      self.pooled_height = pooled_height
      self.pooled_width = pooled_width

      super(ROIPoolingLayer, self).__init__(**kwargs)
  
   def compute_output_shape(self, input_shape):
      """ Returns the shape of the ROI Layer output
      """
      feature_map_shape, rois_shape = input_shape
      assert feature_map_shape[0] == rois_shape[0]
      batch_size = feature_map_shape[0]
      n_channels = feature_map_shape[3]
      return (batch_size, self.pooled_width, 
             self.pooled_height, n_channels)

   def call(self, x):
      """ Maps the input tensor of the ROI layer to its output

         # Parameters
             x[0] -- Convolutional feature map tensor,
                     shape (batch_size, height, width, n_channels)
             x[1] -- Tensor of region of interests from candidate bounding boxes,
                     shape (batch_size, 4)
                     Each region of interest is defined by four relative 
                     coordinates (x_min, y_min, x_max, y_max) between 0 and 1
         # Output
             pooled_areas -- Tensor with the pooled region of interest, shape
                 (batch_size, pooled_height, pooled_width, n_channels)
      """
      def curried_pool_rois(x): 
         return ROIPoolingLayer._pool_roi(x[0], x[1], self.pooled_width, self.pooled_height)
     
      pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)
      return pooled_areas
   
   @staticmethod
   def _pool_roi(feature_map, roi, pooled_width, pooled_height):
      """ Applies ROI pooling to a single image and a single region of interest
      """

      # Compute the region of interest        
      feature_map_height = int(feature_map.shape[0])
      feature_map_width  = int(feature_map.shape[1])

      x_start = tf.cast(feature_map_width  * roi[0], 'int32')
      y_start = tf.cast(feature_map_height * roi[1], 'int32')
      x_end   = tf.cast(feature_map_width  * roi[2], 'int32')
      y_end   = tf.cast(feature_map_height * roi[3], 'int32')

      region = feature_map[y_start:y_end, x_start:x_end, :]

      ## Divide the region into non overlapping areas
      region_height_int = K.int_shape(region)[0]
      region_width_int  = K.int_shape(region)[1]
      region_height = y_end - y_start
      region_width  = x_end - x_start
      nbins_x = pooled_width
      nbins_y = pooled_height
      x_inds = [0]*(nbins_x)
      y_inds = [0]*(nbins_y)
      overhang_x = region_width_int // nbins_x
      overhang_y = region_height_int // nbins_y
      
      normal_bin_width_y = tf.math.floordiv(region_height, pooled_height)
      normal_bin_width_x = tf.math.floordiv(region_width, pooled_width)
      ## (nbins - overhang) * normal_bin_width + overhang*(normal_bin_width + 1) == numpixels
      cw = normal_bin_width_x
      for i in range(nbins_x-overhang_x):
         x_inds[i] = (i*cw, (i+1)*cw)
      done = (nbins_x-overhang_x)*normal_bin_width_x
      cw   = normal_bin_width_x + 1
      for i in range(overhang_x):
         x_inds[i + nbins_x-overhang_x] = (done + i*cw, done + (i+1)*cw)

      cw = normal_bin_width_y
      for i in range(nbins_y-overhang_y):
         y_inds[i] = (i*cw, (i+1)*cw)
      done = (nbins_y-overhang_y)*normal_bin_width_y
      cw   = normal_bin_width_y + 1
      for i in range(overhang_y):
         y_inds[i + nbins_y-overhang_y] = (done + i*cw, done + (i+1)*cw)

      areas = [ [(xx[0], yy[0], xx[1], yy[1]) for xx in x_inds] for yy in y_inds ]

      # take the maximum of each area and stack the result
      def pool_area(x): 
         return K.max(region[x[1]:x[3], x[0]:x[2], :], axis=[0,1])

      pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
      return pooled_features


