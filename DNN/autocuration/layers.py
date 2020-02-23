import tensorflow as tf
from keras.layers import Layer
import keras.backend as K
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from keras.initializers import Ones, Zeros

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

#class ROIPoolingLayer(Layer):
#      """ Implements Region Of Interest Max Pooling 
#        for channel-last images and relative bounding box coordinates
#        
#        # Constructor parameters
#            pooled_width, pooled_height (int) -- 
#              specify width and height of layer outputs
#        
#        Shape of inputs
#            [(batch_size, pooled_width, pooled_height, n_channels),
#             (batch_size, 4)]
#           
#        Shape of output
#            (batch_size, pooled_width, pooled_height, n_channels)

#      """
#      def __init__(self, pooled_width, pooled_height, **kwargs):
#         self.pooled_height = pooled_height
#         self.pooled_width = pooled_width

#         super(ROIPoolingLayer, self).__init__(**kwargs)
#        
#      def compute_output_shape(self, input_shape):
#         """ Returns the shape of the ROI Layer output
#         """
#         feature_map_shape, rois_shape = input_shape
#         assert feature_map_shape[0] == rois_shape[0]
#         batch_size = feature_map_shape[0]
#         n_channels = feature_map_shape[3]
#         return (batch_size, self.pooled_width, 
#                self.pooled_height, n_channels)

#      def call(self, x):
#         """ Maps the input tensor of the ROI layer to its output

#            # Parameters
#                x[0] -- Convolutional feature map tensor,
#                        shape (batch_size, pooled_height, pooled_width, n_channels)
#                x[1] -- Tensor of region of interests from candidate bounding boxes,
#                        shape (batch_size, 4)
#                        Each region of interest is defined by four relative 
#                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
#            # Output
#                pooled_areas -- Tensor with the pooled region of interest, shape
#                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
#         """
#         pooled_areas = self._pool_roi(x[0], x[1], 
#                                      self.pooled_width, 
#                                      self.pooled_height)
#         return pooled_areas
#    
#      def _pool_roi(self, feature_map, roi, pooled_width, pooled_height):
#         """ Applies ROI pooling to a single image and a single region of interest
#         """

#         # Compute the region of interest        
#         feature_map_height = int(feature_map.shape[1])
#         feature_map_width  = int(feature_map.shape[2])

#         x_start = tf.cast(feature_map_width  * roi[0], 'int32')
#         y_start = tf.cast(feature_map_height * roi[1], 'int32')
#         x_end   = tf.cast(feature_map_width  * roi[2], 'int32')
#         y_end   = tf.cast(feature_map_height * roi[3], 'int32')

#         region = feature_map[y_start:y_end, x_start:x_end, :]

#         # Divide the region into non overlapping areas
#         region_height = y_end - y_start
#         region_width  = x_end - x_start
#         y_step = tf.cast( region_height / pooled_height, 'int32')
#         x_step = tf.cast( region_width  / pooled_width , 'int32')
#        
#        # this just puts all remainder at edge, get better binning!
##        areas = [[(
##                    i*y_step, 
##                    j*x_step, 
##                    (i+1)*y_step if i+1 < pooled_height else region_height, 
##                    (j+1)*x_step if j+1 < pooled_width else region_width
##                   ) 
##                   for j in range(pooled_width)] 
##                  for i in range(pooled_height)]

#         numpixels_x = K.get_value(region_width)
#         nbins_x = pooled_width
#         x_inds = [0]*(nbins_x)
#         overhang_x  = numpixels_x % nbins_x
#         normal_bin_width_x = numpixels_x // nbins_x
#         # (nbins - overhang) * normal_bin_width + overhang*(normal_bin_width + 1) == numpixels
#         cw = normal_bin_width_x
#         for i in range(nbins_x-overhang_x):
#            x_inds[i] = (i*cw, (i+1)*cw)
#         done = (nbins_x-overhang_x)*normal_bin_width_x
#         cw   = normal_bin_width_x + 1
#         for i in range(overhang_x):
#            x_inds[i + nbins_x-overhang_x] = (done + i*cw, done + (i+1)*cw)
#            
#         numpixels_y = K.get_value(region_height)
#         nbins_y = pooled_height
#         y_inds = [0]*(nbins_y)
#         overhang_y  = numpixels_y % nbins_y
#         normal_bin_width_y = numpixels_y // nbins_y
#         cw = normal_bin_width_y
#         for i in range(nbins_y-overhang_y):
#            y_inds[i] = (i*cw, (i+1)*cw)
#         done = (nbins_y-overhang_y)*normal_bin_width_y
#         cw   = normal_bin_width_y + 1
#         for i in range(overhang_y):
#            y_inds[i + nbins_y-overhang_y] = (done + i*cw, done + (i+1)*cw)

#         areas = [[(
#                       xx[0], 
#                       yy[0], 
#                       xx[1], 
#                       yy[1]
#                    ) 
#                      for xx in x_inds] 
#                     for yy in y_inds]
#        
#         # take the maximum of each area and stack the result
#         def pool_area(x): 
#            return K.mean(region[x[1]:x[3], x[0]:x[2], :], axis=[0,1])
#        
#         pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
#         return pooled_features



#class ROIPoolingLayer(Layer):
#      """ Implements Region Of Interest Max Pooling 
#        for channel-last images and relative bounding box coordinates
#        
#        # Constructor parameters
#            pooled_width, pooled_height (int) -- 
#              specify width and height of layer outputs
#        
#        Shape of inputs
#            [(batch_size, pooled_width, pooled_height, n_channels),
#             (batch_size, num_rois, 4)]
#           
#        Shape of output
#            (batch_size, num_rois, pooled_width, pooled_height, n_channels)

#      """
#      def __init__(self, pooled_width, pooled_height, **kwargs):
#         self.pooled_height = pooled_height
#         self.pooled_width = pooled_width

#         super(ROIPoolingLayer, self).__init__(**kwargs)
#        
#      def compute_output_shape(self, input_shape):
#         """ Returns the shape of the ROI Layer output
#         """
#         feature_map_shape, rois_shape = input_shape
#         assert feature_map_shape[0] == rois_shape[0]
#         batch_size = feature_map_shape[0]
#         n_rois = rois_shape[1]
#         n_channels = feature_map_shape[3]
#         return (batch_size, n_rois, self.pooled_width, 
#                self.pooled_height, n_channels)

#      def call(self, x):
#         """ Maps the input tensor of the ROI layer to its output

#            # Parameters
#                x[0] -- Convolutional feature map tensor,
#                        shape (batch_size, pooled_height, pooled_width, n_channels)
#                x[1] -- Tensor of region of interests from candidate bounding boxes,
#                        shape (batch_size, num_rois, 4)
#                        Each region of interest is defined by four relative 
#                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
#            # Output
#                pooled_areas -- Tensor with the pooled region of interest, shape
#                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
#         """
#         def curried_pool_rois(x): 
#            return ROIPoolingLayer._pool_rois(x[0], x[1], 
#                                            self.pooled_width, 
#                                            self.pooled_height)
#        
#        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

#        return pooled_areas
#    
#      @staticmethod
#      def _pool_rois(feature_map, rois, pooled_width, pooled_height):
#         """ Applies ROI pooling for a single image and varios ROIs
#         """
#         def curried_pool_roi(roi): 
#            return ROIPoolingLayer._pool_roi(feature_map, roi, 
#                                        pooled_width, pooled_height)

#         pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
#         return pooled_areas
#    
#      @staticmethod
#      def _pool_roi(feature_map, roi, pooled_width, pooled_height):
#         """ Applies ROI pooling to a single image and a single region of interest
#         """

#         # Compute the region of interest        
#         feature_map_height = int(feature_map.shape[0])
#         feature_map_width  = int(feature_map.shape[1])

#         x_start = tf.cast(feature_map_width  * roi[0], 'int32')
#         y_start = tf.cast(feature_map_height * roi[1], 'int32')
#         x_end   = tf.cast(feature_map_width  * roi[2], 'int32')
#         y_end   = tf.cast(feature_map_height * roi[3], 'int32')

#         region = feature_map[y_start:y_end, x_start:x_end, :]

#         # Divide the region into non overlapping areas
#         region_height = y_end - y_start
#         region_width  = x_end - x_start
#         y_step = tf.cast( region_height / pooled_height, 'int32')
#         x_step = tf.cast( region_width  / pooled_width , 'int32')
#        
#        # this just puts all remainder at edge, get better binning!
##        areas = [[(
##                    i*y_step, 
##                    j*x_step, 
##                    (i+1)*y_step if i+1 < pooled_height else region_height, 
##                    (j+1)*x_step if j+1 < pooled_width else region_width
##                   ) 
##                   for j in range(pooled_width)] 
##                  for i in range(pooled_height)]

#         numpixels_x = K.get_value(region_width)
#         nbins_x = pooled_width
#         x_inds = [0]*(nbins_x)
#         overhang_x  = numpixels_x % nbins_x
#         normal_bin_width_x = numpixels_x // nbins_x
#         # (nbins - overhang) * normal_bin_width + overhang*(normal_bin_width + 1) == numpixels
#         cw = normal_bin_width_x
#         for i in range(nbins_x-overhang_x):
#            x_inds[i] = (i*cw, (i+1)*cw)
#         done = (nbins_x-overhang_x)*normal_bin_width_x
#         cw   = normal_bin_width_x + 1
#         for i in range(overhang_x):
#            x_inds[i + nbins_x-overhang_x] = (done + i*cw, done + (i+1)*cw)
#            
#         numpixels_y = K.get_value(region_height)
#         nbins_y = pooled_height
#         y_inds = [0]*(nbins_y)
#         overhang_y  = numpixels_y % nbins_y
#         normal_bin_width_y = numpixels_y // nbins_y
#         cw = normal_bin_width_y
#         for i in range(nbins_y-overhang_y):
#            y_inds[i] = (i*cw, (i+1)*cw)
#         done = (nbins_y-overhang_y)*normal_bin_width_y
#         cw   = normal_bin_width_y + 1
#         for i in range(overhang_y):
#            y_inds[i + nbins_y-overhang_y] = (done + i*cw, done + (i+1)*cw)

#         areas = [[(
#                       xx[0], 
#                       yy[0], 
#                       xx[1], 
#                       yy[1]
#                    ) 
#                      for xx in x_inds] 
#                     for yy in y_inds]
#        
#         # take the maximum of each area and stack the result
#         def pool_area(x): 
#            return K.mean(region[x[1]:x[3], x[0]:x[2], :], axis=[0,1])
#        
#         pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
#         return pooled_features
        
#class LayerNormalization(Layer):
#  """Layer normalization layer (Ba et al., 2016).
#  Normalize the activations of the previous layer for each given example in a
#  batch independently, rather than across a batch like Batch Normalization.
#  i.e. applies a transformation that maintains the mean activation within each
#  example close to 0 and the activation standard deviation close to 1.
#  Arguments:
#    axis: Integer or List/Tuple. The axis that should be normalized
#      (typically the features axis).
#    epsilon: Small float added to variance to avoid dividing by zero.
#    center: If True, add offset of `beta` to normalized tensor.
#        If False, `beta` is ignored.
#    scale: If True, multiply by `gamma`.
#      If False, `gamma` is not used.
#      When the next layer is linear (also e.g. `nn.relu`),
#      this can be disabled since the scaling
#      will be done by the next layer.
#    beta_initializer: Initializer for the beta weight.
#    gamma_initializer: Initializer for the gamma weight.
#    beta_regularizer: Optional regularizer for the beta weight.
#    gamma_regularizer: Optional regularizer for the gamma weight.
#    beta_constraint: Optional constraint for the beta weight.
#    gamma_constraint: Optional constraint for the gamma weight.
#    trainable: Boolean, if `True` the variables will be marked as trainable.
#  Input shape:
#    Arbitrary. Use the keyword argument `input_shape`
#    (tuple of integers, does not include the samples axis)
#    when using this layer as the first layer in a model.
#  Output shape:
#    Same shape as input.
#  References:
#    - [Layer Normalization](https://arxiv.org/abs/1607.06450)
#  """

#  def __init__(self,
#               axis=-1,
#               epsilon=1e-3,
#               center=True,
#               scale=True,
#               beta_initializer='zeros',
#               gamma_initializer='ones',
#               beta_regularizer=None,
#               gamma_regularizer=None,
#               beta_constraint=None,
#               gamma_constraint=None,
#               trainable=True,
#               name=None,
#               **kwargs):
#    super(LayerNormalization, self).__init__(
#        name=name, trainable=trainable, **kwargs)
#    if isinstance(axis, (list, tuple)):
#      self.axis = axis[:]
#    elif isinstance(axis, int):
#      self.axis = axis
#    else:
#      raise ValueError('Expected an int or a list/tuple of ints for the '
#                       'argument \'axis\', but received instead: %s' % axis)

#    self.epsilon = epsilon
#    self.center = center
#    self.scale = scale
#    self.beta_initializer = initializers.get(beta_initializer)
#    self.gamma_initializer = initializers.get(gamma_initializer)
#    self.beta_regularizer = regularizers.get(beta_regularizer)
#    self.gamma_regularizer = regularizers.get(gamma_regularizer)
#    self.beta_constraint = constraints.get(beta_constraint)
#    self.gamma_constraint = constraints.get(gamma_constraint)

#    self.supports_masking = True

#    # Indicates whether a faster fused implementation can be used. This will be
#    # set to True or False in build()"
#    self._fused = None

#  def _fused_can_be_used(self, ndims):
#    """Return false if fused implementation cannot be used.
#    Check if the axis is contiguous and can be collapsed into the last axis.
#    The self.axis is assumed to have no duplicates.
#    """
#    axis = sorted(self.axis)
#    can_use_fused = False

#    if axis[-1] == ndims - 1 and axis[-1] - axis[0] == len(axis) - 1:
#      can_use_fused = True

#    # fused_batch_norm will silently raise epsilon to be at least 1.001e-5, so
#    # we cannot used the fused version if epsilon is below that value. Also, the
#    # variable dtype must be float32, as fused_batch_norm only supports float32
#    # variables.
#    if self.epsilon < 1.001e-5:
#      can_use_fused = False

#    return can_use_fused

#  def build(self, input_shape):
#    ndims = len(input_shape)
#    if ndims is None:
#      raise ValueError('Input shape %s has undefined rank.' % input_shape)

#    # Convert axis to list and resolve negatives
#    if isinstance(self.axis, int):
#      self.axis = [self.axis]
#    elif isinstance(self.axis, tuple):
#      self.axis = list(self.axis)
#    for idx, x in enumerate(self.axis):
#      if x < 0:
#        self.axis[idx] = ndims + x

#    # Validate axes
#    for x in self.axis:
#      if x < 0 or x >= ndims:
#        raise ValueError('Invalid axis: %d' % x)
#    if len(self.axis) != len(set(self.axis)):
#      raise ValueError('Duplicate axis: {}'.format(tuple(self.axis)))

#    param_shape = [input_shape[dim] for dim in self.axis]
#    if self.scale:
#      self.gamma = self.add_weight(
#          name='gamma',
#          shape=param_shape,
#          initializer=self.gamma_initializer,
#          regularizer=self.gamma_regularizer,
#          constraint=self.gamma_constraint,
#          trainable=True)
#    else:
#      self.gamma = None

#    if self.center:
#      self.beta = self.add_weight(
#          name='beta',
#          shape=param_shape,
#          initializer=self.beta_initializer,
#          regularizer=self.beta_regularizer,
#          constraint=self.beta_constraint,
#          trainable=True)
#    else:
#      self.beta = None

#    self._fused = self._fused_can_be_used(ndims)

#    self.built = True

#  def call(self, inputs):
#    # Compute the axes along which to reduce the mean / variance
#    input_shape = inputs.shape
#    ndims = len(input_shape)

#    # Broadcasting only necessary for norm where the axis is not just
#    # the last dimension
#    broadcast_shape = [1] * ndims
#    for dim in self.axis:
#      broadcast_shape[dim] = input_shape.dims[dim].value
#    def _broadcast(v):
#      if (v is not None and len(v.shape) != ndims and
#          self.axis != [ndims - 1]):
#        return array_ops.reshape(v, broadcast_shape)
#      return v

#    if not self._fused:
#      # Calculate the moments on the last axis (layer activations).
#      mean, variance = nn.moments(inputs, self.axis, keep_dims=True)

#      scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

#      # Compute layer normalization using the batch_normalization function.
#      outputs = nn.batch_normalization(
#          inputs,
#          mean,
#          variance,
#          offset=offset,
#          scale=scale,
#          variance_epsilon=self.epsilon)
#    else:
#      # Collapse dims before self.axis, and dims in self.axis
#      pre_dim, in_dim = (1, 1)
#      axis = sorted(self.axis)
#      tensor_shape = array_ops.shape(inputs)
#      for dim in range(0, ndims):
#        dim_tensor = tensor_shape[dim]
#        if dim < axis[0]:
#          pre_dim = pre_dim * dim_tensor
#        else:
#          assert dim in axis
#          in_dim = in_dim * dim_tensor

#      squeezed_shape = [1, pre_dim, in_dim, 1]
#      # This fused operation requires reshaped inputs to be NCHW.
#      data_format = 'NCHW'

#      inputs = array_ops.reshape(inputs, squeezed_shape)

#      def _set_const_tensor(val, dtype, shape):
#        return array_ops.fill(shape, constant_op.constant(val, dtype=dtype))

#      # self.gamma and self.beta have the wrong shape for fused_batch_norm, so
#      # we cannot pass them as the scale and offset parameters. Therefore, we
#      # create two constant tensors in correct shapes for fused_batch_norm and
#      # later contuct a separate calculation on the scale and offset.
#      scale = _set_const_tensor(1.0, inputs.dtype, [pre_dim])
#      offset = _set_const_tensor(0.0, inputs.dtype, [pre_dim])

#      # Compute layer normalization using the fused_batch_norm function.
#      outputs, _, _ = nn.fused_batch_norm(
#          inputs,
#          scale=scale,
#          offset=offset,
#          epsilon=self.epsilon,
#          data_format=data_format)

#      outputs = array_ops.reshape(outputs, tensor_shape)

#      scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

#      if scale is not None:
#        outputs = outputs * scale
#      if offset is not None:
#        outputs = outputs + offset

#    # If some components of the shape got lost due to adjustments, fix that.
#    outputs.set_shape(input_shape)

#    return outputs

#  def compute_output_shape(self, input_shape):
#    return input_shape

#  def get_config(self):
#    config = {
#        'axis': self.axis,
#        'epsilon': self.epsilon,
#        'center': self.center,
#        'scale': self.scale,
#        'beta_initializer': initializers.serialize(self.beta_initializer),
#        'gamma_initializer': initializers.serialize(self.gamma_initializer),
#        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
#        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
#        'beta_constraint': constraints.serialize(self.beta_constraint),
#        'gamma_constraint': constraints.serialize(self.gamma_constraint)
#    }
#    base_config = super(LayerNormalization, self).get_config()
#    return dict(list(base_config.items()) + list(config.items()))

