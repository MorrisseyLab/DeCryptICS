#!/usr/bin/env python3
# Taken from https://medium.com/xplore-ai/implementing-attention-in-tensorflow-keras-using-roi-pooling-992508b6592b
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from bbox import single_xcycwh_to_xy_min_xy_max


class ROIPoolingLayer(Layer):
    """ Implements Region Of Interest Max Pooling 
        for channel-first images and relative bounding box coordinates
        
        # Constructor parameters
            pooled_height, pooled_width (int) -- 
              specify height and width of layer outputs
        
        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]
           
        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)
    
    """
    def __init__(self, pooled_height, pooled_width):
        super(ROIPoolingLayer, self).__init__()
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
          
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width,
        })
        return config
        
    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height, 
                self.pooled_width, n_channels)
    
    def cast_inputs(self, inputs):
        # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output
        
            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four
                        coordinates (x_c, y_c, w, h) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x): 
          return ROIPoolingLayer._pool_rois(x[0], x[1], 
                                            self.pooled_height, 
                                            self.pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_rois, x, fn_output_signature=tf.float16) #float32

        return pooled_areas
    
    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi): 
          return ROIPoolingLayer._pool_roi(feature_map, roi, 
                                           pooled_height, pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_roi, rois, fn_output_signature=tf.float16) #float32
        return pooled_areas
    
    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """
        epxnd_feature_map = tf.expand_dims(feature_map, 0)
        
        roi = single_xcycwh_to_xy_min_xy_max(tf.dtypes.cast(roi, tf.float32))
#        roi = tf.clip_by_value(roi, clip_value_min=0, clip_value_max=1)
        epxnd_roi = tf.expand_dims([roi[0], roi[1], roi[2], roi[3]], 0)
        pooled_features = tf.image.crop_and_resize(epxnd_feature_map, epxnd_roi,
                                                [0], (pooled_height, pooled_width))
        # Try casting
        pooled_features = tf.dtypes.cast(pooled_features, tf.float16)
            
        return pooled_features[0]
    
#Implement the patch encoding layer
class PatchEncoder_w_position(layers.Layer):
    def __init__(self, num_patches, projection_dim, fmap_dim):
        super(PatchEncoder_w_position, self).__init__()
        self.num_patches = num_patches
        self.fmap_dim    = fmap_dim
        self.projection  = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=fmap_dim*fmap_dim + 1, output_dim=projection_dim
        )
    
    def cast_inputs(self, inputs):
         # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)

    def call(self, x):
        patch, bbox = x
        patch = layers.Reshape((self.num_patches ,-1))(patch)
        # (0,1) convert to (0,127)
        bbox = (self.fmap_dim-1)*bbox
        # Map to flattened indices
        positions = bbox[:,:,0] + self.fmap_dim*bbox[:,:,1]
        positions = tf.cast(positions, tf.int32)
        encoded   = self.position_embedding(positions)
        encoded   = self.projection(patch) + self.position_embedding(positions)
        return encoded
    




    
