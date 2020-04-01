class ROIPoolingLayer(Layer):
    """ Implements Region Of Interest Max Pooling 
        for channel-last images and relative bounding box coordinates
        
        # Constructor parameters
            pooled_width, pooled_height (int) -- 
              specify width and height of layer outputs
        
        Shape of inputs
            [(batch_size, pooled_width, pooled_height, n_channels),
             (batch_size, num_rois, 4)]
           
        Shape of output
            (batch_size, num_rois, pooled_width, pooled_height, n_channels)
    
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
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_width, 
                self.pooled_height, n_channels)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output
        
            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative 
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x): 
          return ROIPoolingLayer._pool_rois(x[0], x[1], 
                                            self.pooled_width, 
                                            self.pooled_height)
        
        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

        return pooled_areas
    
    @staticmethod
    def _pool_rois(feature_map, rois, pooled_width, pooled_height):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi): 
          return ROIPoolingLayer._pool_roi(feature_map, roi, 
                                           pooled_width, pooled_height)
        
        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas




b,m = valid_datagen[0]
b[1].shape
ROI = b[1][0,0,:]
roi = K.constant(ROI)

img = b[0][0,:,:,:]
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
feature_map = K.constant(img)

pooled_height = 7
pooled_width = 7
    
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

         # Divide the region into non overlapping areas
         region_height = y_end - y_start
         region_width  = x_end - x_start
         y_step = tf.cast( region_height / pooled_height, 'int32')
         x_step = tf.cast( region_width  / pooled_width , 'int32')
        
        # this just puts all remainder at edge, get better binning!
#        areas = [[(
#                    i*y_step, 
#                    j*x_step, 
#                    (i+1)*y_step if i+1 < pooled_height else region_height, 
#                    (j+1)*x_step if j+1 < pooled_width else region_width
#                   ) 
#                   for j in range(pooled_width)] 
#                  for i in range(pooled_height)]

         numpixels_x = K.get_value(region_width)
         nbins_x = pooled_width
         x_inds = [0]*(nbins_x)
         overhang_x  = numpixels_x % nbins_x
         normal_bin_width_x = numpixels_x // nbins_x
         # (nbins - overhang) * normal_bin_width + overhang*(normal_bin_width + 1) == numpixels
         cw = normal_bin_width_x
         for i in range(nbins_x-overhang_x):
            x_inds[i] = (i*cw, (i+1)*cw)
         done = (nbins_x-overhang_x)*normal_bin_width_x
         cw   = normal_bin_width_x + 1
         for i in range(overhang_x):
            x_inds[i + nbins_x-overhang_x] = (done + i*cw, done + (i+1)*cw)
            
         numpixels_y = K.get_value(region_height)
         nbins_y = pooled_height
         y_inds = [0]*(nbins_y)
         overhang_y  = numpixels_y % nbins_y
         normal_bin_width_y = numpixels_y // nbins_y
         cw = normal_bin_width_y
         for i in range(nbins_y-overhang_y):
            y_inds[i] = (i*cw, (i+1)*cw)
         done = (nbins_y-overhang_y)*normal_bin_width_y
         cw   = normal_bin_width_y + 1
         for i in range(overhang_y):
            y_inds[i + nbins_y-overhang_y] = (done + i*cw, done + (i+1)*cw)

         areas = [[(
                       xx[0], 
                       yy[0], 
                       xx[1], 
                       yy[1]
                    ) 
                      for xx in x_inds] 
                     for yy in y_inds]
        
         # take the maximum of each area and stack the result
         def pool_area(x): 
          return K.mean(region[x[1]:x[3], x[0]:x[2], :], axis=[0,1])

         pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
         return pooled_features

