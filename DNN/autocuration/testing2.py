import tensorflow as tf
# declare variables

pooled_width = 7
pooled_height = 7
   
    
def curried_pool_rois(x): 
   return _pool_roi(x[0], x[1], pooled_width, pooled_height)

pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

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
