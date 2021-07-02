import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

## github: twairball/keras-subpixel-conv
class SubpixelConv2D(Layer):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    def __init__(self, scale):
        super(SubpixelConv2D, self).__init__()
        self.scale = scale
            
    def compute_output_shape(self, input_shape):
        """ Returns the shape of the pixel-shuffled output
        """
        dims = [input_shape[0],
                input_shape[1] * self.scale,
                input_shape[2] * self.scale,
                int(input_shape[3] / (self.scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    
    def cast_inputs(self, inputs):
        # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)
        
    def call(self, x):
        return tf.nn.depth_to_space(x, self.scale)
