from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf

MASK_VALUE = -1

def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function

def masked_accuracy(y_true, y_pred, mask_value=MASK_VALUE):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    total = K.sum(y_true * mask)
    correct = K.sum(K.equal(y_true * mask, K.round(y_pred) * mask))
    return correct / total

def masked_dice_coeff(y_true, y_pred, mask_value=MASK_VALUE):
    smooth = 1.
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    y_true_f = K.flatten(y_true * mask)
    y_pred_f = K.flatten(y_pred * mask)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score

def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss
    
def masked_dice_coeff_perchannel1(y_true, y_pred, mask_value=MASK_VALUE):
   y_true = y_true[:,:,:,0]
   y_pred = y_pred[:,:,:,0]
   smooth = 1.
   mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
   y_true_f = K.flatten(y_true * mask)
   y_pred_f = K.flatten(y_pred * mask)
   intersection = K.sum(y_true_f * y_pred_f)
   score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
   return score
   
def masked_dice_coeff_perchannel2(y_true, y_pred, mask_value=MASK_VALUE):
   y_true = y_true[:,:,:,1]
   y_pred = y_pred[:,:,:,1]
   smooth = 1.
   mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
   y_true_f = K.flatten(y_true * mask)
   y_pred_f = K.flatten(y_pred * mask)
   intersection = K.sum(y_true_f * y_pred_f)
   score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
   return score
   
def masked_dice_coeff_perchannel3(y_true, y_pred, mask_value=MASK_VALUE):
   y_true = y_true[:,:,:,2]
   y_pred = y_pred[:,:,:,2]
   smooth = 1.
   mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
   y_true_f = K.flatten(y_true * mask)
   y_pred_f = K.flatten(y_pred * mask)
   intersection = K.sum(y_true_f * y_pred_f)
   score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
   return score

def masked_dice_coeff_perchannel4(y_true, y_pred, mask_value=MASK_VALUE):
   y_true = y_true[:,:,:,3]
   y_pred = y_pred[:,:,:,3]
   smooth = 1.
   mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
   y_true_f = K.flatten(y_true * mask)
   y_pred_f = K.flatten(y_pred * mask)
   intersection = K.sum(y_true_f * y_pred_f)
   score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
   return score

def masked_dice_coeff_perchannel5(y_true, y_pred, mask_value=MASK_VALUE):
   y_true = y_true[:,:,:,4]
   y_pred = y_pred[:,:,:,4]
   smooth = 1.
   mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
   y_true_f = K.flatten(y_true * mask)
   y_pred_f = K.flatten(y_pred * mask)
   intersection = K.sum(y_true_f * y_pred_f)
   score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
   return score
   
def masked_dice_coeff_perchannel6(y_true, y_pred, mask_value=MASK_VALUE):
   y_true = y_true[:,:,:,5]
   y_pred = y_pred[:,:,:,5]
   smooth = 1.
   mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
   y_true_f = K.flatten(y_true * mask)
   y_pred_f = K.flatten(y_pred * mask)
   intersection = K.sum(y_true_f * y_pred_f)
   score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
   return score

def masked_dice_coeff_perchannel7(y_true, y_pred, mask_value=MASK_VALUE):
   y_true = y_true[:,:,:,6]
   y_pred = y_pred[:,:,:,6]
   smooth = 1.
   mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
   y_true_f = K.flatten(y_true * mask)
   y_pred_f = K.flatten(y_pred * mask)
   intersection = K.sum(y_true_f * y_pred_f)
   score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
   return score
    
 

