
def set_params():
    chan_num = 3
    normalize = True
    dilate_masks = True
    reset_binaries = False
    shuffle = True
    normalize = True
    aug = True   
    cpfr_fracs = [1,2,1,1] # clone, partial, fufi, random
    tilesize_train = 1024
    input_shape_train = (tilesize_train, tilesize_train, 3)
    batch_size = 8
    umpp = 1.1
    stride_bool = True
    num_bbox = 400
    crypt_class = False

    params = {'tilesize_train' : tilesize_train,
               'input_shape_train' : input_shape_train,
               'batch_size' : batch_size,
               'umpp' : umpp,
               'dilate_masks' : dilate_masks,
               'cpfr_frac' : cpfr_fracs,
               'aug' : aug,
               'shuffle' : shuffle,
               'reset_binaries' : reset_binaries,
               'normalize' : normalize,
               'stride_bool' : stride_bool,
               'chan_num' : chan_num,
               'num_bbox': num_bbox,
               'crypt_class': crypt_class}
                  
    return params

