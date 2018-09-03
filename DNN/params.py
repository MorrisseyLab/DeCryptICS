from DNN.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_256_for_X

# Reduce input_size if GPU is running out of memory (increase if not!)
input_size = 256

max_epochs = 5000
batch_size = 18

orig_width = input_size
orig_height = input_size

threshold = 0.5

model_factory = get_unet_256_for_X
