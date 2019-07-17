from DNN.u_net import get_unet_256_for_X

# Reduce input_size_run if GPU is running out of memory (increase if not!)
input_size_run = 1024
input_size_train = 256

max_epochs = 10000
batch_size = 13

model_factory = get_unet_256_for_X
