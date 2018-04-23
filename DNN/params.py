from DNN.model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_4096 

input_size = 1024 #4096 #1024

max_epochs = 8
batch_size = 4

orig_width = 1024 #4096 #1024
orig_height = 1024 #4096 #1024

threshold = 0.5

model_factory = get_unet_1024 #get_unet_4096
