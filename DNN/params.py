from DNN.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_1536, get_unet_2048 , get_unet_4096 , get_unet_256_for_2048

input_size = 2048

max_epochs = 300
batch_size = 18

orig_width = 2048
orig_height = 2048

threshold = 0.5

model_factory = get_unet_256_for_2048
