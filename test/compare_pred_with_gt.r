library(tidyverse)

res = read_csv("~/Work/py_code/new_DeCryptICS/newfiles/test_output_batched.csv")
pl1 = res %>% ggplot() + geom_point(aes(mask_x, -mask_y, col=p_clone), size=0.7) + scale_color_viridis_c()


gt = read_delim("~/Work/images/Leeds_May2019/curated_cryptdata/train/KM3/Analysed_slides/Analysed_KM3M_428042/crypt_network_data.txt", delim = "\t")
pl2 = gt %>% ggplot() + geom_point(aes(x, -y, col=mutant), size=0.7) + scale_color_viridis_c()
gridExtra::grid.arrange(pl1, pl2)
