library(tidyverse)

dat = read_tsv("./performance_database.csv")

dat %>% gather(errtype, value, -Image_path) %>% filter(value>0) %>% 
   ggplot() + geom_histogram(aes(x=value), bins=60) + facet_wrap(~errtype) + xlim(0,5000)

dat2 = dat %>% gather(errtype, value, -Image_path) %>% filter(value>0)
dat2 %>% filter(value>600) %>% select(Image_path) %>% distinct() %>% write_tsv("./bad_tiles_need_check.tsv")
