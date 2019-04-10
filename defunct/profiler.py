#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 18:07:07 2018

@author: doran
"""
from DNN_segment import predict_svs_slide
file_name = '/home/doran/Work/images/KDM6A_March2018/raw_images/642739.svs'
folder_to_analyse = '/home/doran/Work/images/KDM6A_March2018/Analysed_slides/Analysed_642739/'
clonal_mark_type = 'N-N'
predict_svs_slide(file_name, folder_to_analyse, clonal_mark_type, prob_thresh = 0.5, find_clones = True)