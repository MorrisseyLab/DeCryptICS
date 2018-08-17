#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:15:00 2018

@author: doran
"""
from Bayes_segment import GetThresholdsPrepareRun, SegmentFromFolder
from DNN_segment  import predict_svs_slide
from deconv_mat   import *

def GetThresholdsPrepareRun_gen(folder_in, file_in, folder_out, clonal_mark_type):
     GetThresholdsPrepareRun(folder_in, file_in, folder_out, clonal_mark_type)

def SegmentFromFolder_wrapper(folder_name, clonal_mark_type, find_clones = False):
    SegmentFromFolder(folder_name, clonal_mark_type, find_clones)

def predict_svs_slide_DNN(filename, folder_out, clonal_mark_type, find_clones = False, prob_thresh = 0.5):
    predict_svs_slide(filename, folder_out, clonal_mark_type, find_clones, prob_thresh)
