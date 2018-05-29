#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:15:00 2018

@author: doran
"""
from SegmentTiled import GetThresholdsPrepareRun, SegmentFromFolder
from DNN_segment  import predict_svs_slide
from deconv_mat   import *

# All these functions, and the main Segment() functions
# should be made general for implementation of different
# clonal marks. Or, if greater variation of implementation
# is required, make copies of all relevant functions and
# write a wrapper function with a "clonal mark" argument
# that then calls the correct analysis function.

def GetThresholdsPrepareRun_gen(folder_in, file_in, folder_out, clonal_mark_type):
     GetThresholdsPrepareRun(folder_in, file_in, folder_out, clonal_mark_type)

def SegmentFromFolder_wrapper(folder_name, clonal_mark_type):
    SegmentFromFolder(folder_name, clonal_mark_type)

def predict_svs_slide_DNN(filename, folder_out, clonal_mark_type, prob_thresh = 0.5):
    predict_svs_slide(filename, folder_out, clonal_mark_type, prob_thresh)
