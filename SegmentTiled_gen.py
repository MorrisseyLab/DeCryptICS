#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:15:00 2018

@author: doran
"""
from Clonal_Stains.SegmentTiled_MPAS import GetThresholdsPrepareRun_MPAS, SegmentFromFolder_MPAS
from Clonal_Stains.SegmentTiled_MAOA import GetThresholdsPrepareRun_MAOA, SegmentFromFolder_MAOA
from Clonal_Stains.SegmentTiled_KDM6A import GetThresholdsPrepareRun_KDM6A, SegmentFromFolder_KDM6A
from Clonal_Stains.SegmentTiled_NONO import GetThresholdsPrepareRun_NONO, SegmentFromFolder_NONO
from Clonal_Stains.SegmentTiled_STAG import GetThresholdsPrepareRun_STAG, SegmentFromFolder_STAG

# All these functions, and the main Segment() functions
# should be made general for implementation of different
# clonal marks. Or, if greater variation of implementation
# is required, make copies of all relevant functions and
# write a wrapper function with a "clonal mark" argument
# that then calls the correct analysis function.

def GetThresholdsPrepareRun(folder_in, file_in, folder_out, clonal_mark):
    if (clonal_mark=="mPAS"):
        GetThresholdsPrepareRun_MPAS(folder_in, file_in, folder_out)
    if (clonal_mark=="MAOA"):
        GetThresholdsPrepareRun_MAOA(folder_in, file_in, folder_out)
    if (clonal_mark=="STAG"):
        GetThresholdsPrepareRun_STAG(folder_in, file_in, folder_out)
    if (clonal_mark=="KDM6A"):
        GetThresholdsPrepareRun_KDM6A(folder_in, file_in, folder_out)
    if (clonal_mark=="NONO"):
        GetThresholdsPrepareRun_NONO(folder_in, file_in, folder_out)


def SegmentFromFolder(folder_name, clonal_mark):
    if (clonal_mark=="mPAS"):
        SegmentFromFolder_MPAS(folder_name)
    if (clonal_mark=="MAOA"):
        SegmentFromFolder_MAOA(folder_name)
    if (clonal_mark=="STAG"):
        SegmentFromFolder_STAG(folder_name)
    if (clonal_mark=="KDM6A"):
        SegmentFromFolder_KDM6A(folder_name)
    if (clonal_mark=="NONO"):
        SegmentFromFolder_NONO(folder_name)
    