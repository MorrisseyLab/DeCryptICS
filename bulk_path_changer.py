#!/usr/bin/env python3
import sys, os, glob
from path_changer import change_paths

targets = glob.glob('/home/doran/Work/images/Leeds_May2019/splitbyKM/KM*')
base_newpath = 'N:\\Faculty-of-Medicine-and-Health\\LICAP\\DATA\\PTHY\\Pathology\\Labwork\\Kate Sutton\\ACF\\mnm\\Analysis\\Research_5_LIMM_K_M\\'

for ff in targets:
   km_num = ff.split('KM')[-1]
   this_newpath = base_newpath + 'KM' + km_num + '\\'
   this_projfolder = ff + '/block_analysis/'
   change_paths(this_projfolder, this_newpath)
