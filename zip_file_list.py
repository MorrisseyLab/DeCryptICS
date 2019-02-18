#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:11:01 2019

@author: doran
"""

import glob, os, sys
import numpy as np
import pandas as pd
import argparse
import zipfile
from random import getrandbits

if (len(sys.argv) > 2):
        sys.stderr.write('Only one argument required (the full path to folder containing image files)\nRunning with no arguments uses the current working directory.\n')
        sys.exit(1)
        
if __name__=="__main__":

   if len(sys.argv)>1:
      print("Zipping files listed in %s" % sys.argv[1])

      ## Find output folder
      input_file = os.path.abspath(sys.argv[1])
      linux_test = len(input_file.split('/'))
      windows_test = len(input_file.split('\\'))
      if (linux_test==1 and windows_test>1):
         base_path = '\\' + os.path.join(*input_file.split('\\')[:-1]) + '\\'
      if (linux_test>1 and windows_test==1):
         base_path = '/' + os.path.join(*input_file.split('/')[:-1]) + '/'
      if (linux_test==1 and windows_test==1):
         base_path = os.getcwd() + '/'
    
      ## Read input file
      ftype = input_file.split('.')[-1]
      
      ext1 = "svs"; ext2 = "svs"
      if (input_file.split('_')[-1].split('.')[0] == "tif"):
         ext1 = "tif"; ext2 = "tiff"
      if (input_file.split('_')[-1].split('.')[0] == "png"):
         ext1 = "png"; ext2 = "png"
      if (input_file.split('_')[-1].split('.')[0] == "jpg"):
         ext1 = "jpg"; ext2 = "jpeg"

      # check if we loaded a value as a header
      if (ftype=="csv"):      a = pd.read_csv(input_file)
      elif (ftype[:2]=="xl"): a = pd.read_excel(input_file)
      else:                   a = pd.read_table(input_file)
      heads = list(a.columns.values)
      img_sum = 0
      for hh in heads:
         if (hh.split('.')[-1]==ext1 or hh.split('.')[-1]==ext2): img_sum += 1
      if img_sum>0:
         if (ftype=="csv"):      a = pd.read_csv(input_file  , header=None)
         elif (ftype[:2]=="xl"): a = pd.read_excel(input_file, header=None)
         else:                   a = pd.read_table(input_file, header=None)
      in_shape = a.shape
      a = np.asarray(a).reshape(in_shape)
      
      # extract file paths
      if len(a.shape)>1: # 2D input
         ncols = a.shape[1]
         for i in range(ncols):
            if type(a[0,i]) == str:
               if (a[0,i].split('.')[-1]==ext1 or a[0,i].split('.')[-1]==ext2):
                  pathind = i
         full_paths = list(a[:,pathind]) # take correct column
      else: # 1D input
         ncols = a.shape[0]         
         img_sum = 0
         for i in range(ncols):
            if type(a[i]) == str:
               if (a[i].split('.')[-1]==ext1 or a[i].split('.')[-1]==ext2):
                  pathind = i
                  img_sum += 1                  
         if (img_sum==1):
            full_paths = list(a[pathind]) # take one entry
         if (img_sum==ncols):
            full_paths = list(a) # take all entries
           
      randbits = str(getrandbits(8 * 4))
      outfile = base_path + 'images_' + randbits + '.zip'
      randbits = print("Creating zip file at %s" % outfile)
      with zipfile.ZipFile(outfile, 'w') as myzip:
         for f in full_paths:   
            myzip.write(f)
            
            
            
            
            
            
