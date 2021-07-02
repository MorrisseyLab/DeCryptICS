#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:01:29 2018

@author: doran
"""
import os, time, glob, fileinput
import pandas as pd
import numpy as np
import csv
from MiscFunctions import mkdir_p

def create_qupath_project(path_to_project, full_paths, file_in, folder_out, decryptics_folder):    
   num_to_run = len(file_in)
   # Create directory and essential sub-directories
   mkdir_p(path_to_project)
   mkdir_p(path_to_project+"/data")
   mkdir_p(path_to_project+"/thumbnails")
   mkdir_p(path_to_project+"/scripts")
   
   # Write project file
   with open(path_to_project+"/project.qpproj", 'w') as file:
      file.write('{' + '\n')
      file.write('\t'+"\"createTimestamp\": %d,\n" % int(time.time()))
      file.write('\t'+"\"modifyTimestamp\": %d,\n" % int(time.time()))
      file.write('\t'+"\"images\": [" + '\n')
      if (num_to_run>1):
         for i in range(num_to_run-1):
            file.write('\t'+'\t'+'{' + '\n')
            file.write('\t'+'\t'+'\t'+"\"path\": \"%s\",\n" % full_paths[i])
            file.write('\t'+'\t'+'\t'+"\"name\": \"%s\"\n" % file_in[i])
            file.write('\t'+'\t'+"},\n")
      i = num_to_run-1 # pick out last image to remove trailing comma
      file.write('\t'+'\t'+'{'+'\n')
      file.write('\t'+'\t'+'\t'+"\"path\": \"%s\",\n" % full_paths[i])
      file.write('\t'+'\t'+'\t'+"\"name\": \"%s\"\n" % file_in[i])
      file.write('\t'+'\t'+"}\n")
      file.write('\t'+']'+'\n')
      file.write('}'+'\n')
      
   # Write script file
   lines = []
   target = 'BASEPATH_PLACEHOLDER'
   with open(decryptics_folder+'/load_contours_template.groovy', 'r') as f:
      line = 1
      while line:
         line = f.readline()
         if (target in line): line = line.replace(target, '"'+folder_out+'"')
         lines.append(line)
   with open(path_to_project+"/scripts/load_contours.groovy", 'w') as fo:
      for line in lines: fo.write(line)
      
        
