#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:18:17 2018

@author: doran
"""

import glob, os, sys
import numpy as np
import pandas as pd
import argparse, datetime
from qupath_project import create_qupath_project, extract_counts_csv, file_len, folder_from_image

def run_analysis():
   parser = argparse.ArgumentParser(description = "This script takes as input a list of full paths (local or remote) to .svs files, "
                                                   "or a larger dataframe that contains such a list as a column. "
                                                   "The output is a QuPath project containing all .svs slides. "
                                                   "Optionally, the .svs slides can be analysed to count crypts and clones, "
                                                   "or crypts only; detected contours are added to the QuPath project. " )
    
   parser.add_argument("action", choices = ["read" , "count"], 
                                 default = "read", 
                                 help = "Action to carry out. "
                                        "read: create QuPath project for input slides; "
                                        "count: counts crypts (and possibly clones) and creates QuPath project. ")

   parser.add_argument("input_file", help = "A file containing a list of full paths to slides (or array with column acting as said list). ")

   parser.add_argument('-q', action = "store",
                        dest = "qp_proj_name", 
                        default = "qupath_project_"+datetime.datetime.now().strftime("%d-%m-%Y_%H-%M"),
                        help = "Optionally set the name of the QuPath project folder to be created. "
                               "Defaults to 'qupath_project_DATE_TIME'. ")
                       
   parser.add_argument('-c', choices = ["None" , "P-N" , "P-L" , "P-B" , "N-N" , "N-L" , "N-B"],
                             default = "None", 
                             dest    = "clonal_mark",
                             help    = "Clonal mark type, if clones are to be counted. "
                                       "Input format is X-Y, where: "
                                       "X = P (positive) / N (negative); and "
                                       "Y = N (nuclear) / L (lumen) / B (both). "
                                       "For example, mPAS is a positive stain expressed in the cytoplasm and "
                                       "should be indicated as P-L, whereas MAOA is a negative stain that appears "
                                       "in the nucleus and cytoplasm and should be indicated as N-B. "
                                       "Defaults to None if -c is not passed, meaning clone finding will not be performed. "
                                       "Note: all slides in input list will be analysed using the same clonal mark type. ")

   parser.add_argument('-m', choices = ["D" , "B"],
                             default = "D", 
                             dest    = "method",
                             help    = "Method of crypt finding: D uses a deep neural network, B uses a Bayesian model (default is D). ")

   parser.add_argument('-r', action = "store_true", 
                             default = False,
                             help = "Forces repeat analysis of slides with existing crypt contour files. "
                                    "Defaults to False if -r flag missing. ")


   args = parser.parse_args()
   ## check args
   print("Running with the following inputs:")
   print('action       = {!r}'.format(args.action))
   print('input_file   = {!r}'.format(args.input_file))
   print('qp_proj_name = {!r}'.format(args.qp_proj_name))
   print('clonal_mark  = {!r}'.format(args.clonal_mark))
   print('method       = {!r}'.format(args.method))
   print('force_repeat = {!r}'.format(args.r))
   print("\n...Working...\n")
   
   ## Standardise clonal mark type string
   clonal_mark_type = args.clonal_mark
   find_clones = False
   if (clonal_mark_type.upper()=="P-N"): clonal_mark_type="P-N"
   if (clonal_mark_type.upper()=="P-L"): clonal_mark_type="P-L"
   if (clonal_mark_type.upper()=="P-B"): clonal_mark_type="P-B"
   if (clonal_mark_type.upper()=="N-N"): clonal_mark_type="N-N"
   if (clonal_mark_type.upper()=="N-L"): clonal_mark_type="N-L"
   if (clonal_mark_type.upper()=="N-B"): clonal_mark_type="N-B"
   if (clonal_mark_type!="None"): find_clones = True

   ## Standardise method string
   method = args.method
   if (method.upper()=="D"): method="D"
   if (method.upper()=="B"): method="B"

   ## Find output folder
   input_file = os.path.abspath(args.input_file)
   linux_test = len(input_file.split('/'))
   windows_test = len(input_file.split('\\'))
   if (linux_test==1 and windows_test>1):
      base_path = '/' + os.path.join(*input_file.split('\\')[:-1]) + '/'
   if (linux_test>1 and windows_test==1):
      base_path = '/' + os.path.join(*input_file.split('/')[:-1]) + '/'
   if (linux_test==1 and windows_test==1):
      base_path = os.getcwd() + '/'
 
   ## Read input file
   ftype = input_file.split('.')[-1]   

   # check if we loaded a value as a header
   if (ftype=="csv"):                       a = pd.read_csv(input_file)
   if (ftype[:2]=="xl"): a = pd.read_excel(input_file)
   else:                                    a = pd.read_table(input_file)
   heads = list(a.columns.values)
   svs_sum = 0
   for hh in heads:
      if (hh.split('.')[-1]=="svs"): svs_sum += 1
   if svs_sum>0:
      if (ftype=="csv"):                       a = pd.read_csv(input_file  , header=None)
      if (ftype[:2]=="xl"): a = pd.read_excel(input_file, header=None)
      else:                                    a = pd.read_table(input_file, header=None)
   a = np.asarray(a)
   
   # extract file paths
   if len(a.shape)>1: # 2D input
      ncols = a.shape[1]   
      for i in range(ncols):
         if type(a[0,i]) == str:
            if (a[0,i].split('.')[-1]=="svs"):
               pathind = i
      full_paths = list(a[:,pathind]) # take correct column
   else: # 1D input
      ncols = a.shape[0]         
      svs_sum = 0
      for i in range(ncols):
         if type(a[i]) == str:
            if (a[i].split('.')[-1]=="svs"):
               pathind = i
               svs_sum += 1                  
      if (svs_sum==1):
         full_paths = list(a[pathind]) # take one entry
      if (svs_sum==ncols):
         full_paths = list(a) # take all entries
   linux_test = len(full_paths[0].split('/'))
   windows_test = len(full_paths[0].split('\\'))
   if (linux_test==1 and windows_test>1):
      file_in = [name.split("\\")[-1].split(".")[0] for name in full_paths]
   if (linux_test>1 and windows_test==1):
      file_in = [name.split("/")[-1].split(".")[0] for name in full_paths]
   
   ## Define file structures
   folder_out = base_path + '/' + "Analysed_slides/" 
   try:
      os.mkdir(folder_out)
   except:
      pass
         
   ## Create QuPath project for the current batch
   qupath_project_path = base_path + '/' + args.qp_proj_name
   create_qupath_project(qupath_project_path, full_paths, file_in, folder_out)
   print("QuPath project created in %s" % qupath_project_path)
     
   ## CHANGE BELOW HERE
   if args.action == "count":
      from SegmentTiled_gen import GetThresholdsPrepareRun, SegmentFromFolder, predict_svs_slide_DNN
      num_to_run = len(file_in)
      folders_to_analyse = [folder_out+fldr for fldr in ["Analysed_"+fnum+"/" for fnum in file_in]]

      ## Perform crypt segmentation
      ######################################
      if (method == "D"):
         for i in range(num_to_run):
            if (os.path.isfile(folders_to_analyse[i]+"crypt_contours.txt") and args.r==False):         
               print("Passing on %s, image previously analysed." % folders_to_analyse[i])
               pass
            else:
               print("Beginning segmentation on %s." % folders_to_analyse[i])
               predict_svs_slide_DNN(full_paths[i], folders_to_analyse[i], clonal_mark_type, find_clones = find_clones, prob_thresh = 0.5)
               
      if (method=="B"):    
         ## Get parameters and pickle for all desired runs
         print("Pickling parameters for analysis")
         for i in range(num_to_run):
            if (os.path.isfile(folders_to_analyse[i]+"params.pickle") and args.r==False):
               print("Passing on %s, parameters previously pickled." % file_in[i])
               pass
            else: 
               GetThresholdsPrepareRun(full_paths[i], file_in[i], folder_out, clonal_mark_type)                    
         ## Perform analysis
         print("Performing analysis")
         for i in range(num_to_run):
            if (os.path.isfile(folders_to_analyse[i]+"crypt_contours.txt") and args.r==False):
               print("Passing on %s, image previously analysed." % folders_to_analyse[i])
               pass
            else:
               SegmentFromFolder(folders_to_analyse[i], clonal_mark_type, find_clones)
                
      ## Extract crypt counts from all analysed slides into base path
      extract_counts_csv(file_in, folder_out, base_path, args.qp_proj_name, find_clones)
            

        
if __name__=="__main__":
   run_analysis()
        
