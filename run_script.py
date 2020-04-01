#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:18:17 2018

@author: doran
"""

import glob, os, sys
import numpy  as np
import pandas as pd
import argparse, datetime
from qupath_project import create_qupath_project, extract_counts_csv, file_len, folder_from_image
from MiscFunctions  import mkdir_p

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

   parser.add_argument('-c', choices = ["1" , "2" , "3", "0"],
                             default = "1", 
                             dest    = "clonal_mark",
                             help    = "Clonal mark type, if clones are to be counted. "
                                       "1 for KDM6A/NONO/MAOA/STAG2/HDAC6 type (brown clone on blue nuclear). "
                                       "2 for p53 type (dark brown clone on light brown nuclear). "
                                       "3 for mPAS type (purple clone in cytoplasm). "
				       "0 for non-mutational stainings like H&E. "
                                       "Defaults to '1' if -c is not passed, meaning clones assumed KDM6A/NONO/MAOA/STAG2/HDAC6 type. "
                                       "(Unless clonal marks found in input file, which take precendent.) ")

   parser.add_argument('-method', choices = ["D" , "B"],
                                  default = "D", 
                                  dest    = "method",
                                  help    = "Method of crypt finding: D uses a deep neural network, B uses a Bayesian model (default is D; B not implemented properly). ")

   parser.add_argument('-r', action = "store_true", 
                             default = False,
                             help = "Forces repeat analysis of slides with existing crypt contour files. "
                                    "Defaults to False if -r flag missing. ")

   parser.add_argument('-mouse', action = "store_true", 
                                 default = False,
                                 help    = "Indicates we are analysing mouse tissue. ")
                                 
   args = parser.parse_args()
   ## check args
   print("Running with the following inputs:")
   print('action       = {!r}'.format(args.action))
   print('input_file   = {!r}'.format(args.input_file))
   print('qp_proj_name = {!r}'.format(args.qp_proj_name))
   print('clonal_mark  = {!r}'.format(args.clonal_mark))
   print('method       = {!r}'.format(args.method))
   print('force_repeat = {!r}'.format(args.r))
   print('mouse        = {!r}'.format(args.mouse))
   print("\n...Working...\n")
   
   ## Standardise clonal mark type string
   clonal_mark_type = args.clonal_mark
   if (clonal_mark_type=="1"): clonal_mark_type = 1 # KDM6A/MAOA/NONO/HDAC6/STAG2
   if (clonal_mark_type=="2"): clonal_mark_type = 2 # p53
   if (clonal_mark_type=="3"): clonal_mark_type = 3 # mPAS
   if (clonal_mark_type=="0"): clonal_mark_type = 0 # H&E

   ## Standardise method string
   method = args.method
   if (method.upper()=="D"): method="D"
   if (method.upper()=="B"): method="B"

   ## Find output folder
   input_file = os.path.abspath(args.input_file)
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
   a = np.asarray(a).reshape(a.shape)
   
   # extract file paths
   if len(a.shape)>1: # 2D input
      ncols = a.shape[1]   
      for i in range(ncols):
         if type(a[0,i]) == str:
            if (a[0,i].split('.')[-1]==ext1 or a[0,i].split('.')[-1]==ext2):
               pathind = i
      full_paths = list(a[:,pathind]) # take correct column
      clonal_mark_list = list(a[:,pathind+1])
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
      clonal_mark_list = [clonal_mark_type] * len(full_paths)
   linux_test = len(full_paths[0].split('/'))
   windows_test = len(full_paths[0].split('\\'))
   if (linux_test==1 and windows_test>1):
      file_in = [name.split("\\")[-1].split(".")[0] for name in full_paths]
   if (linux_test>1 and windows_test==1):
      file_in = [name.split("/")[-1].split(".")[0] for name in full_paths]
   
   ## Define file structures
   folder_out = base_path + '/' + "Analysed_slides/" 
   mkdir_p(folder_out)
         
   ## Create QuPath project for the current batch
   qupath_project_path = base_path + '/' + args.qp_proj_name
   create_qupath_project(qupath_project_path, full_paths, file_in, folder_out)
   print("QuPath project created in %s" % qupath_project_path)

   ## Turn clonal mark names to integer labels (if needed)
   for m in range(len(clonal_mark_list)):
      if type(clonal_mark_list[m])==type("string"):
         if   clonal_mark_list[m].upper()=="KDM6A": clonal_mark_list[m] = 1
         elif clonal_mark_list[m].upper()=="MAOA": clonal_mark_list[m]  = 1
         elif clonal_mark_list[m].upper()=="NONO": clonal_mark_list[m]  = 1
         elif clonal_mark_list[m].upper()=="HDAC6": clonal_mark_list[m] = 1
         elif clonal_mark_list[m].upper()=="STAG2": clonal_mark_list[m] = 1
         elif clonal_mark_list[m].upper()=="P53": clonal_mark_list[m]   = 2
         elif clonal_mark_list[m].upper()=="MPAS": clonal_mark_list[m]  = 3
         elif clonal_mark_list[m].upper()=="H&E": clonal_mark_list[m]   = 0

   if args.action == "count":
      from SegmentTiled_gen import predict_slide_DNN
      num_to_run = len(file_in)
      folders_to_analyse = [folder_out+fldr for fldr in ["Analysed_"+fnum+"/" for fnum in file_in]]

      ## Perform crypt segmentation
      ######################################
      if (method == "D"):
         # Load DNN model
#         import keras
         import DNN.params as params
#         if keras.backend._BACKEND=="tensorflow":
         import tensorflow as tf
         input_shape = (params.input_size_run, params.input_size_run, 3)
         chan_num = 3
#         elif keras.backend._BACKEND=="mxnet":
#            import mxnet
#            input_shape = (3, params.input_size_run, params.input_size_run)
#            chan_num = 1
         dnn_model = params.model_factory(input_shape=input_shape, num_classes=5, chan_num=chan_num)
         maindir = os.path.dirname(os.path.abspath(__file__))
         if (args.mouse==True):
            weightsin = os.path.join(maindir, 'DNN', 'weights', 'mousecrypt_weights.hdf5')         
         else:
            weightsin = os.path.join(maindir, 'DNN', 'weights', 'cryptfuficlone_weights.hdf5')      
         dnn_model.load_weights(weightsin)
         for i in range(num_to_run):
            if (os.path.isfile(folders_to_analyse[i]+"crypt_contours.txt") and args.r==False):         
               print("Passing on %s, image previously analysed." % folders_to_analyse[i])
               pass
            else:
               print("Beginning segmentation on %s with clonal mark type %d." % (full_paths[i], clonal_mark_list[i]))
               predict_slide_DNN(full_paths[i], folders_to_analyse[i], clonal_mark_list[i], dnn_model, chan_num, prob_thresh = 0.6, clone_prob_thresh = 0.45)
               
      ## DO NOT USE -- NOT FULLY IMPLEMENTED
      if (method=="B"):
         print("Don't use method=='B': Bayesian segmentation method not implemented in new software version.")
         return 0
#         from SegmentTiled_gen import GetThresholdsPrepareRun, SegmentFromFolder
#         ## Get parameters and pickle for all desired runs
#         print("Pickling parameters for analysis")
#         for i in range(num_to_run):
#            if (os.path.isfile(folders_to_analyse[i]+"params.pickle") and args.r==False):
#               print("Passing on %s, parameters previously pickled." % file_in[i])
#               pass
#            else: 
#               GetThresholdsPrepareRun(full_paths[i], file_in[i], folder_out, clonal_mark_list[i])                    
#         ## Perform analysis
#         print("Performing analysis")
#         for i in range(num_to_run):
#            if (os.path.isfile(folders_to_analyse[i]+"crypt_contours.txt") and args.r==False):
#               print("Passing on %s, image previously analysed." % folders_to_analyse[i])
#               pass
#            else:
#               SegmentFromFolder(folders_to_analyse[i], clonal_mark_list[i], False)
                
      ## Extract crypt counts from all analysed slides into base path
      extract_counts_csv(file_in, folder_out)
                    
if __name__=="__main__":
   run_analysis()
        
