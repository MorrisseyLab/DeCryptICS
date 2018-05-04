#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:18:17 2018

@author: doran
"""

import glob, os, sys
import numpy as np
from SegmentTiled_gen import GetThresholdsPrepareRun, SegmentFromFolder, predict_svs_slide_DNN
from qupath_project   import create_qupath_project, extract_counts_csv, file_len, folder_from_image

if (len(sys.argv) < 5):
        sys.stderr.write('Wrong number of arguments!\n')
        sys.stderr.write('4 required, %d found.\n' % (len(sys.argv)-1))
        sys.stderr.write("Bad command Line!\nUsage: %s <base_path> <batch_ID> <clonal_mark_type==P/N/PNN/NNN> <method==D/B>\nDefault folder structure: /base_path/batch_ID/raw_images/images.svs\n" % str(sys.argv[0]))
        sys.exit(1)

base_path = sys.argv[1]
batch_ID = sys.argv[2]
clonal_mark_type = sys.argv[3]
method = sys.argv[4]

def run_analysis(base_path, batch_ID, clonal_mark_type, method):
    ## Clonal mark type can be either:
    # P: positive nuclear
    # N: negative nuclear
    # PNN: positive non-nuclear
    # NNN: negative non-nuclear
    # BN: both negative
    # BP: both positive
    if (clonal_mark_type.upper()=="P"): clonal_mark_type="P"
    if (clonal_mark_type.upper()=="N"): clonal_mark_type="N"
    if (clonal_mark_type.upper()=="PNN"): clonal_mark_type=="PNN"
    if (clonal_mark_type.upper()=="NNN"): clonal_mark_type=="NNN"
    if (clonal_mark_type.upper()=="BN"): clonal_mark_type=="BN"
    if (clonal_mark_type.upper()=="BP"): clonal_mark_type=="BP"

    ## Choose either DNN ("D") or Bayesian ("B") crypt segmentation    
    if (method.upper()=="D"): method="D"
    if (method.upper()=="B"): method="B"
    
    ## Define file structures
    ######################################    
    folder_in  = base_path + "/" + batch_ID + "/raw_images/"
    folder_out = base_path + "/" + batch_ID + "/Analysed_slides/"
    try:
        os.mkdir(folder_out)
    except:
        pass        
    full_paths = glob.glob(folder_in + "*.svs")
    file_in = [name.split("/")[-1].split(".")[0] for name in full_paths]
    num_to_run = len(file_in)
    folders_to_analyse = [folder_out+fldr for fldr in ["Analysed_"+fnum+"/" for fnum in file_in]]
    
    ## Perform crypt segmentation
    ######################################
    if (method=="B"):    
        ## Get parameters and pickle for all desired runs
        print("Pickling parameters for analysis")
        for i in range(num_to_run):
            if (os.path.isfile(folders_to_analyse[i]+"params.pickle")):
                print("Passing on %s, parameters previously pickled." % file_in[i])
                pass
            else: 
                GetThresholdsPrepareRun(folder_in, file_in[i], folder_out, clonal_mark_type)    
                
        ## Perform analysis
        print("Performing analysis")
        for i in range(num_to_run):
            if (os.path.isfile(folders_to_analyse[i]+"crypt_contours.txt")):
                print("Passing on %s, image previously analysed." % folders_to_analyse[i])
                pass
            else:
                SegmentFromFolder(folders_to_analyse[i], clonal_mark_type)
                
    if (method == "D"):
        for i in range(num_to_run):
            if (os.path.isfile(folders_to_analyse[i]+"crypt_contours.txt")):
                print("Passing on %s, image previously analysed." % folders_to_analyse[i])
                pass
            else:
                print("Beginning segmentation on %s." % folders_to_analyse[i])
                predict_svs_slide_DNN(full_paths[i], folders_to_analyse[i], clonal_mark_type)
                
    ## Create QuPath project for the current batch
    path_to_project = base_path + "/qupath_projects/" + batch_ID
    create_qupath_project(path_to_project, full_paths, file_in, folder_out)
    ## Extract crypt counts from all analysed slides
    extract_counts_csv(folder_in, folder_out, path_to_project)
        
if __name__=="__main__":
    run_analysis(base_path, batch_ID, clonal_mark_type, method)
        
