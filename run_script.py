#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:18:17 2018

@author: doran
"""

from SegmentTiled_gen import GetThresholdsPrepareRun, SegmentFromFolder
import glob, os
import numpy as np
from qupath_project import create_qupath_project

def folder_from_image(image_num_str):
    return "/Analysed_"+str(image_num_str)+'/'

def file_len(fname):
    with open(fname) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1

def extract_counts_csv(im_folder, folder_out):
    images = glob.glob(im_folder + "/*.svs")
    images = [name[:-4] for name in images] 
    images = [name[len(im_folder):] for name in images]
    contour_folders = [folder_from_image(im) for im in images]
    num = len(contour_folders)
    slidecounts = np.zeros([num,2], dtype=np.int32)
    for i in range(num):
        cnt_file = folder_out + contour_folders[i] + "crypt_contours.txt"
        if (os.path.isfile(cnt_file)):
            wcout = file_len(cnt_file)            
            wcout = int(wcout/2)
            slidecounts[i,0] = int(images[i])
            slidecounts[i,1] = wcout
    
    np.savetxt(folder_out+"/slide_counts.csv", slidecounts, delimiter=",")
        

if __name__=="__main__":
    ## Define file structures
    base_path = "/home/doran/Work/images/"
    ## Read/define batch ID for this processing run
    clonal_mark = "NONO"
    batch_ID = "NONO_March2018"
    # this batch ID should then define the folder structure etc:
    folder_in  = base_path + batch_ID + "/raw_images/"
    folder_out = base_path + batch_ID + "/Analysed_slides/"
    try:
        os.mkdir(folder_out)
    except:
        pass
    file_in = glob.glob(folder_in + "*.svs")
    full_paths = file_in
    # But only want file number as input: trim .svs and folder_in length
    file_in = [name[:-4] for name in file_in] 
    file_in = [name[len(folder_in):] for name in file_in]
    # Get parameters and pickle for all desired runs
    num_to_run = len(file_in)
    #num_to_run = 1
    print("Pickling parameters for analysis")
    for i in range(num_to_run):
        if (os.path.isfile(folder_out+"Analysed_"+file_in[i]+"/params.pickle")):
            print("Passing on %s, parameters previously pickled." % file_in[i])
            pass
        else: 
            GetThresholdsPrepareRun(folder_in, file_in[i], folder_out, clonal_mark)
    
    ## Perform analysis
    print("Performing analysis")
    folders_to_analyse = glob.glob(folder_out + "Analysed_*/")
    num_to_run = len(folders_to_analyse)
    for i in range(num_to_run):
        if (os.path.isfile(folders_to_analyse[i]+"crypt_contours.txt")):
            print("Passing on %s, image previously analysed." % folders_to_analyse[i])
            pass
        else:
            SegmentFromFolder(folders_to_analyse[i], clonal_mark)

    ## Extract crypt counts from all analysed slides
    extract_counts_csv(folder_in, folder_out)   
    ## Create QuPath project for the current batch
    path_to_project = base_path + "/qupath_projects/" + batch_ID
    create_qupath_project(path_to_project, full_paths, file_in, folder_out)
        
