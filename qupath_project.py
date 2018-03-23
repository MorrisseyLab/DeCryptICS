#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:01:29 2018

@author: doran
"""
import os, time

def create_qupath_project(path_to_project, full_paths, file_in, folder_out):
    num_to_run = len(file_in)
    # Create directory and essential sub-directories
    try:
        os.mkdir(path_to_project)
    except:
        pass
    try:
        os.mkdir(path_to_project+"/data")
    except:
        pass
    try:
        os.mkdir(path_to_project+"/thumbnails")
    except:
        pass
    try:
        os.mkdir(path_to_project+"/scripts")
    except:
        pass
    # Write project file
    with open(path_to_project+"/project.qpproj", 'w') as file:
        file.write('{' + '\n')
        file.write('\t'+"\"createTimestamp\": %d,\n" % int(time.time()))
        file.write('\t'+"\"modifyTimestamp\": %d,\n" % int(time.time()))
        file.write('\t'+"\"images\": [" + '\n')
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
    with open(path_to_project+"/scripts/load_contours.groovy", 'w') as file:
        file.write("import qupath.lib.objects.*" + '\n')
        file.write("import qupath.lib.roi.*" + '\n')
        file.write('\n')
        file.write("// Some code taken from here https://groups.google.com/forum/#!topic/qupath-users/j_Wd1hy4eKM" + '\n')
        file.write('\n')
        file.write("def base_folder = \"" + folder_out + "\"" + '\n')
        file.write("def cur_file = getCurrentImageData().getServer().getPath()" + '\n')
        file.write("print cur_file" + '\n')
        file.write("def ff = cur_file.tokenize(\'/\')[-1].tokenize(\'.\')" + '\n')
        file.write('\n')
        file.write("def file1 = new File(base_folder+\"Analysed_\"+ff[0]+\"/crypt_contours.txt\")" + '\n')
        file.write("def lines1 = file1.readLines()" + '\n')
        file.write("num_rois = lines1.size/2" + '\n')
        file.write("def pathObjects = []" + '\n')
        file.write("for (i = 0; i <num_rois; i++) {" + '\n')
        file.write('\t'+"float[] x1 = lines1[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t'+"float[] y1 = lines1[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t'+"// Create object" + '\n')
        file.write('\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t'+"pathObjects << new PathDetectionObject(roi)" + '\n')
        file.write('}' + '\n')
        file.write("// Add object to hierarchy" + '\n')
        file.write("addObjects(pathObjects)" + '\n')
        file.write("print(\"Done!\")" + '\n')
        file.write('\n')
        file.write("def file2 = new File(base_folder+\"Analysed_\"+ff[0]+\"/clone_contours.txt\")" + '\n')
        file.write("def lines2 = file2.readLines()" + '\n')
        file.write("num_rois = lines2.size/2" + '\n')
        file.write("def pathObjects2 = []" + '\n')
        file.write("for (i = 0; i <num_rois; i++) {" + '\n')
        file.write('\t'+"float[] x1 = lines2[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t'+"float[] y1 = lines2[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t'+"// Create object" + '\n')
        file.write('\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t'+"pathObjects2 << new PathDetectionObject(roi)" + '\n')
        file.write('}' + '\n')
        file.write("// Add object to hierarchy" + '\n')
        file.write("addObjects(pathObjects2)" + '\n')
        file.write("print(\"Done!\")" + '\n')
        file.write('\n')