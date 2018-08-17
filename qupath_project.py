#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:01:29 2018

@author: doran
"""
import os, time, glob, fileinput
import numpy as np
import csv

def folder_from_image(image_num_str):
    return "/Analysed_"+str(image_num_str)+'/'

def file_len(fname):
    with open(fname) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1

def extract_counts_csv(im_folder, folder_out, save_counts_here):
   images = glob.glob(im_folder + "/*.svs")
   images = [name.split("/")[-1].split(".")[0] for name in images]
   contour_folders = [folder_from_image(im) for im in images]
   num = len(contour_folders)
   slidecounts = np.zeros([num,3], dtype=np.int32)
   for i in range(num):
      cnt_file = folder_out + contour_folders[i] + "crypt_contours.txt"
      cln_file = folder_out + contour_folders[i] + "clone_contours.txt"
      if (os.path.isfile(cnt_file)):
         wcout = file_len(cnt_file)
         wcout = int(wcout/2)
         clcnt = file_len(cln_file)
         clcnt = int(clcnt/2)
         slidecounts[i,0] = int(images[i])
         slidecounts[i,1] = wcout
         slidecounts[i,2] = clcnt
   np.savetxt(save_counts_here+"/slide_counts.csv", slidecounts, delimiter=",")
   # Now add column headers (messy!)
   with open(save_counts_here+"/slide_counts.csv", newline='') as f:
      r = csv.reader(f)
      data = [line for line in r]
   with open(save_counts_here+"/slide_counts.csv", 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['Slide_ID', 'NCrypts', 'NClones'])
      w.writerows(data)         
   #for line in fileinput.input(files=[save_counts_here+"/slide_counts.csv"], inplace=True):
   #   if fileinput.isfirstline():
   #      print('Slide_ID,NCrypts,NClones')
   #print(line,)

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
    with open(path_to_project+"/scripts/load_contours.groovy", 'w') as file:
        file.write("//guiscript=true" + '\n')
        file.write("import qupath.lib.objects.*" + '\n')
        file.write("import qupath.lib.roi.*" + '\n')
        file.write("import qupath.lib.objects.classes.PathClass;" + '\n')
        file.write("import qupath.lib.common.ColorTools;" + '\n')
        file.write("import qupath.lib.objects.classes.PathClassFactory;" + '\n')
        file.write('\n')
        file.write("// Some code taken from:\n// https://groups.google.com/forum/#!topic/qupath-users/j_Wd1hy4eKM\n// https://groups.google.com/forum/#!topic/qupath-users/QyzvMjQ08cY\n// https://github.com/qupath/qupath/issues/169" + '\n')
        file.write('\n')
        
        file.write("// Add object classes" + '\n')
        file.write("def CryptClass = PathClassFactory.getPathClass(\"Crypt\")" + '\n')
        file.write("def CloneClass = PathClassFactory.getPathClass(\"Clone\")" + '\n')
        file.write("def pathClasses = getQuPath().getAvailablePathClasses()" + '\n')
        file.write("pathClasses.remove(CryptClass)" + '\n')
        file.write("pathClasses.remove(CloneClass)" + '\n')
        file.write("if (!pathClasses.contains(CryptClass)) {" + '\n')
        file.write("\tpathClasses.add(CryptClass)" + '\n')
        file.write("}" + '\n')
        file.write("if (!pathClasses.contains(CloneClass)) {" + '\n')
        file.write("\tpathClasses.add(CloneClass)" + '\n')
        file.write("}" + '\n')
        file.write("PathClassFactory.getPathClass(\"Crypt\").setColor(ColorTools.makeRGB(175,0,0))" + '\n')
        file.write("PathClassFactory.getPathClass(\"Clone\").setColor(ColorTools.makeRGB(10,152,30))" + '\n')
        file.write('\n')
        
        file.write("// Define folder structure" + '\n')
        file.write("def base_folder = \"" + folder_out + "\"" + '\n')
        file.write("def cur_file = getCurrentImageData().getServer().getPath()" + '\n')
        file.write("print cur_file" + '\n')
        file.write("def ff = cur_file.tokenize(\'/\')[-1].tokenize(\'.\')" + '\n')
        file.write('\n')
        
        file.write("// Add clone contours" + '\n')
        file.write("def file2 = new File(base_folder+\"Analysed_\"+ff[0]+\"/clone_contours.txt\")" + '\n')
        file.write("def lines2 = file2.readLines()" + '\n')
        file.write("num_rois = lines2.size/2" + '\n')
        file.write("def pathObjects2 = []" + '\n')
        file.write("for (i = 0; i <num_rois; i++) {" + '\n')
        file.write('\t'+"float[] x1 = lines2[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t'+"float[] y1 = lines2[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t'+"pathObjects2 << new PathAnnotationObject(roi, CloneClass)" + '\n')
        file.write('}' + '\n')
        file.write("addObjects(pathObjects2)" + '\n')
        file.write("print(\"Done!\")" + '\n')
        file.write('\n')
               
        file.write("// Add crypt contours" + '\n')
        file.write("def file1 = new File(base_folder+\"Analysed_\"+ff[0]+\"/crypt_contours.txt\")" + '\n')
        file.write("def lines1 = file1.readLines()" + '\n')
        file.write("num_rois = lines1.size/2" + '\n')
        file.write("def pathObjects = []" + '\n')
        file.write("for (i = 0; i <num_rois; i++) {" + '\n')
        file.write('\t'+"float[] x1 = lines1[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t'+"float[] y1 = lines1[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t'+"pathObjects << new PathDetectionObject(roi, CryptClass)" + '\n')
        file.write('}' + '\n')
        file.write("addObjects(pathObjects)" + '\n')
        file.write("print(\"Done!\")" + '\n')
        file.write('\n')
        

