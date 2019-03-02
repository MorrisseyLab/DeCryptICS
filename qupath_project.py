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

def folder_from_image(image_num_str):
    return "/Analysed_"+str(image_num_str)+'/'

def file_len(fname):
    with open(fname) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1

def extract_counts_csv(file_in, folder_out):
   contour_folders = [folder_from_image(im) for im in file_in]
   num = len(contour_folders)
   slidecounts = np.zeros([num, 5], dtype=np.int32)
   slidenames = []
   for i in range(num):
      cnt_file = folder_out + contour_folders[i] + "crypt_contours.txt"
      fuf_file = folder_out + contour_folders[i] + "fufi_contours.txt"
      mut_file  = folder_out + contour_folders[i] + "clone_contours.txt"
      ptch_file = folder_out + contour_folders[i] + "patch_contours.txt"
      if (os.path.isfile(cnt_file)):
         crcnt = file_len(cnt_file)
         crcnt = int(crcnt/2)
         fucnt = file_len(fuf_file)
         fucnt = int(fucnt/2)
         mutcnt = file_len(mut_file)
         mutcnt = int(mutcnt/2)
         ptcnt = file_len(ptch_file)
         ptcnt = int(ptcnt/2)
         ptchsize_sum = np.sum(np.loadtxt(folder_out + contour_folders[i] + "patch_sizes.txt"))
         slidenames.append(file_in[i])
         slidecounts[i,0] = crcnt
         slidecounts[i,1] = fucnt
         slidecounts[i,2] = mutcnt
         slidecounts[i,3] = mutcnt - ptchsize_sum + ptcnt
         slidecounts[i,4] = ptcnt
   slidecounts_p = pd.DataFrame(slidecounts)
   slidenames_p = pd.DataFrame({'img_name':slidenames})
   slidecounts_p = pd.concat([slidenames_p, slidecounts_p], axis=1)
   outname = "/slide_counts.csv"
   slidecounts_p.columns = ['Slide_ID', 'NCrypts', 'NFufis', 'NMutantCrypts', 'NClones', 'NPatches']
   slidecounts_p.to_csv(folder_out + outname, sep=',', index=False)

def create_qupath_project(path_to_project, full_paths, file_in, folder_out):
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
    with open(path_to_project+"/scripts/load_contours.groovy", 'w') as file:
        file.write("//guiscript=true" + '\n')
        file.write("import qupath.lib.objects.*" + '\n')
        file.write("import qupath.lib.roi.*" + '\n')
        file.write("import qupath.lib.objects.classes.PathClass;" + '\n')
        file.write("import qupath.lib.common.ColorTools;" + '\n')
        file.write("import qupath.lib.objects.classes.PathClassFactory;" + '\n')
        file.write('\n')
        file.write("// Change CLONE_THRESHOLD to vary the clone sensitivity" + '\n')
        file.write("// 0 will load all potential clones, 1 will load only those that have been manually curated." + '\n')
        file.write("// (This won't change the patch contours.)" + '\n')
        file.write("double CLONE_THRESHOLD = 0.5" + '\n')
        file.write('\n')
        file.write("// Some code taken from:\n// https://groups.google.com/forum/#!topic/qupath-users/j_Wd1hy4eKM\n// https://groups.google.com/forum/#!topic/qupath-users/QyzvMjQ08cY\n// https://github.com/qupath/qupath/issues/169" + '\n')
        file.write('\n')
        
        file.write("// Add object classes" + '\n')
        file.write("def CryptClass = PathClassFactory.getPathClass(\"Crypt\")" + '\n')
        file.write("def CloneClass = PathClassFactory.getPathClass(\"Clone\")" + '\n')
        file.write("def PatchClass = PathClassFactory.getPathClass(\"Patch\")" + '\n')
        file.write("def FufiClass  = PathClassFactory.getPathClass(\"Fufi\")" + '\n')
        file.write("def pathClasses = getQuPath().getAvailablePathClasses()" + '\n')
        file.write("pathClasses.remove(CryptClass)" + '\n')
        file.write("pathClasses.remove(CloneClass)" + '\n')
        file.write("pathClasses.remove(PatchClass)" + '\n')
        file.write("pathClasses.remove(FufiClass)" + '\n')
        file.write("if (!pathClasses.contains(CryptClass)) {" + '\n')
        file.write("\tpathClasses.add(CryptClass)" + '\n')
        file.write("}" + '\n')
        file.write("if (!pathClasses.contains(CloneClass)) {" + '\n')
        file.write("\tpathClasses.add(CloneClass)" + '\n')
        file.write("}" + '\n')
        file.write("if (!pathClasses.contains(PatchClass)) {" + '\n')
        file.write("\tpathClasses.add(PatchClass)" + '\n')
        file.write("}" + '\n')
        file.write("if (!pathClasses.contains(FufiClass)) {" + '\n')
        file.write("\tpathClasses.add(FufiClass)" + '\n')
        file.write("}" + '\n')
        file.write("PathClassFactory.getPathClass(\"Crypt\").setColor(ColorTools.makeRGB(175,0,0))" + '\n')
        file.write("PathClassFactory.getPathClass(\"Clone\").setColor(ColorTools.makeRGB(10,152,30))" + '\n')
        file.write("PathClassFactory.getPathClass(\"Patch\").setColor(ColorTools.makeRGB(10,15,135))" + '\n')
        file.write("PathClassFactory.getPathClass(\"Fufi\").setColor(ColorTools.makeRGB(120,5,100))" + '\n')
        file.write('\n')
        
        file.write("// Define folder structure" + '\n')
        file.write("def base_folder = \"" + folder_out + "\"" + '\n')
        file.write("def cur_file = getCurrentImageData().getServer().getPath()" + '\n')
        file.write("print cur_file" + '\n')
        file.write("def ff = cur_file.tokenize(\'/\')[-1].tokenize(\'.\')[0]" + '\n')
        file.write('\n')
                               
        file.write("// Add crypt contours" + '\n')
        file.write("def file1 = new File(base_folder+\"Analysed_\"+ff+\"/crypt_contours.txt\")" + '\n')
        file.write("if( file1.exists() ) {" + '\n') 
        file.write('\t'+"def lines1 = file1.readLines()" + '\n')
        file.write('\t'+"num_rois = lines1.size/2" + '\n')
        file.write('\t'+"def pathObjects = []" + '\n')
        file.write('\t'+"for (i = 0; i<num_rois; i++) {" + '\n')
        file.write('\t\t'+"float[] x1 = lines1[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t'+"float[] y1 = lines1[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t\t'+"pathObjects << new PathDetectionObject(roi, CryptClass)" + '\n')
        file.write('\t'+'}' + '\n')
        file.write('\t'+"addObjects(pathObjects)" + '\n')
        file.write('\t'+"print(\"Done!\")" + '\n')
        file.write('}' + '\n')
        file.write('\n')
        
        file.write("// Add clone contours" + '\n')
        file.write("def file2 = new File(base_folder+\"Analysed_\"+ff+\"/clone_contours.txt\")" + '\n')
        file.write("def filescores = new File(base_folder+\"Analysed_\"+ff+\"/clone_scores.txt\")" + '\n')
        file.write("boolean scoreflag = false" + '\n')
        file.write("if( filescores.exists() ) {" + '\n')  
        file.write('\t'+"scoreflag = true" + '\n')
        file.write('}' + '\n')
        file.write("if( file2.exists() ) {" + '\n')         
        file.write('\t'+"def lines2 = file2.readLines()" + '\n')
        file.write('\t'+"num_rois = lines2.size/2" + '\n')
        file.write('\t'+"def pathObjects2 = []" + '\n')
        file.write('\t'+"if (scoreflag == true) {" + '\n')
        file.write('\t\t'+"def linesscores = filescores.readLines()" + '\n')
        file.write('\t\t'+"for (i = 0; i<num_rois; i++) {" + '\n')
        file.write('\t\t\t'+"double score = linesscores[i] as double" + '\n')
        file.write('\t\t\t'+"if ( score >= CLONE_THRESHOLD ) {" + '\n')
        file.write('\t\t\t\t'+"float[] x1 = lines2[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t\t\t'+"float[] y1 = lines2[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t\t\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t\t\t\t'+"pathObjects2 << new PathAnnotationObject(roi, CloneClass)" + '\n')
        file.write('\t\t\t' + '}' + '\n')
        file.write('\t\t' + '}' + '\n')
        file.write('\t' + '}' + '\n')
        file.write('\t'+"if (scoreflag == false) {" + '\n')
        file.write('\t\t'+"for (i = 0; i<num_rois; i++) {" + '\n')
        file.write('\t\t\t'+"float[] x1 = lines2[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t\t'+"float[] y1 = lines2[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t\t\t'+"pathObjects2 << new PathAnnotationObject(roi, CloneClass)" + '\n')
        file.write('\t\t'+'}' + '\n')
        file.write('\t'+'}' + '\n')
        file.write('\t'+"addObjects(pathObjects2)" + '\n')
        file.write('\t'+"print(\"Done!\")" + '\n')
        file.write('}' + '\n')
        file.write('\n')
        
        file.write("// Add patch contours" + '\n')
        file.write("def file3 = new File(base_folder+\"Analysed_\"+ff+\"/patch_contours.txt\")" + '\n')
        file.write("if( file3.exists() ) {" + '\n')         
        file.write('\t'+"def lines3 = file3.readLines()" + '\n')
        file.write('\t'+"num_rois = lines3.size/2" + '\n')
        file.write('\t'+"def pathObjects3 = []" + '\n')
        file.write('\t'+"for (i = 0; i<num_rois; i++) {" + '\n')
        file.write('\t\t'+"float[] x1 = lines3[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t'+"float[] y1 = lines3[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t\t'+"pathObjects3 << new PathAnnotationObject(roi, PatchClass)" + '\n')
        file.write('\t'+'}' + '\n')
        file.write('\t'+"addObjects(pathObjects3)" + '\n')
        file.write('\t'+"print(\"Done!\")" + '\n')
        file.write('}' + '\n')
        file.write('\n')
        
        file.write("// Add fufi contours" + '\n')
        file.write("def file4 = new File(base_folder+\"Analysed_\"+ff+\"/fufi_contours.txt\")" + '\n')
        file.write("if( file4.exists() ) {" + '\n')         
        file.write('\t'+"def lines4 = file4.readLines()" + '\n')
        file.write('\t'+"num_rois = lines4.size/2" + '\n')
        file.write('\t'+"def pathObjects4 = []" + '\n')
        file.write('\t'+"for (i = 0; i<num_rois; i++) {" + '\n')
        file.write('\t\t'+"float[] x1 = lines4[2*i].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t'+"float[] y1 = lines4[2*i+1].tokenize(\',\') as float[]" + '\n')
        file.write('\t\t'+"def roi = new PolygonROI(x1, y1, -300, 0, 0)" + '\n')
        file.write('\t\t'+"pathObjects4 << new PathAnnotationObject(roi, FufiClass)" + '\n')
        file.write('\t'+'}' + '\n')
        file.write('\t'+"addObjects(pathObjects4)" + '\n')
        file.write('\t'+"print(\"Done!\")" + '\n')
        file.write('}' + '\n')
        file.write('\n')
        
