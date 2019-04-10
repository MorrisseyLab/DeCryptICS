#!/usr/bin/env python3

import glob, os, sys
import argparse
import numpy as np
import pandas as pd
if (len(sys.argv) > 2):
        sys.stderr.write('\n\n')
        sys.exit(1)

def output_filelist(fnames, ext=''):
   fpaths = [os.path.abspath(f) for f in fnames]
   initpath = fpaths[0]
   linux_test = len(initpath.split('/'))
   windows_test = len(initpath.split('\\'))
   if (linux_test==1 and windows_test>1):
      outpath = '/' + os.path.join(*initpath.split('\\')[:-1]) + '/'
   if (linux_test>1 and windows_test==1):
      outpath = '/' + os.path.join(*initpath.split('/')[:-1]) + '/'
   if (linux_test==1 and windows_test==1):
      outpath = os.getcwd() + '/'
   with open(outpath + 'input_files'+ext+'.txt', 'w') as fo:      
      fo.write("#<full paths to slides>\t<clonal mark>\n")
      for i in range(len(fpaths)):
         fo.write(fpaths[i] + '\t' + str(marks[imorder[i]]) + '\n')

def main():
   parser = argparse.ArgumentParser(description = "First argument -f denotes the full path to folder containing "
                                                  "the images and slide information file). "
                                                  "Running with no arguments uses the current working directory. "
                                                  "Second argument -c can be given to set the mutation "
                                                  "mark to be used for all slides in the input folder." )
   parser.add_argument('-f', dest    = "input_folder", 
                             default = "",
                             help    = "Input folder. Defaults to current working directory if -f missing. ")
   
   parser.add_argument('-c', dest    = "clonal_mark",
                             default = "1", 
                             help    = "Clonal mark type for all input images. "
                                       "1 for KDM6A/NONO/MAOA type (brown clone on blue nuclear). "
                                       "2 for STAG2 type (brown clone on grey-brown-blue nuclear). "
                                       "3 for mPAS type (purple clone in cytoplasm). "
                                       "Note: all slides in input list will be analysed using the same clonal mark type. "
                                       "If -c is not passed, and no slide info file is found, defaults to 1.")
   args = parser.parse_args()
   ## check args
   print("Running with the following inputs:")
   print('input_folder   = {!r}'.format(args.input_folder))
   print('clonal_mark  = {!r}'.format(args.clonal_mark))                                       
          
   ## get file list                     
   if args.input_folder!="":
      print("Reading files in %s" % args.input_folder)
      fnames_svs = glob.glob(args.input_folder+'/'+"*.svs")
      fnames_tif = glob.glob(args.input_folder+'/'+"*.tif*")
      fnames_png = glob.glob(args.input_folder+'/'+"*.png")
      fnames_jpg = np.hstack([glob.glob(args.input_folder+'/'+"*.jpg"), glob.glob(args.input_folder+'/'+"*.jpeg")])
      slideinfo_f = args.input_folder + "/slide_info.csv"
   else:
      print("Reading files in %s" % os.getcwd())
      # case sensitive for now
      fnames_svs = glob.glob("*.svs")
      fnames_tif = glob.glob("*.tif*")
      fnames_png = glob.glob("*.png")
      fnames_jpg = np.hstack([glob.glob("*.jpg"), glob.glob("*.jpeg")])
      slideinfo_f = os.getcwd() + "/slide_info.csv"
      
   ## get clonal mark information
   if os.path.isfile(slideinfo_f):
      ftype = slideinfo_f.split('.')[-1]
      ext1 = "svs"; ext2 = "svs"
      if (slideinfo_f.split('_')[-1].split('.')[0] == "tif"):
         ext1 = "tif"; ext2 = "tiff"
      if (slideinfo_f.split('_')[-1].split('.')[0] == "png"):
         ext1 = "png"; ext2 = "png"
      if (slideinfo_f.split('_')[-1].split('.')[0] == "jpg"):
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
   
   blockids = a[:,1]
   slideids = a[:,4]   
   
   if (len(fnames_svs)>0): output_filelist(fnames_svs)
   if (len(fnames_tif)>0): output_filelist(fnames_tif, '_tif')
   if (len(fnames_png)>0): output_filelist(fnames_png, '_png')
   if (len(fnames_jpg)>0): output_filelist(fnames_jpg, '_jpg')
                                    

if __name__=="__main__":
   main()
   
