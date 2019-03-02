#!/usr/bin/env python3

import glob, os, sys
import numpy as np

if (len(sys.argv) > 2):
        sys.stderr.write('Only one argument required (the full path to folder containing image files)\nRunning with no arguments uses the current working directory.\n')
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
      fo.write("#<full paths to slides>\n")
      for p in fpaths:
         fo.write(p + '\n')

if __name__=="__main__":
   if len(sys.argv)>1:
      print("Reading files in %s" % sys.argv[1])
      fnames_svs = glob.glob(sys.argv[1]+'/'+"*.svs")
      fnames_tif = glob.glob(sys.argv[1]+'/'+"*.tif*")
      fnames_png = glob.glob(sys.argv[1]+'/'+"*.png")
      fnames_jpg = np.hstack([glob.glob(sys.argv[1]+'/'+"*.jpg"), glob.glob(sys.argv[1]+'/'+"*.jpeg")])
   else:
      print("Reading files in %s" % os.getcwd())
      # case sensitive for now
      fnames_svs = glob.glob("*.svs")
      fnames_tif = glob.glob("*.tif*")
      fnames_png = glob.glob("*.png")
      fnames_jpg = np.hstack([glob.glob("*.jpg"), glob.glob("*.jpeg")])
   if (len(fnames_svs)>0): output_filelist(fnames_svs)
   if (len(fnames_tif)>0): output_filelist(fnames_tif, '_tif')
   if (len(fnames_png)>0): output_filelist(fnames_png, '_png')
   if (len(fnames_jpg)>0): output_filelist(fnames_jpg, '_jpg')
