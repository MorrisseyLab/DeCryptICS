#!/usr/bin/env python3

import glob, os, sys

if (len(sys.argv) > 2):
        sys.stderr.write('Only one argument required (the full path to folder containing .svs files)\nRunning with no arguments uses the current working directory.\n')
        sys.exit(1)

if __name__=="__main__":
   if len(sys.argv)>1:
      print("Reading files in %s" % sys.argv[1])
      fnames = glob.glob(sys.argv[1]+'/'+"*.svs")
   else:
      print("Reading files in %s" % os.getcwd())
      fnames = glob.glob("*.svs")
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
   with open(outpath + 'input_files.txt', 'w') as fo:      
      fo.write("#<full paths to slides>\n")
      for p in fpaths:
         fo.write(p + '\n')
