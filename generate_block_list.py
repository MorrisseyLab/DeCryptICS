#!/usr/bin/env python3

import glob, os, sys
import argparse
import numpy as np
import pandas as pd

def output_blocklist(fnames, imgids, marks, ext=''):
   fpaths = [os.path.abspath(f) for f in fnames]
   fpath_nums = [int(im.split('/')[-1].split('.')[0]) for im in fpaths]
   imorder = [np.where(fpath_nums==imgid)[0][0] for imgid in imgids if imgid in fpath_nums]
   # cut down marks list if files missing/unmatched
   marks_order = [marks[np.where(imgids==imgid)[0][0]] for imgid in imgids if imgid in fpath_nums]   
   #imorder = [np.where(imgids==int(im.split('/')[-1].split('.')[0]))[0] for im in fpaths]
   #imorder = np.concatenate(imorder).ravel()
   initpath = fpaths[0]
   linux_test = len(initpath.split('/'))
   windows_test = len(initpath.split('\\'))
   if (linux_test==1 and windows_test>1):
      outpath = '/' + os.path.join(*initpath.split('\\')[:-1]) + '/'
   if (linux_test>1 and windows_test==1):
      outpath = '/' + os.path.join(*initpath.split('/')[:-1]) + '/'
   if (linux_test==1 and windows_test==1):
      outpath = os.getcwd() + '/'   
   with open(outpath + 'input_files' + ext + '.txt', 'w') as fo:      
      fo.write("#<full paths to slides>\t<clonal mark>\n")
      for i in range(len(imorder)):
         fo.write(fpaths[imorder[i]] + '\t' + str(marks_order[i]) + '\n')

def main():
   parser = argparse.ArgumentParser(description = "First argument -f denotes the full path to folder containing "
                                                  "the images and slide information file). "
                                                  "Running with no arguments uses the current working directory. "
                                                  "Second argument -c can be given to set the mutation "
                                                  "mark to be used for all slides in the input folder." )
   parser.add_argument("input_folder", default = "",
                            help = "Input folder. Defaults to current working directory if missing. ")
   
   parser.add_argument('-c', dest    = "clonal_mark",
                             default = "1", 
                             help    = "Clonal mark type for all input images. "
                                       "1: KDM6A/NONO/MAOA/HDAC6/STAG2 "
                                       "2: p53 "
                                       "3: mPAS "
                                       "0: H&E "
                                       "A slide info file will supersede -c flag. "
                                       "If -c is not passed and no slide info file is found, defaults to 1.")
   args = parser.parse_args()
   ## check args
   print("Running with the following inputs:")
   print('input_folder   = {!r}'.format(args.input_folder))
   print('clonal_mark  = {!r}'.format(args.clonal_mark))                                       
   input_folder = args.input_folder
   clonal_mark = args.clonal_mark
   if str(clonal_mark).upper()=="KDM6A": clonal_mark = 1
   if str(clonal_mark).upper()=="MAOA":  clonal_mark = 1
   if str(clonal_mark).upper()=="NONO":  clonal_mark = 1
   if str(clonal_mark).upper()=="HDAC6": clonal_mark = 1
   if str(clonal_mark).upper()=="STAG2": clonal_mark = 1
   if str(clonal_mark).upper()=="P53":   clonal_mark = 2
   if str(clonal_mark).upper()=="MPAS":  clonal_mark = 3
   if str(clonal_mark).upper()=="H&E":   clonal_mark = 0
   ## get file list                     
   if input_folder!="":
      print("Reading files in %s" % input_folder)
      fnames_svs = glob.glob(input_folder+'/'+"*.svs")
      fnames_tif = glob.glob(input_folder+'/'+"*.tif*")
      fnames_png = glob.glob(input_folder+'/'+"*.png")
      fnames_jpg = glob.glob(input_folder+'/'+"*.jpg") + glob.glob(input_folder+'/'+"*.jpeg")
      slideinfo_f = input_folder + "/slide_info.csv"
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

      if (ftype=="csv"):      a = pd.read_csv(slideinfo_f)
      elif (ftype[:2]=="xl"): a = pd.read_excel(slideinfo_f)
      else:                   a = pd.read_table(slideinfo_f)
      heads = list(a.columns.values)      
      imgids = np.asarray(a['Image ID'])
      marks = np.asarray(a['mark'])
   else:
      imgids = np.asarray([int(im.split('/')[-1].split('.')[0]) for im in fnames_svs+fnames_tif+fnames_png+fnames_jpg])
      marks = np.ones(len(imgids), dtype=np.int32) * int(clonal_mark)
   
   if (len(fnames_svs)>0): output_blocklist(fnames_svs, imgids, marks)
   if (len(fnames_tif)>0): output_blocklist(fnames_tif, imgids, marks, '_tif')
   if (len(fnames_png)>0): output_blocklist(fnames_png, imgids, marks, '_png')
   if (len(fnames_jpg)>0): output_blocklist(fnames_jpg, imgids, marks, '_jpg')
                                    

if __name__=="__main__":
   main()
   
