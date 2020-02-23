import cv2, os, sys
import openslide as osl
import pandas as pd
import numpy as np
import glob
import shutil


## relabel clones as true or false after manual curation

masks = glob.glob("/home/doran/Work/py_code/DeCryptICS/DNN/input/new_train_set/premask/mask/set*/slide*/*.png")

true_thresh = 40

for ll in range(len(masks)):
   loadmask = cv2.imread(masks[ll], cv2.IMREAD_GRAYSCALE)
   # find image
   a = masks[ll].split("premask/mask/")         
   a1 = a[0] + 'img/' + a[1].replace('mask', 'img').split('_clone.png')[0][:-1] + 'T_clone.png'
   a2 = a[0] + 'img/' + a[1].replace('mask', 'img').split('_clone.png')[0][:-1] + 'F_clone.png'
   if (cv2.countNonZero(loadmask)>true_thresh):
      # if labelled as false, rename to true
      if os.path.exists(a2):
         shutil.move(a2, a1)
   else:
      # if labelled as true, rename to false
      if os.path.exists(a1):
         shutil.move(a1, a2)
         

## also move not-done images to a holding directory
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if os.path.isdir(path):
            pass
        else: raise
        
notdone = glob.glob("/home/doran/Work/py_code/DeCryptICS/DNN/input/new_train_set/premask/not_done/set*/*.png")
notdone = [s.split('/')[-1] for s in notdone]
origpaths = np.loadtxt("/home/doran/Work/py_code/DeCryptICS/DNN/input/new_train_set/bad_tiles_need_check.tsv", skiprows=1, dtype=str)

holding_dir = "/home/doran/Work/py_code/DeCryptICS/DNN/input/new_train_set/not_curated/"

for ff in notdone:
   thisfile = [s for s in origpaths if ff in s][0]
   thisfile_s = thisfile.split('/')
   outfile = holding_dir + '/' + thisfile_s[-3] + '/' + thisfile_s[-2] + '/'
   mkdir_p(outfile)
   shutil.move(thisfile, outfile + thisfile_s[-1])
