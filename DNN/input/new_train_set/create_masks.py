import cv2, os, sys
import openslide as osl
import pandas as pd
import numpy as np
import glob
from MiscFunctions import plot_img

def load_sample2(data):
    img_f = data
    img = cv2.imread(img_f, cv2.IMREAD_COLOR)
    return img

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if os.path.isdir(path):
            pass
        else: raise

def set_background_to_black(img):
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey_image, 254, 255, cv2.THRESH_BINARY)
    return mask

def generate_mask_from_prediction(imgfile, p_thresh=0.45):
    grey_image = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(grey_image, p_thresh*255, 255, cv2.THRESH_BINARY)
    return mask

def lower_intensity_by_one(img):
    inds = np.where(img>0)
    img[inds] = img[inds] - 1
    return img

# get tile reference
tileref = np.loadtxt('/home/doran/immwork/py_code/DeCryptICS/DNN/input/new_train_set/bad_tiles_need_check.tsv', dtype=str, skiprows=1)

dnnpath = "/home/doran/immwork/py_code/DeCryptICS/DNN/input/new_train_set/"
inpath = dnnpath + "/premask/"
outpath = dnnpath + "/mask/"
imfiles = glob.glob(inpath + "set_*/*.png")

true_threshold = 40

# run processing and save
for path in imfiles:
   img = cv2.imread(path, cv2.IMREAD_COLOR)
   mask = set_background_to_black(img)
   im_number = path.split("/")[-1][4:] # remove "img_"
   im_number = im_number.split("_clone.png")[0][:-1]
   # find set and slide
   origfile = [s for s in tileref if im_number in s][0].split('/')
   outfile = outpath + '/' + origfile[-3] + '/' + origfile[-2] + '/'
   mkdir_p(outfile)
   # check if actually a T clone
   if (cv2.countNonZero(mask)>true_threshold):
      outfile = outfile + "mask_" + im_number + 'T_clone.png'
   else:
      outfile = outfile + "mask_" + im_number + 'F_clone.png'
   cv2.imwrite(outfile, mask)
   
   
