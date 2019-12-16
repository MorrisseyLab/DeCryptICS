#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:15:22 2018

@author: doran
"""
import cv2, glob, os, errno
import numpy as np

def load_sample2(data):
    img_f = data
    img = cv2.imread(img_f, cv2.IMREAD_COLOR)
    return img

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
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
    
if __name__=="__main__":
   ## Lowering intensity by one prior to manual masking
   ##########################################################################
   # load image list
   #train_path =     "/home/doran/Work/py_code/DeCryptICS/DNN/input/pre-mask/"
   #train_path =     "/home/doran/Work/py_code/experimental_DeCryptICS/DNN/input/pre-mask/"
   train_path =     "/home/doran/Work/py_code/DeCryptICS/DNN/input/new_train_set/premask/"
   flist = glob.glob(train_path+"set*/*.png")
#   flist = glob.glob(train_path+"*.png")
   #mlist = ['/' + os.path.join(*f.split('/')[:-2]) + '/pre-mask/premask_' + f.split('/')[-1] for f in flist]
   mlist = flist
   for i in range(len(flist)):
      img_f = flist[i]
      mask_f = mlist[i]
      img = load_sample2(img_f)
      # Lower overall intensity to enable thresholding later
      img = lower_intensity_by_one(img)
      cv2.imwrite(mask_f, img)
    
    ## Setting background to black for manually masked images
    ##########################################################################
    # load images
    #dnnpath = "/home/doran/Work/py_code/experimental_DeCryptICS/DNN/input/"
    #dnnpath = "/home/doran/Work/py_code/DeCryptICS/DNN/input/"
    #dnnpath = "/home/doran/Work/py_code/DeCryptICS/DNN/input/mouse/"
    dnnpath = "/home/doran/Work/py_code/DeCryptICS/DNN/input/immune_nodes/"
    dnnpath = "/home/doran/Work/py_code/DeCryptICS/DNN/input/p53/"
    inpath = dnnpath + "/pre-mask/"
    outpath = dnnpath + "/train_masks/"
    imfiles = glob.glob(inpath + "*.png")

    # run processing and save
    for path in imfiles:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = set_background_to_black(img)
        #im_number = path.split("/")[-1][8:] # remove "premask_"
        im_number = path.split("/")[-1][4:] # remove "img_"
        outfile = "mask_" + im_number
        cv2.imwrite(outpath + outfile, mask)




    revertpath = dnnpath + "/pre-mask/"
    for path in imfiles:
       im_number = path.split("/")[-1][8:] # remove "premask_"        
       imfile = outpath + "mask_" + im_number # replace with mask
       img = cv2.imread(imfile, cv2.IMREAD_COLOR)        
       outfile = "img_" + im_number
       cv2.imwrite(revertpath + outfile, img)

   ## Thresholding masks incorrectly made in three-colour
   ##########################################################################
   training_base_folder = "/home/doran/Work/py_code/DeCryptICS/DNN/"
   maskfolder = training_base_folder + "/input/train_masks/"
   masks = glob.glob(maskfolder + "*.png")
   for f in masks:
      mm = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
      retval, threshold = cv2.threshold(mm, 1, 255, cv2.THRESH_BINARY)
      cv2.imwrite(f, threshold)
        
    ## Generating masks from predictions for new ground-truth training data
    ##########################################################################
    # Find non-problem files
#    file = open("/home/doran/Work/py_code/DNN_ImageMasking/input/problem_files.txt", 'r') 
#    flist = file.read() 
#    flist = flist.split('\n')[:-1]
#    mask_path = "/home/doran/Work/py_code/DNN_ImageMasking/input/train_masks/"
#    preds_path = "/home/doran/Work/py_code/DNN_ImageMasking/test_out/preds/"
#    allfiles = glob.glob(preds_path+"*.png")
#    allfiles = [f.split("/")[-1].split(".")[0] for f in allfiles]
#    allfiles = [f[5:] for f in allfiles]
#    predfiles = ["pred_"+f+".png" for f in allfiles if f not in flist]
#    for ff in predfiles:
#        mask = generate_mask_from_prediction(preds_path + ff, p_thresh=0.45)
#        outname = "mask" + ff[4:]
#        cv2.imwrite(mask_path + outname, mask)

    ## Resizing errant images/masks
    ##########################################################################
    dnnpath = "/home/doran/Work/py_code/experimental_DeCryptICS/DNN/input/"
    inpath = dnnpath + "/train/"
    imfiles = glob.glob(inpath + "*.png")
    maskpath = dnnpath + "/train_masks/"
    maskfiles = glob.glob(maskpath + "*.png")

    # run processing and save
    for path in imfiles:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if (img.shape!=(256,256,3)):
           img = img[:256, :256, :3]
           cv2.imwrite(path, img)
    
    for path in maskfiles:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if (mask.shape!=(256,256)):
           mask = mask[:256, :256]
           cv2.imwrite(path, mask)


    ## Setting background to black for manually masked images
    ##########################################################################
    # load images
    dnnpath = "/home/doran/Work/py_code/joining_serial_slides/DNN/input/"
    inpath = dnnpath + "/tissue_segment/mask3/"
    outpath = dnnpath + "/tissue_segment/mask3/"
    imfiles = glob.glob(inpath + "*.png")

    # run processing and save
    for path in imfiles:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = set_background_to_black(img)
        #im_number = path.split("/")[-1][8:] # remove "premask_"
        im_number = path.split("/")[-1][4:] # remove "img_"
        outfile = "mask_" + im_number
        cv2.imwrite(outpath + outfile, mask)

    
