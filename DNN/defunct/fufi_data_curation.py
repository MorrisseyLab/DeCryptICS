import glob, cv2, pickle, os, random
import pandas as pd
import numpy as np
from MiscFunctions import plot_img
import argparse
import zipfile

def plot_img_keep_decision(list_to_plot, nrow = 1,  nameWindow = 'Plots', NewWindow = True, hold_plot = True):
   num_images = len(list_to_plot)
   num_cols   = int(num_images/nrow)
   if num_images%nrow != 0:
      raise(UserWarning, "If more than one row make sure there are enough images!")
   if NewWindow:
      screen_res = 1600, 1000
      cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
      cv2.resizeWindow(nameWindow, screen_res[0], screen_res[1])
   if isinstance(list_to_plot, tuple) == 0: 
      vis = list_to_plot
   else:
      last_val = num_cols 
      vis      = np.concatenate(list_to_plot[0:last_val], axis=1)
      for row_i in range(1, nrow):
         first_val = last_val
         last_val  = first_val + num_cols
         #            print (first_val,last_val)
         vis_aux   = np.concatenate(list_to_plot[first_val:last_val], axis=1)
         vis       = np.concatenate((vis, vis_aux), axis=0)        
   cv2.imshow(nameWindow, vis)
   if (hold_plot):
      inkey = 0xFF & cv2.waitKey(0)
      cv2.destroyWindow(nameWindow)
      cv2.destroyAllWindows()
      cv2.waitKey(1)
   return inkey

def load_last():
   ii = random.randint(1,152)
   curdir = os.getcwd()
   if os.path.isfile(curdir+"/DNN/input/sprites/"+str(ii)+".png"):
      img = cv2.imread(curdir+"/DNN/input/sprites/"+str(ii)+".png")
   elif os.path.isfile(curdir+"/DNN/input/sprites/sprites.zip"):   
      with zipfile.ZipFile(curdir+"/DNN/input/sprites/sprites.zip", 'r') as zfile:
         data = zfile.read(str(ii)+".png")
      img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)    
   else:
      img = np.zeros([50,50])
   return img

if __name__=="__main__":

   folder_to_analyse = "/home/doran/Work/py_code/DeCryptICS/DNN/input/fufis/"
   fufi_imgs = glob.glob(folder_to_analyse+"/train/*.png")
   imgfolder = folder_to_analyse+"/train/"
   maskfolder = folder_to_analyse+"/train_masks/"
   good_inds = []
   bad_inds = []
   i = 0
   while i<(len(fufi_imgs)+1):
      if i==len(fufi_imgs):
         # allow one last 'p' in case final image was wrongly catergorised
         print("Fufis all done -- last chance to press 'p' and go back...")
         decision_flag = plot_img_keep_decision(load_last())
         if (decision_flag == ord('p')):
            i -= 1
            if lastbinned=="good": good_inds = good_inds[:-1]
            if lastbinned=="bad": bad_inds = bad_inds[:-1]
         else:
            i += 1
      else:
         img = cv2.imread( fufi_imgs[i] , cv2.IMREAD_GRAYSCALE )
         mask_f = maskfolder+"mask"+fufi_imgs[i][(len(imgfolder)+3):]
         mask = cv2.imread( mask_f , cv2.IMREAD_GRAYSCALE )
         decision_flag = plot_img_keep_decision((img, mask))
         if (decision_flag == ord('a') or decision_flag == ord('1')):
            good_inds.append((fufi_imgs[i]))
            i += 1
            lastbinned = "good"
         elif (decision_flag == ord('d') or decision_flag == ord('0')):
            bad_inds.append((fufi_imgs[i]))
            i += 1
            lastbinned = "bad"
         elif (decision_flag == ord('p')):
            i -= 1
            if lastbinned=="good": good_inds = good_inds[:-1]
            if lastbinned=="bad": bad_inds = bad_inds[:-1]
         else:
            print("Use '1' or 'a' for good clones and '0' or 'd' for bad clones!")
            print("(Or press 'p' to undo the last choice and go back to the previous image.)")
            print("Repeating image...")
         print("%d of %d" % (i+1 , len(fufi_imgs)))

   imgs_good = []
   masks_good = []
   for i in range(len(good_inds)):
      imgs_good.append(good_inds[i][0])
      masks_good.append(good_inds[i][1])
   imgs_good = np.asarray(imgs_good)
   masks_good = np.asarray(masks_good)
   imgs_bad = []
   masks_bad = []
   for i in range(len(bad_inds)):
      imgs_bad.append(bad_inds[i][0])
      masks_bad.append(bad_inds[i][1])
   imgs_bad = np.asarray(imgs_bad)
   masks_bad = np.asarray(masks_bad)

   imgs_bad2 = []
   for i in range(len(bad_inds)):
      imgs_bad2.append(bad_inds[i][0])
   imgs_bad2 = np.asarray(imgs_bad2)


   for i in range(len(imgs_good_trim)):
      name_i = imgs_good_trim[i].split('/')[-1]
      mask_f = maskfolder+"mask"+imgs_good_trim[i][(len(imgfolder)+3):]
      name_m = mask_f.split('/')[-1]
      os.rename(imgs_good_trim[i], imgfolder+"/good/"+name_i)
      os.rename(mask_f, maskfolder+"/good/"+name_m)
   for i in range(len(imgs_bad)):
      name_i = imgs_bad[i].split('/')[-1]
      mask_f = maskfolder+"mask"+imgs_bad[i][(len(imgfolder)+3):]
      name_m = mask_f.split('/')[-1]
      os.rename(imgs_bad[i], imgfolder+"/bad/"+name_i)
      os.rename(mask_f, maskfolder+"/bad/"+name_m)
      
