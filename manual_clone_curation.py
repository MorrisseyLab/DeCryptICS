import glob, cv2, pickle, os, random, shutil
import pandas as pd
import numpy as np
import openslide as osl
from MiscFunctions import plot_img, write_score_text_file, getROI_img_osl, centred_tile
import argparse
import zipfile

# Want to cycle through images of clones and assign a yes no label to them, saving their local clone index and associated label
# then, using this new label and clone index, automatically update clone counts, patch indices, patch size and clone scores files

def plot_img_keep_decision(list_to_plot, nameWindow = 'Plots', NewWindow = True, hold_plot = True):
   if NewWindow:
      screen_res = 800, 800
      cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
      cv2.resizeWindow(nameWindow, screen_res[0], screen_res[1])
   vis = list_to_plot     
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

def main():
   parser = argparse.ArgumentParser(description = "This script allows manual curation of clones for a single slide. "
                                                   "If the crypt and clone contours for a slide are saved in /base/path/Analysed_123456, "
                                                   "then the clone images are loaded from /base/path/123456.svs using coordinates from  "
                                                   "/base/path/Analysed_123456/crypt_network_data.txt. "
                                                   "Clone images are curated using the '1' and '0' keys for 'good clone' or 'bad clone', respectively "
                                                   "(alternatively the 'a' and 'd' keys can be used for good and bad). "
                                                   "Pressing 'p' will undo the last choice and go back to the previous image. "
                                                   "The slide counts, patch sizes and clone scores will be automatically updated to account for the manual curation. " )

   parser.add_argument("folder_to_analyse", help = "The full or relative path to the Analysed_XXXXX folder containing crypt/clone contours, "
                                                   "clone scores, patch sizes and crypt_network_data for the slide XXXXX.svs. ")
   args = parser.parse_args()
   random.seed()
   folder_to_analyse = os.path.abspath(args.folder_to_analyse)
   if "\\" in folder_to_analyse:
      listout = folder_to_analyse.split("\\")[-1].split('_')[1:] 
   if "/" in folder_to_analyse:
      listout = folder_to_analyse.split("/")[-1].split('_')[1:]
   sep = '_'
   slide_number = sep.join(listout)
   img_path = folder_to_analyse.split("Analysed")[-3] + slide_number + ".svs"
   dwnsmpl_lvl = 1
   slide = osl.OpenSlide(img_path)
   scale = slide.level_downsamples[dwnsmpl_lvl]
   imgsize = 256
   ## create backup of crypt network data
   if os.path.isfile(folder_to_analyse + "/crypt_network_data.txt"):
      shutil.copy2(folder_to_analyse + "/crypt_network_data.txt", folder_to_analyse + "/crypt_network_data.bac")
   ## load graph of slide analysis output
   graphpath = folder_to_analyse + "/crypt_network_data.txt"
   try:
      gg = np.loadtxt(graphpath)
      if gg.shape[0]==0:
         gg = np.zeros(10)
   except:
      gg = np.zeros(10)
   if len(gg.shape)==1:
      gg = gg.reshape((1,gg.shape[0]))
   clone_inds = np.where(gg[:,3]>0)[0]
   good_inds = []
   bad_inds = []
   i = 0
   print("The pertinent clone will be centered in the images that pop up.")
   print("Use '1' or 'a' for good clones and '0' or 'd' for bad clones!")
   print("(Or press 'p' to undo the last choice and go back to the previous image.)")   
   while i<(len(clone_inds)+1):
      if i==len(clone_inds):
         # allow one last 'p' in case final image was wrongly catergorised
         print("Clones all done -- last chance to press 'p' and go back...")
         decision_flag = plot_img_keep_decision(load_last())
         if (decision_flag == ord('p') and i>0):
            i -= 1
            if lastbinned=="good": good_inds = good_inds[:-1]
            if lastbinned=="bad": bad_inds = bad_inds[:-1]
         elif (decision_flag == ord('p') and i==0): pass
         else:
            i += 1
      else:
         ind = clone_inds[i]
         xy_vals = centred_tile(gg[ind,:2]/scale, imgsize, slide.level_dimensions[dwnsmpl_lvl])
         img = getROI_img_osl(img_path, xy_vals, (imgsize, imgsize), dwnsmpl_lvl)
         decision_flag = plot_img_keep_decision(img)
         if (decision_flag == ord('a') or decision_flag == ord('1')):
            good_inds.append(int(ind))
            i += 1
            lastbinned = "good"
         elif (decision_flag == ord('d') or decision_flag == ord('0')):
            bad_inds.append(int(ind))
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
         print("%d of %d" % (i+1 , len(clone_inds)))
   cv2.destroyAllWindows()
   if (len(clone_inds)==0):
      print("No clones to curate, doing nothing!")
      return 0
   if (len(clone_inds)>0):
      # update graph file
      for j in bad_inds:
         gg[j,3] = 0 # mutant
         gg[j,4] = 0 # patch size
         gg[j,5] = 0 # patch id

      with open(graphpath, 'w') as fo:
            fo.write("#<x>\t<y>\t<fufi>\t<mutant>\t<patch_size>\t<patch_id>\t<area>\t<eccentricity>\t<major_axis>\t<minor_axis>\n")
            for i in range(gg.shape[0]):
               fo.write("%d\t%d\t%d\t%d\t%d\t%d\t%1.8g\t%1.8g\t%1.8g\t%1.8g\n" % (gg[i,0], gg[i,1], gg[i,2], gg[i,3], gg[i,4], gg[i,5], gg[i,6], gg[i,7], gg[i,8], gg[i,9]))

      # update clone scores to zero for bad inds
      local_bad_inds = [np.where(clone_inds==c)[0][0] for c in bad_inds]
      local_good_inds = [np.where(clone_inds==c)[0][0] for c in good_inds]
      clone_scores = np.loadtxt(folder_to_analyse + "/clone_scores.txt", ndmin=1)
      clone_scores[local_bad_inds] = 0
      clone_scores[local_good_inds] = 1
      write_score_text_file(clone_scores, folder_to_analyse + "/clone_scores.txt")
      # can we also edit patch contours? Or simply get rid of them?

      unique_patches = np.vstack({tuple(row) for row in gg[clone_inds, 4:6]})
      numpatches = unique_patches.shape[0] - 1 # get rid of zero-zero row
      patchsum = int(np.sum(unique_patches, axis=0)[0])
      patch_sizes = unique_patches[unique_patches[:,0]>0, 0].astype(int)

      # then use new lists of indices to update patch size file
      write_score_text_file(patch_sizes, folder_to_analyse + "/patch_sizes.txt")
      
      # finally update the clone and patch counts in the batch csv using the slide number for the row reference
      if "\\" in folder_to_analyse:
         pathsep = "\\"
         listsep = folder_to_analyse.split('\\')[:-1]
         counts_folder = pathsep.join(listsep)
      if "/" in folder_to_analyse:
         counts_folder = '/'+os.path.join(*folder_to_analyse.split('/')[:-1])
      counts = np.asarray(pd.read_csv(counts_folder+"/slide_counts.csv"))
      slide_ind = np.where(counts[:,0]==slide_number)[0][0]
      counts[slide_ind, 3] = len(good_inds)
      counts[slide_ind, 4] = len(good_inds) - patchsum + numpatches
      counts[slide_ind, 5] = numpatches
      counts = pd.DataFrame(counts, columns=['Slide_ID', 'NCrypts', 'NFufis', 'NMutantCrypts', 'NClones', 'NPatches'])
      counts.to_csv(counts_folder + 'slide_counts.csv', sep=',', index=False)

if __name__=="__main__":
   main()


