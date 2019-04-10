import glob, cv2, pickle, os, random
import pandas as pd
import numpy as np
from MiscFunctions import plot_img, write_score_text_file
import argparse
import zipfile

# Want to cycle through images of clones and assign a yes no label to them, saving their local clone index and associated label
# then, using this new label and clone index, automatically update clone counts, patch indices, patch size and clone scores files

# load patch indices with pickle.load( open( folder_to_analyse + "/patch_indices.pickle", "rb" ) )

def plot_img_keep_decision(list_to_plot, nameWindow = 'Plots', NewWindow = True, hold_plot = True):
   if NewWindow:
      screen_res = 400, 400
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
                                                   "then the clone images are saved in the folder /base/path/Analysed_123456/clone_images "
                                                   "and these images are used to curate using the '1' and '0' keys for 'good clone' or 'bad clone', respectively "
                                                   "(alternatively the 'a' and 'd' keys can be used for good and bad). "
                                                   "Pressing 'p' will undo the last choice and go back to the previous image. "
                                                   "The slide counts, patch sizes and clone scores will be automatically updated to account for the manual curation. " )

   parser.add_argument("folder_to_analyse", help = "The full or relative path to the Analysed_123456 folder containing crypt/clone contours, "
                                                   "clone scores, patch sizes and indices for the slide 123456.svs. ")
   args = parser.parse_args()
   random.seed()
   folder_to_analyse = os.path.abspath(args.folder_to_analyse)
   clone_imgs = glob.glob(folder_to_analyse+"/clone_images/*.png")
   slide_number = folder_to_analyse.split('_')[-1].split('/')[0]
   good_inds = []
   bad_inds = []
   i = 0
   while i<(len(clone_imgs)+1):
      if i==len(clone_imgs):
         # allow one last 'p' in case final image was wrongly catergorised
         print("Clones all done -- last chance to press 'p' and go back...")
         decision_flag = plot_img_keep_decision(load_last())
         if (decision_flag == ord('p')):
            i -= 1
            if lastbinned=="good": good_inds = good_inds[:-1]
            if lastbinned=="bad": bad_inds = bad_inds[:-1]
         else:
            i += 1
      else:
         img = cv2.imread( clone_imgs[i] )
         ind = clone_imgs[i].split('/')[-1].split('_')[-1].split('.')[0]
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
         print("%d of %d" % (i+1 , len(clone_imgs)))

   # update clone scores to zero for bad inds
   clone_scores = np.loadtxt(folder_to_analyse + "/clone_scores.txt")
   clone_scores[bad_inds] = 0
   write_score_text_file(clone_scores, folder_to_analyse + "/clone_scores.txt")
   
   # generate new patch indices to remove any bad inds (but leave original pickle file untouched, as cannot recover patches otherwise)
   patch_indices = pickle.load( open( folder_to_analyse + "/patch_indices.pickle", "rb" ) )
   new_patch_indices = []
   patch_sizes = []
   for patch in patch_indices:
      new_patch = [ind for ind in patch if ind not in bad_inds]
      ll = len(new_patch)
      if ll>1:
         patch_sizes.append(ll)
         new_patch_indices.append(new_patch)
   patchsum = np.sum(patch_sizes)
   pickle.dump(new_patch_indices, open( folder_to_analyse + "/patch_indices_curated.pickle", "wb" ) )
   
   # then use new lists of indices to update patch size file
   write_score_text_file(patch_sizes, folder_to_analyse + "/patch_sizes.txt")
   
   # finally update the clone and patch counts in the batch csv using the slide number for the row reference
   counts_folder = '/'+os.path.join(*folder_to_analyse.split('/')[:-1])
   counts = np.asarray(pd.read_csv(counts_folder+"/slide_counts.csv"))
   slide_ind = np.where(counts[:,0]==int(slide_number))[0][0]
   counts[slide_ind, 3] = len(good_inds)
   counts[slide_ind, 4] = len(good_inds) - patchsum + len(patch_sizes)
   counts[slide_ind, 5] = len(patch_sizes)
   counts = pd.DataFrame(counts, columns=['Slide_ID', 'NCrypts', 'NFufis', 'NMutantCrypts', 'NClones', 'NPatches')
   counts.to_csv(counts_folder + 'slide_counts.csv', sep=',', index=False)

if __name__=="__main__":
   main()


