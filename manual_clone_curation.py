import glob, cv2, pickle, os, random, shutil
import pandas as pd
import numpy as np
import openslide as osl
from MiscFunctions import plot_img, write_score_text_file, getROI_img_osl, centred_tile,\
                          read_cnt_text_file, rescale_contours, write_cnt_text_file
import argparse
import zipfile

# Want to cycle through images of clones and assign a yes no label to them, saving their local clone index and associated label
# then, using this new label and clone index, automatically update clone counts, patch indices, patch size and clone scores files

def plot_img_keep_decision(list_to_plot, nameWindow = 'Plots', NewWindow = True, hold_plot = True, resolution = 800):
   if NewWindow:
      screen_res = (resolution, resolution)
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

def add_single_offset(contour, xy_offset):
   shape_prior = contour.shape
   cnt_l = contour.reshape((contour.shape[0], contour.shape[2]))
   new_cnt_l = cnt_l + np.round(xy_offset).astype(np.int32)
   new_cnt_l = new_cnt_l.reshape(shape_prior)
   return new_cnt_l

def pull_contour_check(img_path, cnts, points, scale, imgsize, dwnsmpl_lvl):
   # find the correct patch contour by checking centroids are inside
   inside_cnt = None
   count = 0
   for cnt_i in cnts:
      allgood = True
      for k in range(points.shape[0]):
         x = points[k,0]
         y = points[k,1]
         if not (cv2.pointPolygonTest(cnt_i, (int(x/scale), int(y/scale)), False)==0 or
                 cv2.pointPolygonTest(cnt_i, (int(x/scale), int(y/scale)), False)==1   ):
            allgood = False
            count += 1
            break
      if allgood==True:
         inside_cnt = cnt_i
         break

   cnt = inside_cnt
   # create bounded box
   bb_m = np.asarray(cv2.boundingRect(cnt))
   bb_m[0] = int(bb_m[0] - np.maximum(100, (imgsize - bb_m[2])/2.))
   bb_m[1] = int(bb_m[1] - np.maximum(100, (imgsize - bb_m[3])/2.))
   bb_m[2] = int(bb_m[2] + np.maximum(100, (imgsize - bb_m[2])))
   bb_m[3] = int(bb_m[3] + np.maximum(100, (imgsize - bb_m[3])))
   bb_m[bb_m<0] = 0
   img = getROI_img_osl(img_path, (bb_m[0], bb_m[1]), (bb_m[2], bb_m[3]), dwnsmpl_lvl)
   rec_cnt = add_single_offset(cnt, (-bb_m[0], -bb_m[1]))
   cv2.drawContours(img, [rec_cnt], 0, (80,80,50), 2)
   return img

def pull_contour(img_path, cnt, dwnsmpl_lvl, imgsize):
   # create bounded box
   bb_m = np.asarray(cv2.boundingRect(cnt))
   bb_m[0] = int(bb_m[0] - np.maximum(100, (imgsize - bb_m[2])/2.))
   bb_m[1] = int(bb_m[1] - np.maximum(100, (imgsize - bb_m[3])/2.))
   bb_m[2] = int(bb_m[2] + np.maximum(100, (imgsize - bb_m[2])))
   bb_m[3] = int(bb_m[3] + np.maximum(100, (imgsize - bb_m[3])))
   bb_m[bb_m<0] = 0
   img = getROI_img_osl(img_path, (bb_m[0], bb_m[1]), (bb_m[2], bb_m[3]), dwnsmpl_lvl)
   rec_cnt = add_single_offset(cnt, (-bb_m[0], -bb_m[1]))
   cv2.drawContours(img, [rec_cnt], 0, (80,80,50), 2)
   return img

def load_contours(filename, scale = None):
   cnts_m = read_cnt_text_file(filename)
   if (scale is not None):
      cnts_m = rescale_contours(cnts_m, 1./scale)
   return cnts_m

def load_last():
   ii = random.randint(1,151)
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

   parser.add_argument('-s', action = "store",
                        dest = "imgsize", 
                        default = 1024,
                        help = "Optionally set the size of the images used for curation. "
                               "Default set at 1024 pixels. ")
   parser.add_argument('-d', action = "store",
                        dest = "dwnsample", 
                        default = 0,
                        help = "Optionally set the (integer) downsample level of the images used for curation. "
                               "Default set at 0 (highest resolution). ")
   parser.add_argument('-r', action = "store",
                        dest = "resolution", 
                        default = 800,
                        help = "Optionally set the resolution of the pop-up window showing the curation images. "
                               "Default is 800 pixels. ")
                                                                              
                                                   
   args = parser.parse_args()
   dwnsmpl_lvl = int(args.dwnsample)
   resolution = int(args.resolution)
   imgsize = int(args.imgsize)
   random.seed()
   folder_to_analyse = os.path.abspath(args.folder_to_analyse)
   if "\\" in folder_to_analyse:
      listout = folder_to_analyse.split("\\")
      if listout[-1]=='':
         listout = listout[-2].split('_')[1:]
      else:
         listout = listout[-1].split('_')[1:]         
   if "/" in folder_to_analyse:
      listout = folder_to_analyse.split("/")
      if listout[-1]=='':
         listout = listout[-2].split('_')[1:]
      else:
         listout = listout[-1].split('_')[1:]
   sep = '_'
   slide_number = sep.join(listout)
   img_path = folder_to_analyse.split("Analysed")[-3] + slide_number + ".svs"
   slide = osl.OpenSlide(img_path)
   scale = slide.level_downsamples[dwnsmpl_lvl]
   ## create backup of crypt network data; replace with backup each time you run to allow do-overs
   ## (Also, re-running on an already-curated set of clones will mix up the indices, so either need to
   ##  always use the backup, or fix the issue.)
   if os.path.isfile(folder_to_analyse + "/crypt_network_data.txt"):
      if not os.path.isfile(folder_to_analyse + "/crypt_network_data.bac"):
         shutil.copy2(folder_to_analyse + "/crypt_network_data.txt", folder_to_analyse + "/crypt_network_data.bac")
      else:
         shutil.copy2(folder_to_analyse + "/crypt_network_data.bac", folder_to_analyse + "/crypt_network_data.txt")
   if os.path.isfile(folder_to_analyse + "/crypt_network_data.npy"):
      os.remove(folder_to_analyse + "/crypt_network_data.npy") # remove binary as may need to update with new one
   if os.path.isfile(folder_to_analyse + "/clone_scores.txt"):
      if not os.path.isfile(folder_to_analyse + "/clone_scores.bac"):
         shutil.copy2(folder_to_analyse + "/clone_scores.txt", folder_to_analyse + "/clone_scores.bac")
      else:
         shutil.copy2(folder_to_analyse + "/clone_scores.bac", folder_to_analyse + "/clone_scores.txt")
         
   if os.path.isfile(folder_to_analyse + "/patch_sizes.txt"):
      if not os.path.isfile(folder_to_analyse + "/patch_sizes.bac"):
         shutil.copy2(folder_to_analyse + "/patch_sizes.txt", folder_to_analyse + "/patch_sizes.bac")
      else:
         shutil.copy2(folder_to_analyse + "/patch_sizes.bac", folder_to_analyse + "/patch_sizes.txt")
   if os.path.isfile(folder_to_analyse + "/patch_contours.txt"):
      if not os.path.isfile(folder_to_analyse + "/patch_contours.bac"):
         shutil.copy2(folder_to_analyse + "/patch_contours.txt", folder_to_analyse + "/patch_contours.bac")
      else:
         shutil.copy2(folder_to_analyse + "/patch_contours.bac", folder_to_analyse + "/patch_contours.txt")
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
      
   ## Put a check in here to alter the patch sizes to match the number of clones with the unique id
   allclones = np.where(gg[:,3]>0)[0]
   patchid_inds = np.where(gg[:,5]>0)[0]
   sizes_ = gg[patchid_inds,4]
   ids_ = gg[patchid_inds,5].astype(np.int32)
   uniq_ids_ = np.unique(ids_).astype(np.int32)

   all_okay = False
   while all_okay==False:
      all_okay = True
      for jj in uniq_ids_:
         quoted_sizes = sizes_[np.where(ids_==jj)[0]]
         crypts_with_size = len(sizes_[np.where(ids_==jj)[0]])
         if (np.all(quoted_sizes==crypts_with_size)==False):
            all_okay = False
            # reduce the patch size for the rest of the patch
            thispatch = np.where(gg[:,5]==jj)[0]
            for k in thispatch:
               gg[k,4] = crypts_with_size
            if len(thispatch)==1:
               # no longer a patch
               gg[thispatch[0],4] = 0
               gg[thispatch[0],5] = 0
      # recalculate arrays
      allclones = np.where(gg[:,3]>0)[0]
      patchid_inds = np.where(gg[:,5]>0)[0]
      sizes_ = gg[patchid_inds,4]
      ids_ = gg[patchid_inds,5].astype(np.int32)
      uniq_ids_ = np.unique(ids_).astype(np.int32)
   
#   #tests 
#   for jj in uniq_ids_:
#      print(sizes_[np.where(ids_==jj)[0]])
#      print(jj)
#      print(np.all(sizes_[np.where(ids_==jj)[0]]==len(sizes_[np.where(ids_==jj)[0]])))
      
   # get patch contours
   patchcnts_fn = folder_to_analyse + "/patch_contours.txt"
   patchcnts = load_contours(patchcnts_fn, scale = scale)

   good_patches = []
   bad_patches = []
   examine_inds = []
   lastbinned_l = []
   i = 0
   print("Use '1' or 'a' for good patches and '0' or 'd' for bad patches!")
   print("Use '3' or 'w' for patches that you want to curate in more detail afterwards.")
   print("(Or press 'p' to undo the last choice and go back to the previous image.)")
   while i<(len(uniq_ids_)+1):
      if i==len(uniq_ids_):
         # allow one last 'p' in case final image was wrongly catergorised
         print("Patches all done -- last chance to press 'p' and go back...")
         decision_flag = plot_img_keep_decision(load_last(), resolution = resolution)
         if (decision_flag == ord('p') and i>0):
            if lastbinned_l[-1]=="good": good_patches = good_patches[:-1]
            if lastbinned_l[-1]=="bad": bad_patches = bad_patches[:-1]
            if lastbinned_l[-1]=="unsure":
               examine_inds = examine_inds[:-1]
            # roll back binning
            lastbinned_l = lastbinned_l[:-1]
            i -= 1
         elif (decision_flag == ord('p') and i==0): pass
         else:
            i += 1
      else:
         ind = uniq_ids_[i]
         thissize = sizes_[np.where(ids_==ind)[0][0]]
         print("patch size = %d" % int(thissize))
#         thiscnt = patchcnts[i]
#         img = pull_contour(img_path, thiscnt, dwnsmpl_lvl, imgsize)
         pnts = gg[np.where(gg[:,5]==ind)[0], :2]
         img = pull_contour_check(img_path, patchcnts, pnts, scale, imgsize, dwnsmpl_lvl)
         decision_flag = plot_img_keep_decision(img, resolution = resolution)
         if (decision_flag == ord('a') or decision_flag == ord('1')):
            good_patches.append(int(ind))
            i += 1
            lastbinned_l.append("good")
         elif (decision_flag == ord('w') or decision_flag == ord('3')):
            examine_inds.append(int(ind))
            i += 1
            lastbinned_l.append("unsure")
         elif (decision_flag == ord('d') or decision_flag == ord('0')):
            bad_patches.append(int(ind))
            i += 1
            lastbinned_l.append("bad")
         elif (decision_flag == ord('p') and i>0):
            if lastbinned_l[-1]=="good": good_patches = good_patches[:-1]
            if lastbinned_l[-1]=="bad": bad_patches = bad_patches[:-1]
            if lastbinned_l[-1]=="unsure":
               examine_inds = examine_inds[:-1]
            # roll back binning
            lastbinned_l = lastbinned_l[:-1]
            i -= 1
         elif (decision_flag == ord('p') and i==0): pass
         elif (decision_flag == ord('h')):
            print("Exit signal received. Breaking out.")
            break
         else:
            print("Use '1' or 'a' for good patches and '0' or 'd' for bad patches!")
            print("Use '3' or 'w' for patches that you want to curate in more detail afterwards.")
            print("(Or press 'p' to undo the last choice and go back to the previous image.)")
            print("Repeating image...")
         print("%d of %d" % (i+1 , len(uniq_ids_)))
   cv2.destroyAllWindows()
   if (decision_flag == ord('h')):
      print("Quitting!")
      return 0
   
   if (len(uniq_ids_)>0):
      # update graph file
      for kk in bad_patches:
         cinds = patchid_inds[np.where(ids_==kk)[0]]
         for j in cinds:
            gg[j,3] = 0 # mutant
            gg[j,4] = 0 # patch size
            cur_pid = gg[j,5]
            gg[j,5] = 0 # patch id
            if cur_pid>0:
               # reduce the patch size for the rest of the patch
               thispatch = np.where(gg[:,5]==cur_pid)[0]
               for k in thispatch:
                  gg[k,4] = gg[k,4] - 1
               if len(thispatch)==1:
                  # no longer a patch
                  gg[thispatch[0],4] = 0
                  gg[thispatch[0],5] = 0
   
   # find the clones remaining after patch curation
   clone_inds = np.setdiff1d(np.where(gg[:,3]>0)[0], patchid_inds)
   extra_clone_inds = []
   for ii in range(len(examine_inds)):
      extra_clone_inds = extra_clone_inds + [ps for ps in patchid_inds[np.where(ids_==examine_inds[ii])[0]]]
   clone_inds = np.hstack([clone_inds, np.asarray(extra_clone_inds)]).astype(np.int32)
   
   good_inds = []
   bad_inds = []
   partials = []
   lastbinned_l = []
   i = 0
   print("The pertinent clone will be centered in the images that pop up.")
   print("Use '1' or 'a' for good clones and '0' or 'd' for bad clones!")
   print("Use '3' or 'w' for partial clones, which will also label them 'good'.")
   print("(Or press 'p' to undo the last choice and go back to the previous image.)")
   print("If you want to quit, press 'h'.")
   while i<(len(clone_inds)+1):
      if i==len(clone_inds):
         # allow one last 'p' in case final image was wrongly catergorised
         print("Clones all done -- last chance to press 'p' and go back...")
         decision_flag = plot_img_keep_decision(load_last(), resolution = resolution)
         if (decision_flag == ord('p') and i>0):
            if lastbinned_l[-1]=="good": good_inds = good_inds[:-1]
            if lastbinned_l[-1]=="bad": bad_inds = bad_inds[:-1]
            if lastbinned_l[-1]=="partial":
               good_inds = good_inds[:-1]
               partials = partials[:-1]
            # roll back binning
            lastbinned_l = lastbinned_l[:-1]
            i -= 1
         elif (decision_flag == ord('p') and i==0): pass
         else:
            i += 1
      else:
         ind = clone_inds[i]
         xy_vals = centred_tile(gg[ind,:2]/scale, imgsize, slide.level_dimensions[dwnsmpl_lvl])
         img = getROI_img_osl(img_path, xy_vals, (imgsize, imgsize), dwnsmpl_lvl)
         decision_flag = plot_img_keep_decision(img, resolution = resolution)
         if (decision_flag == ord('a') or decision_flag == ord('1')):
            good_inds.append(int(ind))
            i += 1
            lastbinned_l.append("good")
         elif (decision_flag == ord('w') or decision_flag == ord('3')):
            good_inds.append(int(ind))
            partials.append(int(ind))
            i += 1
            lastbinned_l.append("partial")
         elif (decision_flag == ord('d') or decision_flag == ord('0')):
            bad_inds.append(int(ind))
            i += 1
            lastbinned_l.append("bad")
         elif (decision_flag == ord('p') and i>0):
            if lastbinned_l[-1]=="good": good_inds = good_inds[:-1]
            if lastbinned_l[-1]=="bad": bad_inds = bad_inds[:-1]
            if lastbinned_l[-1]=="partial":
               good_inds = good_inds[:-1]
               partials = partials[:-1]
            # roll back binning
            lastbinned_l = lastbinned_l[:-1]
            i -= 1
         elif (decision_flag == ord('p') and i==0): pass
         elif (decision_flag == ord('h')):
            print("Exit signal received. Breaking out.")
            break
         else:
            print("Use '1' or 'a' for good clones and '0' or 'd' for bad clones!")
            print("Use '3' or 'w' for partial clones, which will also label them 'good'.")
            print("(Or press 'p' to undo the last choice and go back to the previous image.)")
            print("Repeating image...")
         print("%d of %d" % (i+1 , len(clone_inds)))
   cv2.destroyAllWindows()
   if (decision_flag == ord('h')):
      print("Quitting!")
      return 0
      
   if (len(clone_inds)==0):
      print("No clones to curate, doing nothing!")
      return 0
      
   if (len(clone_inds)>0):
      # update graph file
      for j in bad_inds:
         gg[j,3] = 0 # mutant
         gg[j,4] = 0 # patch size
         cur_pid = gg[j,5]
         gg[j,5] = 0 # patch id
         if cur_pid>0:
            # reduce the patch size for the rest of the patch
            thispatch = np.where(gg[:,5]==cur_pid)[0]
            for k in thispatch:
               gg[k,4] = gg[k,4] - 1
            if len(thispatch)==1:
               # no longer a patch
               gg[thispatch[0],4] = 0
               gg[thispatch[0],5] = 0

      with open(graphpath, 'w') as fo:
            fo.write("#<x>\t<y>\t<fufi>\t<mutant>\t<patch_size>\t<patch_id>\t<area>\t<eccentricity>\t<major_axis>\t<minor_axis>\n")
            for i in range(gg.shape[0]):
               fo.write("%d\t%d\t%d\t%d\t%d\t%d\t%1.8g\t%1.8g\t%1.8g\t%1.8g\n" % (gg[i,0], gg[i,1], gg[i,2], gg[i,3], gg[i,4], gg[i,5], gg[i,6], gg[i,7], gg[i,8], gg[i,9]))

      # update clone scores to zero for bad inds, one for good inds
      local_bad_inds = [np.where(allclones==c)[0][0] for c in bad_inds]
      local_good_inds = [np.where(allclones==c)[0][0] for c in good_inds]
      for kk in bad_patches:
         cinds = patchid_inds[np.where(ids_==kk)[0]]
         local_pinds = [np.where(allclones==c)[0][0] for c in cinds]
         local_bad_inds = local_bad_inds + local_pinds
      for kk in good_patches:
         cinds = patchid_inds[np.where(ids_==kk)[0]]
         local_pinds = [np.where(allclones==c)[0][0] for c in cinds]
         local_good_inds = local_good_inds + local_pinds
            
      clone_scores = np.loadtxt(folder_to_analyse + "/clone_scores.txt", ndmin=1)
      clone_scores[local_bad_inds] = 0
      clone_scores[local_good_inds] = 1
      write_score_text_file(clone_scores, folder_to_analyse + "/clone_scores.txt")

      unique_patches = np.vstack(list({tuple(row) for row in gg[allclones, 4:6]})) # size, id
      numpatches = unique_patches.shape[0] - 1 # get rid of zero-zero row
      patchsum = int(np.sum(unique_patches, axis=0)[0])
      patch_sizes = unique_patches[unique_patches[:,0]>0, 0].astype(int)

      # then use new lists of indices to update patch size file
      write_score_text_file(patch_sizes, folder_to_analyse + "/patch_sizes.txt")
      
      # update the patch contour file using the new graph data
      patchid_inds = np.where(gg[:,5]>0)[0]
      uids_ = (np.unique(gg[patchid_inds, 5]) - 1).astype(np.int32) # to start at 0
      patch_contours_new = [patchcnts[ct] for ct in uids_]
      write_cnt_text_file(patch_contours_new, folder_to_analyse + "/patch_contours.txt")
      
      # finally update the clone and patch counts in the batch csv using the slide number for the row reference
      if '\\' in folder_to_analyse:
         pathsep = '\\'
         listsep = folder_to_analyse.split('\\')
         if listsep[-1]=='':
            listsep = listsep[:-2]
         else:
            listsep = listsep[:-1]
         counts_folder = pathsep.join(listsep)
      if '/' in folder_to_analyse:
         listsep = folder_to_analyse.split('/')
         if listsep[-1]=='':
            listsep = listsep[:-2]
         else:
            listsep = listsep[:-1]      
         counts_folder = '/'+os.path.join(*listsep)
      counts_pd = pd.read_csv(counts_folder+"/slide_counts.csv")
      colnames = np.asarray(counts_pd.columns)         
      counts = np.asarray(pd.read_csv(counts_folder+"/slide_counts.csv"))      
      slide_ind = np.where(counts[:,0].astype(str)==str(slide_number))[0][0]
      counts[slide_ind, np.where(colnames=='NMutantCrypts')[0][0]] = len(local_good_inds)
      counts[slide_ind, np.where(colnames=='NClones')[0][0]] = len(local_good_inds) - patchsum + numpatches
      counts[slide_ind, np.where(colnames=='NPatches')[0][0]] = numpatches
      # check if partials column exists
      if 'NPartials' in colnames:
         counts[slide_ind, np.where(colnames=='NPartials')[0][0]] = len(partials)
      else:
         parcnts = np.zeros((counts.shape[0], 1), dtype=np.int32)
         parcnts[slide_ind,0] = len(partials)
         counts = np.hstack([counts, parcnts])
      counts = pd.DataFrame(counts, columns=['Slide_ID', 'NCrypts', 'NFufis', 'NMutantCrypts', 'NClones', 'NPatches', 'NPartials'])
      counts.to_csv(counts_folder + '/slide_counts.csv', sep=',', index=False)

if __name__=="__main__":
   main()


