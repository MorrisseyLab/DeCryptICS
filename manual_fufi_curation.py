import glob, cv2, pickle, os, random, shutil
import pandas as pd
import numpy as np
import openslide as osl
import matplotlib.pyplot as plt
from MiscFunctions import plot_img, write_score_text_file, getROI_img_osl, centred_tile,\
                          read_cnt_text_file, rescale_contours, write_cnt_text_file
from  process_output import construct_event_counts
import argparse
import zipfile

class Matplotlib_hack:
   def __init__(self, img):
      self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      self.event_key = np.nan
   
   def close_figure(self, event):
      if (event.key == '1' or
         event.key == '0' or 
         event.key == '3' or
         event.key == 'a' or 
         event.key == 'w' or 
         event.key == 'd' or 
         event.key == 'p' or
         event.key == 'h'):
         plt.close(event.canvas.figure)
         self.event_key = ord(event.key)
   
   def plot(self):
      plt.imshow(self.img)
      plt.gcf().canvas.mpl_connect('key_press_event', self.close_figure)
      plt.show()

def plot_img_keep_decision(list_to_plot, nameWindow = 'Plots', NewWindow = True, hold_plot = True, resolution = 800):
   try:
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
   except:
      inkey = np.nan
      mpl_fig = Matplotlib_hack(list_to_plot)
      mpl_fig.plot()
      plt.pause(1)
      inkey = mpl_fig.event_key
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
   if os.path.isfile(curdir+"/sprites/"+str(ii)+".png"):
      img = cv2.imread(curdir+"/sprites/"+str(ii)+".png")
   elif os.path.isfile(curdir+"/sprites.zip"):   
      with zipfile.ZipFile(curdir+"/sprites.zip", 'r') as zfile:
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
                               
   parser.add_argument('-b', action = "store_true",
                        default = False,
                        help = "Revert to backup version before running curation. Use this if a mistake was made lase time you ran the curation. ")
                                                                              
                                                   
   args = parser.parse_args()
   dwnsmpl_lvl = int(args.dwnsample)
   resolution = int(args.resolution)
   imgsize = int(args.imgsize)
   bac = args.b
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
   if os.path.isfile(folder_to_analyse + "/processed_output.csv"):
      if not os.path.isfile(folder_to_analyse + "/processed_output.bac"):
         shutil.copy2(folder_to_analyse + "/processed_output.csv", folder_to_analyse + "/processed_output.bac")
   if os.path.isfile(folder_to_analyse + "/processed_output.npy"):
      os.remove(folder_to_analyse + "/processed_output.npy") # remove binary as may need to update with new one
   if os.path.isfile(folder_to_analyse + "/clone_mask.txt"):
      if not os.path.isfile(folder_to_analyse + "/clone_mask.bac"):
         shutil.copy2(folder_to_analyse + "/clone_mask.txt", folder_to_analyse + "/clone_mask.bac")
   if os.path.isfile(folder_to_analyse + "/patch_contours.txt"):
      if not os.path.isfile(folder_to_analyse + "/patch_contours.bac"):
         shutil.copy2(folder_to_analyse + "/patch_contours.txt", folder_to_analyse + "/patch_contours.bac")
         
   if bac==True:
      try:
         shutil.copy2(folder_to_analyse + "/processed_output.bac", folder_to_analyse + "/processed_output.csv")
         shutil.copy2(folder_to_analyse + "/clone_mask.bac", folder_to_analyse + "/clone_mask.txt")
         shutil.copy2(folder_to_analyse + "/patch_contours.bac", folder_to_analyse + "/patch_contours.txt")
      except:
         print('No backups found.')
         
   ## load graph of slide analysis output
   graphpath = folder_to_analyse + '/processed_output.csv'
   try:
      gg = pd.read_csv(graphpath)
      colnames = gg.columns
      fufi_col = np.where(colnames == 'p_fufi')[0][0]
      x_col = np.where(colnames == 'x')[0][0]
      y_col = np.where(colnames == 'y')[0][0]
      if gg.shape[0]==0:
         gg = np.zeros(19)
   except:
      gg = np.zeros(19)
      clone_col = 4
      x_col = 1
      y_col = 2
   if len(gg.shape)==1:
      gg = gg.reshape((1,gg.shape[0]))      
   gg = np.asarray(gg)
   
   fufi_inds = np.where(gg[:,fufi_col]>0)[0]
   good_inds = []
   bad_inds = []
   lastbinned_l = []
   i = 0
   print("The pertinent clone will be centered in the images that pop up.")
   print("Use '1' or 'a' for good fufis and '0' or 'd' for bad fufis!")
   print("(Or press 'p' to undo the last choice and go back to the previous image.)")
   print("If you want to quit, press 'h'.")
   while i<(len(fufi_inds)+1):
      if i==len(fufi_inds):
         # allow one last 'p' in case final image was wrongly catergorised
         print("Fufis all done -- last chance to press 'p' and go back...")
         decision_flag = plot_img_keep_decision(load_last(), resolution = resolution)
         if (decision_flag == ord('p') and i>0):
            if lastbinned_l[-1]=="good": good_inds = good_inds[:-1]
            if lastbinned_l[-1]=="bad": bad_inds = bad_inds[:-1]
            # roll back binning
            lastbinned_l = lastbinned_l[:-1]
            i -= 1
         elif (decision_flag == ord('p') and i==0): pass
         else:
            i += 1
      else:
         ind = fufi_inds[i]
         xy_vals = centred_tile(gg[ind, x_col:(y_col+1)]/scale, imgsize, slide.level_dimensions[dwnsmpl_lvl])
         img = getROI_img_osl(img_path, xy_vals, (imgsize, imgsize), dwnsmpl_lvl)
         decision_flag = plot_img_keep_decision(img, resolution = resolution)
         if (decision_flag == ord('a') or decision_flag == ord('1')):
            good_inds.append(int(ind))
            i += 1
            lastbinned_l.append("good")
         elif (decision_flag == ord('d') or decision_flag == ord('0')):
            bad_inds.append(int(ind))
            i += 1
            lastbinned_l.append("bad")
         elif (decision_flag == ord('p') and i>0):
            if lastbinned_l[-1]=="good": good_inds = good_inds[:-1]
            if lastbinned_l[-1]=="bad": bad_inds = bad_inds[:-1]
            # roll back binning
            lastbinned_l = lastbinned_l[:-1]
            i -= 1
         elif (decision_flag == ord('p') and i==0): pass
         elif (decision_flag == ord('h')):
            print("Exit signal received. Breaking out.")
            break
         else:
            print("Use '1' or 'a' for good clones and '0' or 'd' for bad clones!")
            print("(Or press 'p' to undo the last choice and go back to the previous image.)")
            print("Repeating image...")
         print("%d of %d" % (i+1 , len(fufi_inds)))
   cv2.destroyAllWindows()
   if (decision_flag == ord('h')):
      print("Quitting!")
      return 0
      
   if (len(fufi_inds)==0):
      print("No fufis to curate, doing nothing!")
      return 0
      
   if (len(fufi_inds)>0):
      # update graph file
      for j in bad_inds:
         gg[j, fufi_col] = 0
      out_df = pd.DataFrame(gg)
      out_df.columns = colnames
      out_df.to_csv(graphpath, index=False) 

      # update clone scores to zero for bad inds, one for good inds
      clone_scores = np.loadtxt(folder_to_analyse + "/fufi_mask.txt", ndmin=1)
      clone_scores[np.array(bad_inds, dtype=np.int32)] = 0
      clone_scores[np.array(good_inds, dtype=np.int32)] = 1
      write_score_text_file(clone_scores, folder_to_analyse + "/fufi_mask.txt")
      
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
      slide_ind = np.where(counts_pd['Slide_ID'].astype(str)==str(slide_number))[0][0]      
      slide_counts = np.asarray(counts_pd)[:,1:]
      slide_nums = np.asarray(counts_pd)[:,0]
      
      slide_counts = construct_event_counts(slide_counts, out_df, slide_ind)
      
      slidecounts_p = pd.DataFrame(slide_counts, columns=colnames[1:])
      slidecounts_p = pd.concat([pd.DataFrame({'Slide_ID':slide_nums}), slidecounts_p], axis=1)
      slidecounts_p.to_csv(counts_folder+"/slide_counts.csv", sep=',', index=False)

if __name__=="__main__":
   main()


