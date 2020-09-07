import glob, cv2, pickle, os, random, shutil
import pandas as pd
import numpy as np
import openslide as osl
from MiscFunctions import plot_img, write_score_text_file, getROI_img_osl, centred_tile,\
                          read_cnt_text_file, rescale_contours, write_cnt_text_file
import argparse
import zipfile
import tensorflow as tf
#from DNN.autocuration.mutant_net import *
from DNN.autocuration.context_net import *
from DNN.autocuration.datagen import read_slide_data
from DNN.autocuration.run_generator5 import run_generator

num_cores = 16
GPU = True
CPU = False

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

tilesize = 512 # 50
sizesmall = 384
#nn = 30
#nn_sampsize = 200
input_size1 = (tilesize, tilesize, 3)
input_size2 = (sizesmall, sizesmall, 3)
chan_num = 3

def main():
#   parser = argparse.ArgumentParser(description = "This script allows manual curation of clones for a single slide. "
#                                                   "If the crypt and clone contours for a slide are saved in /base/path/Analysed_123456, "
#                                                   "then the clone images are loaded from /base/path/123456.svs using coordinates from  "
#                                                   "/base/path/Analysed_123456/crypt_network_data.txt. "
#                                                   "Clone images are curated using the '1' and '0' keys for 'good clone' or 'bad clone', respectively "
#                                                   "(alternatively the 'a' and 'd' keys can be used for good and bad). "
#                                                   "Pressing 'p' will undo the last choice and go back to the previous image. "
#                                                   "The slide counts, patch sizes and clone scores will be automatically updated to account for the manual curation. " )

#   parser.add_argument("folder_to_analyse", help = "The full or relative path to the Analysed_XXXXX folder containing crypt/clone contours, "
#                                                   "clone scores, patch sizes and crypt_network_data for the slide XXXXX.svs. ")

#   parser.add_argument('-t', action = "store",
#                        dest = "threshold", 
#                        default = 0.15,
#                        help = "The probability threshold on calling mutant clones. ")
#   parser.add_argument('-s', choices = ['c', 'a'],
#                             default = 'c', 
#                             dest    = "test_set",
#                             help    = "Choose the set of crypts to test; the clone candidates (c) or all crypts (a). ")
#   parser.add_argument('-b', action = "store",
#                        dest = "batch_size", 
#                        default = 40,
#                        help = "The batch size of tests. ")
#            
#   args = parser.parse_args()
#   threshold = int(args.threshold)
#   test_set = int(args.test_set)
#   batch_size = int(args.batch_size)
#   random.seed()
#   folder_to_analyse = os.path.abspath(args.folder_to_analyse)

   batch_size = 25
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

   ## create backup of crypt network data; replace with backup each time you run to allow do-overs
   ## (Also, re-running on an already-curated set of clones will mix up the indices, so either need to
   ##  always use the backup, or fix the issue.)
#   if os.path.isfile(folder_to_analyse + "/crypt_network_data.txt"):
#      if not os.path.isfile(folder_to_analyse + "/crypt_network_data.bac"):
#         shutil.copy2(folder_to_analyse + "/crypt_network_data.txt", folder_to_analyse + "/crypt_network_data.bac")
#      else:
#         shutil.copy2(folder_to_analyse + "/crypt_network_data.bac", folder_to_analyse + "/crypt_network_data.txt")
#         
#   if os.path.isfile(folder_to_analyse + "/clone_scores.txt"):
#      if not os.path.isfile(folder_to_analyse + "/clone_scores.bac"):
#         shutil.copy2(folder_to_analyse + "/clone_scores.txt", folder_to_analyse + "/clone_scores.bac")
#      else:
#         shutil.copy2(folder_to_analyse + "/clone_scores.bac", folder_to_analyse + "/clone_scores.txt")
#         
#   if os.path.isfile(folder_to_analyse + "/patch_sizes.txt"):
#      if not os.path.isfile(folder_to_analyse + "/patch_sizes.bac"):
#         shutil.copy2(folder_to_analyse + "/patch_sizes.txt", folder_to_analyse + "/patch_sizes.bac")
#      else:
#         shutil.copy2(folder_to_analyse + "/patch_sizes.bac", folder_to_analyse + "/patch_sizes.txt")
#   if os.path.isfile(folder_to_analyse + "/patch_contours.txt"):
#      if not os.path.isfile(folder_to_analyse + "/patch_contours.bac"):
#         shutil.copy2(folder_to_analyse + "/patch_contours.txt", folder_to_analyse + "/patch_contours.bac")
#      else:
#         shutil.copy2(folder_to_analyse + "/patch_contours.bac", folder_to_analyse + "/patch_contours.txt")
         
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
   allclones = np.where(gg[:,3]>0)[0]
   patchsize_inds = np.where(gg[:,4]>0)[0]
   patchid_inds = np.where(gg[:,5]>0)[0]
   sizes_ = gg[patchsize_inds, 4]
   ids_ = gg[patchid_inds, 5].astype(np.int32)
   uniq_ids_ = np.unique(ids_).astype(np.int32)

   ## create model and load weights
   dnnfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/autocuration/"
   model = att_roi_net2(input_shape1=input_size1, input_shape2=input_size2,
                        d_model=64, depth_k=6, depth_v=8, num_heads=2, dff=128, dropout_rate=0.3)
   model.load_weights(dnnfolder + "/weights/att_roi_net_w2.hdf5")
#   model = inet5(input_size1, input_size2, chan_num, modtype=1)   
#   model.load_weights(dnnfolder + "/weights/autocurate_sqex_stack5.hdf5")

   ## read slide data
   slide_data, cnts, imgpath = read_slide_data(folder_to_analyse, folder_to_analyse)
   ## create image generator
   rungen = run_generator(imgpath, slide_data, cnts, batch_size)
   ## predict on batches
   num_to_test = slide_data.shape[0]
   clone_probs = np.empty(num_to_test, dtype=np.float32)
   st = 0
   t1 = time.time()
   for i in range(len(rungen)+1):
      if (i%5==0): print(i)
      tb = rungen[i]
      pred = model.predict(tb)
      en = st + tb[0].shape[0]
      clone_probs[st:en] = pred[:,0]
      st = en
   t2 = time.time()
   time_taken = t2-t1
   print("Time taken for slide: %1.2f minutes" % (time_taken/60.))
   slide_data = np.column_stack([slide_data, clone_probs])
   np.save(folder_to_analyse + 'autoclone_probs2.npy', slide_data)
   
   
   slide_data = np.load(folder_to_analyse + 'autoclone_probs2.npy')
   from sklearn.metrics import confusion_matrix
   cf = confusion_matrix(slide_data[:,3].astype(bool) , slide_data[:,-1]>0.5)

   # checks
   cp_cut = slide_data[:,-1]#[:en]
   mask = slide_data[:,3]#[:en, 3]
   plt.figure(1)
   plt.hist(abs(mask-cp_cut))
   plt.figure(2)
   inds_1 = np.where(mask==1)[0]
   plt.hist(cp_cut[inds_1])
   plt.figure(3)
   inds_0 = np.where(mask==0)[0]   
   plt.hist(cp_cut[inds_0])


badinds = np.where(abs(mask-cp_cut) > 0.5)[0]
for i in range(500,520,1):
   tb = rungen[badinds[i]]
   pred = model.predict(tb)
   print(pred)
   plot_img(tb[1][0,:,:,:])
   



#   ## divide slide up into chunks and define a reference stack for each chunk that is then re-used
#   from sklearn.cluster import KMeans
#   nclusts = 8
#   kmeans = KMeans(n_clusters=nclusts, init='k-means++', n_init = 1, max_iter = 150).fit(slide_data[:,:2])
#   chunks = np.unique(kmeans.labels_)
#   chunks_lab = kmeans.labels_
#   slide_data = np.column_stack([slide_data, chunks_lab])
#   # create random stacks for each chunk, to re-use for any candidate crypt from that chunk
#   for ch in chunks:
#      sub_data = sub_data[in_inds,:]
#      sldinds = sub_data[:,2].astype(np.int32)
#      ## pull crypts and return stack
#      crypt_stack = np.zeros((scr_size, scr_size, 3*nn), dtype=np.uint8)
#      if multicore:
#         cv2.setNumThreads(0)
#         num_cores = multiprocessing.cpu_count()
#         results = Parallel(n_jobs=num_cores)(delayed(pull_crypt_from_cnt)(sub_data[ii,:2], imgpath, 
#                                                      cnts[int(sub_data[ii,2])], scr_size, ROTATE=False, 
#                                                      RT_m=0, dwnsample_lvl=1) for ii in range(in_inds.shape[0]))
#         for ii in range(in_inds.shape[0]):
#            crypt_stack[:,:,(3*ii):(3*(ii+1))] = results[ii]
#      else:
#      for ii in range(in_inds.shape[0]):
#         tind = int(sub_data[ii,2])
#         crypt_stack[:,:,(3*ii):(3*(ii+1))] = pull_crypt_from_cnt(sub_data[ii,:2], imgpath, 
#                                                                  cnts[tind], scr_size, ROTATE=False, 
#                                                                  RT_m=0, dwnsample_lvl=1)















   ###########################
   
   ## get required paths for image pulling
#   img_path = folder_to_analyse.split("Analysed")[-3] + slide_number + ".svs"
#   contour_path = folder_to_analyse + "/crypt_contours.txt"   
   
   ## create validation set
   folders_validation = ["/home/doran/Work/images/autocurate_validation/Analysed_slides/Analysed_642739/",
                         "/home/doran/Work/images/autocurate_validation/Analysed_slides/Analysed_586572/",
                         "/home/doran/Work/images/autocurate_validation/Analysed_slides/Analysed_595794/"]
   datapath = "/home/doran/Work/py_code/DeCryptICS/DNN/autocuration/data/"
   validation_set = np.empty((0,7), dtype=object)
   for fsl in folders_validation:
      positive_data = np.empty((0,7), dtype=object)
      negative_data = np.empty((0,7), dtype=object)
      all_dat = np.empty((0,7), dtype=object)
      # get curated crypt net data, find image and get slide
      netdat = np.loadtxt(fsl + '/crypt_network_data.txt')
      clone_inds = np.where(netdat[:,3]>0)[0]
      numclones = clone_inds.shape[0]
      imname = fsl.split('Analysed_')[-1].split('/')[0]
      imloc = fsl.split("Analysed_slides")[0]
      impath = imloc + imname + '.svs'
      mark = 'dummy'
      cnts_m_in = load_all_contours2(impath)[0]
      if not (len(cnts_m_in)==netdat.shape[0]): continue
      contour_path = fsl + '/crypt_contours.txt'
      # add positive data
      if numclones>0:
         this_netdat = netdat[clone_inds,:4]
         this_netdat[:,2] = clone_inds
         this_netdat = np.column_stack([this_netdat.astype(object),
                                        np.repeat(mark, numclones).astype(object),
                                        np.repeat(impath, numclones).astype(object),
                                        np.repeat(contour_path, numclones).astype(object)])
         positive_data = np.row_stack([positive_data , this_netdat])
      # add negative data
      this_negdat = netdat[np.setdiff1d(range(netdat.shape[0]), clone_inds),:4]
      this_negdat[:,2] = np.setdiff1d(range(netdat.shape[0]), clone_inds)
      numnegs = this_negdat.shape[0]
      this_negdat = np.column_stack([this_negdat.astype(object),
                                     np.repeat(mark, numnegs).astype(object),
                                     np.repeat(impath, numnegs).astype(object),
                                     np.repeat(contour_path, numnegs).astype(object)])
      negative_data = np.row_stack([negative_data , this_negdat])
      all_dat = np.row_stack([positive_data, negative_data])
      ## output slide data
      np.save(datapath + imname + '_data.npy', all_dat[:,:4].astype(np.float32))
      ## output contours as numpy array
      np.save(datapath + imname + '_cnts.npy', np.asarray(cnts_m_in)) 
      ## take sample for validation set  
      sampinds = np.random.choice(range(all_dat.shape[0]), size = 300, replace=False)
      validation_set = np.row_stack([validation_set, all_dat[sampinds,:]])
   
   if (save_val_set==True):
      ## output validation data
      np.save("/home/doran/Work/py_code/DeCryptICS/DNN/autocuration/data/validation_data.npy", validation_set)

   ## get data format table
   if test_set.lower()[0]=='c': # test candidate clones
      this_netdat = gg[allclones,:4]
      this_netdat[:,2] = allclones
   if test_set.lower()[0]=='a': # test all crypts
      this_netdat = gg[:,:4]
      this_netdat[:,2] = np.asarray(range(gg.shape[0]), dtype=np.int32)
   if test_set.lower()[0]=='s': # test some WT, some clones
      take_n = 20
      this_netdat1 = gg[allclones[:take_n],:4]
      this_netdat1[:,2] = allclones[:take_n]
      this_netdat2 = gg[:take_n,:4]
      this_netdat2[:,2] = np.asarray(range(take_n), dtype=np.int32)
      this_netdat = np.row_stack([this_netdat1, this_netdat2])
      np.random.shuffle(this_netdat)
   num_to_test = this_netdat.shape[0]   
   this_netdat = np.column_stack([this_netdat.astype(object),
                                  np.repeat(mark, num_to_test).astype(object),
                                  np.repeat(img_path, num_to_test).astype(object),
                                  np.repeat(contour_path, num_to_test).astype(object)])
   
   ## load model
   model = inet(input_size1, input_size2, chan_num)
   model2 = inet(input_size1, input_size2, chan_num, modtype=2)
   maindir = os.path.dirname(os.path.abspath(__file__))
   weightsin = os.path.join(maindir, 'DNN', 'autocuration', 'weights', 'autocurate_net3.hdf5')
   model.load_weights(weightsin)
   model.load_weights("./DNN/autocuration/weights/autocurate_net3.hdf5")
   
   model2.load_weights("./DNN/autocuration/weights/autocurate_net.hdf5")
   
   ## create image batch generator
   im_gen = run_generator(tilesize, sizesmall, this_netdat, batch_size)

   ## test by running through batches
   i = 0
   b1 = im_gen[i]
   for k in range(b1[0].shape[0]):
      b1[0][k,:,:,:] = 0 # black-out an image
   p1 = model.predict(b1)
   p2 = model2.predict(b1)
   for k in range(b1[0].shape[0]):
      print("pred1: %1.2f" % p1[k,0])
      print("pred2: %1.2f" % p2[k,0])
      plot_img(b1[0][k,:,:,:], hold_plot=False, nameWindow="a")
      plot_img(b1[1][k,:,:,:])

   ## predict all test set in batches
   clone_probs = np.empty(num_to_test, dtype=np.float32)
   st = 0
   t1 = time.time()
   for i in range(len(im_gen)):
      tb = im_gen[i]
      pred = model.predict(tb)
      en = st + tb[0].shape[0]
      clone_probs[st:en] = pred[:,0]
      st = en
   t2 = time.time()
   time_taken = t2-t1
   print("Time taken for slide: %1.2f minutes" % (time_taken/60.))
   np.save(folder_to_analyse + 'autoclone_probs.npy', clone_probs)

   ## still getting ~100s of "positive" clones in some slides! Get more data?
   ## is it the WTs around/close to clones that are being falsely called? i.e. "non-centered" calling

#   clone_probs = np.empty(num_to_test, dtype=np.float32)
#   for i in range(num_to_test):
#      tb = grab_test_batch(this_netdat[i,:])
#      pred = model.predict(tb)
#      print(i)
#      print(pred)
##      plot_img(tb[0][0,:,:,:], hold_plot=False, nameWindow="a")
##      plot_img(tb[1][0,:,:,:], hold_plot=True, nameWindow="b")
#      clone_probs[i] = pred[0][0]
      
   if (num_to_test==0):
      print("No clones to curate, doing nothing!")
      return 0
   
   ######## Now work out what to do with the predictions!   
   if (len(clone_inds)>0):
      # update graph file
      for j in bad_inds:
         gg[j,3] = 0 # mutant
         gg[j,4] = 0 # patch size
         cur_pid = gg[j,5]
         gg[j,5] = 0 # patch id
         if cur_pid>0:
            thispatch = np.where(gg[:,5]==cur_pid)[0]
            for k in thispatch:
               gg[k,4] = gg[k,4] - 1
            if len(thispatch)==1:
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

      unique_patches = np.vstack(list({tuple(row) for row in gg[allclones, 4:6]}))
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


