# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:18:27 2015

@author: doran
"""
import cv2
import numpy as np
import matplotlib.pylab as plt
import itertools
import math
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
from MiscFunctions         import *
from cnt_Feature_Functions import *
from classContourFeat      import getAllFeatures
from knn_prune             import tukey_lower_thresholdval, tukey_upper_thresholdval
from clone_analysis_funcs  import *

def write_clone_features_to_file(cfl, out_folder):
   num_crypts = cfl[0].shape[0]
   num_bins = cfl[2].shape[1]
   
   # output 1D features
   with open(out_folder + '/clone_features_1Dstats.tsv', 'w') as fname:
      fname.write("halo_n\thalo_c\tx\ty\tcontent_n\tcontent_c\n")
      for i in range(0, num_crypts):
         fname.write("%1.10g\t%1.10g\t%1.10g\t%1.10g\t%1.10g\t%1.10g\t%1.10g\n" % (cfl['global_inds'][i], cfl['halo_n'][i], cfl['halo_c'][i], cfl['xy_coords'][i][0], cfl['xy_coords'][i][1], cfl['content_n'][i], cfl['content_c'][i]))
   
   # output matrix features
   with open(out_folder + '/clone_features_out_sig_n.tsv', 'w') as fname:
      for i in range(0, num_crypts):
         for j in range(0, num_bins-1):         
            fname.write("%1.10g\t" % cfl['out_bins_n'][i][j])
         fname.write("%1.10g\n" % cfl['out_bins_n'][i][num_bins-1])
   with open(out_folder + '/clone_features_out_sig_c.tsv', 'w') as fname:
      for i in range(0, num_crypts):
         for j in range(0, num_bins-1):         
            fname.write("%1.10g\t" % cfl['out_bins_c'][i][j])
         fname.write("%1.10g\n" % cfl['out_bins_c'][i][num_bins-1])
   with open(out_folder + '/clone_features_in_sig_n.tsv', 'w') as fname:
      for i in range(0, num_crypts):
         for j in range(0, num_bins-1):         
            fname.write("%1.10g\t" % cfl['in_bins_n'][i][j])
         fname.write("%1.10g\n" % cfl['in_bins_n'][i][num_bins-1])
   with open(out_folder + '/clone_features_in_sig_c.tsv', 'w') as fname:
      for i in range(0, num_crypts):
         for j in range(0, num_bins-1):         
            fname.write("%1.10g\t" % cfl['in_bins_c'][i][j])
         fname.write("%1.10g\n" % cfl['in_bins_c'][i][num_bins-1])    

def create_empty_clone_feature_dict(numcnts, nbins):
   cfl = {} # clone_feature_list
   cfl['xy_coords']  = np.zeros([numcnts, 2], dtype=np.int32)
   cfl['halo_n']     = np.zeros([numcnts])
   cfl['halo_c']     = np.zeros([numcnts])
   cfl['content_n']  = np.zeros([numcnts])
   cfl['content_c']  = np.zeros([numcnts])
   cfl['out_bins_n'] = np.zeros([numcnts, nbins], dtype=np.float32)
   cfl['out_bins_c'] = np.zeros([numcnts, nbins], dtype=np.float32)
   cfl['in_bins_n']  = np.zeros([numcnts, nbins], dtype=np.float32)
   cfl['in_bins_c']  = np.zeros([numcnts, nbins], dtype=np.float32)
   #cfl['mid_contour'] = []
   return cfl

def read_clone_features_from_file(out_folder):
   # load features
   clone_features_1Dstats   = np.loadtxt(out_folder + '/clone_features_1Dstats.tsv'  , skiprows=1)
   clone_features_out_sig_n = np.loadtxt(out_folder + '/clone_features_out_sig_n.tsv')
   clone_features_out_sig_c = np.loadtxt(out_folder + '/clone_features_out_sig_c.tsv')
   clone_features_in_sig_n  = np.loadtxt(out_folder + '/clone_features_in_sig_n.tsv' )
   clone_features_in_sig_c  = np.loadtxt(out_folder + '/clone_features_in_sig_c.tsv' )
   
   # form dict
   cfl = create_empty_clone_feature_dict(clone_features_1Dstats.shape[0], clone_features_in_sig_c.shape[1])
   cfl['halo_n']    = clone_features_1Dstats[:,0]
   cfl['halo_c']    = clone_features_1Dstats[:,1]
   cfl['xy_coords'] = np.zeros([cfl['halo_n'].shape[0], 2], dtype=np.int32)
   cfl['xy_coords'][:,0] = clone_features_1Dstats[:,2]
   cfl['xy_coords'][:,1] = clone_features_1Dstats[:,3]
   cfl['content_n'] = clone_features_1Dstats[:,4]
   cfl['content_c'] = clone_features_1Dstats[:,5]
   cfl['out_bins_n'] = clone_features_out_sig_n
   cfl['out_bins_c'] = clone_features_out_sig_c
   cfl['in_bins_n'] = clone_features_in_sig_n
   cfl['in_bins_c'] = clone_features_in_sig_c
   return cfl

def find_clone_statistics(crypt_cnt, img_nuc, img_clone, nbins = 20):
   # for each contour do no_threshold_signal_collating()
   numcnts = len(crypt_cnt)
   cfl = create_empty_clone_feature_dict(numcnts, nbins)
   in_bins_c  = np.zeros([numcnts, nbins], dtype=np.float32)
   for i in range(numcnts):
      contour = crypt_cnt[i]
      X = no_threshold_signal_collating(contour, img_nuc, img_clone, nbins)
      cfl['halo_n'][i]        = X[0]
      cfl['halo_c'][i]        = X[1]
      cfl['out_bins_n'][i, :] = X[2]
      cfl['out_bins_c'][i, :] = X[3]
      cfl['xy_coords'][i,0]   = X[4][0]
      cfl['xy_coords'][i,1]   = X[4][1]
      cfl['in_bins_n'][i, :]  = X[5]
      cfl['in_bins_c'][i, :]  = X[6]
      cfl['content_n'][i]     = X[7]
      cfl['content_c'][i]     = X[8]  
      #cfl['mid_contour'].append(X[9])
   return cfl

def combine_feature_lists(clone_feature_list, numcnts, nbins = 20):
   cfl = create_empty_clone_feature_dict(numcnts, nbins)
   cumnum = 0
   for i in range(len(clone_feature_list)):
      curr_num = clone_feature_list[i]['halo_n'].shape[0]
      cfl['halo_n'][cumnum:(cumnum+curr_num)]        = clone_feature_list[i]['halo_n']
      cfl['halo_c'][cumnum:(cumnum+curr_num)]        = clone_feature_list[i]['halo_c']
      cfl['out_bins_n'][cumnum:(cumnum+curr_num), :] = clone_feature_list[i]['out_bins_n']
      cfl['out_bins_c'][cumnum:(cumnum+curr_num), :] = clone_feature_list[i]['out_bins_c']
      cfl['xy_coords'][cumnum:(cumnum+curr_num),0]   = clone_feature_list[i]['xy_coords'][:,0]
      cfl['xy_coords'][cumnum:(cumnum+curr_num),1]   = clone_feature_list[i]['xy_coords'][:,1]
      cfl['in_bins_n'][cumnum:(cumnum+curr_num), :]  = clone_feature_list[i]['in_bins_n']
      cfl['in_bins_c'][cumnum:(cumnum+curr_num), :]  = clone_feature_list[i]['in_bins_c']
      cfl['content_n'][cumnum:(cumnum+curr_num)]     = clone_feature_list[i]['content_n']
      cfl['content_c'][cumnum:(cumnum+curr_num)]     = clone_feature_list[i]['content_c']
      #cfl['mid_contour'] = cfl['mid_contour'] + clone_feature_list[i]['mid_contour']
      cumnum = cumnum + curr_num
   return cfl

def subset_clone_features(cfl, kept_indices, keep_global_inds=True):
   cflnew = create_empty_clone_feature_dict(kept_indices.shape[0], cfl['out_bins_n'].shape[1])
   cflnew['xy_coords'][:,0] = cfl['xy_coords'][kept_indices,0]
   cflnew['xy_coords'][:,1] = cfl['xy_coords'][kept_indices,1]
   cflnew['halo_n']         = cfl['halo_n'][kept_indices]
   cflnew['halo_c']         = cfl['halo_c'][kept_indices]
   cflnew['out_bins_n']     = cfl['out_bins_n'][kept_indices, :]
   cflnew['out_bins_c']     = cfl['out_bins_c'][kept_indices, :]
   cflnew['in_bins_n']      = cfl['in_bins_n'][kept_indices, :]
   cflnew['in_bins_c']      = cfl['in_bins_c'][kept_indices, :]
   cflnew['content_n']      = cfl['content_n'][kept_indices]
   cflnew['content_c']      = cfl['content_c'][kept_indices]
   if (keep_global_inds):
      cflnew['global_inds']  = kept_indices
   else:
      glob_inds = np.where(cflnew['halo_n'])[0]
      cflnew['global_inds']  = glob_inds
   return cflnew 
   
def add_xy_offset_to_clone_features(cfl, xy_offset):
   for i in range(cfl['halo_n'].shape[0]):
      cfl['xy_coords'][i, 0] = cfl['xy_coords'][i, 0] + xy_offset[0]
      cfl['xy_coords'][i, 1] = cfl['xy_coords'][i, 1] + xy_offset[1]
   return cfl

def determine_clones_gridways(cfl, clonal_mark_type):
   # cut up clone_feature_list into roughly equal chunks (~2000 crypts each)
   groupcryptnum = 2000.
   numcnts = cfl['halo_n'].shape[0]
   xy_coords_all = cfl['xy_coords']
   pc_y = 100.*math.sqrt(groupcryptnum/numcnts)
   num_y = int(np.ceil(100./pc_y))
   pc_y = 100./num_y
   highlim = 0
   clone_signal_width = np.array([-1])
   clone_inds = np.array([-1])
   for i in range(1,num_y+1):
      lowlim = highlim
      highlim = np.percentile(xy_coords_all[:,1], i*pc_y)
      inds_yl = np.where(xy_coords_all[:,1]>lowlim)[0]
      inds_yh = np.where(xy_coords_all[:,1]<highlim)[0]
      inds_y = np.intersect1d(inds_yl,inds_yh)
      # divide x
      pc_x = groupcryptnum/xy_coords_all[inds_y,1].shape[0] * 100.
      num_x = int(np.ceil(100./pc_x))
      pc_x = 100./num_x
      highlimx = 0
      for j in range(1,num_x+1):
         lowlimx = highlimx
         highlimx = np.percentile(xy_coords_all[inds_y,0], j*pc_x)
         inds_xl = np.where(xy_coords_all[inds_y,0]>lowlimx)[0]
         inds_xh = np.where(xy_coords_all[inds_y,0]<highlimx)[0]
         inds_x = np.intersect1d(inds_xl,inds_xh)
         inds = inds_y[inds_x] # note these are not the global indices
         grid_feats = subset_clone_features(cfl, inds, keep_global_inds=True)
         newinds, newwidth = determine_clones(grid_feats, clonal_mark_type)
         clone_inds = np.hstack([clone_inds, newinds])
         clone_signal_width = np.hstack([clone_signal_width, newwidth])
   clone_signal_width = clone_signal_width[1:]
   clone_inds = clone_inds[1:]
   return clone_inds, clone_signal_width  

def find_outlier_truncated_normal(Signal, below=True, numSD=2, nbins=100):
   # extract symmetric parent normal distribution from data
   hist_oc = np.histogram(Signal, bins=nbins)
   midbin = np.argmax(hist_oc[0])
   maxinds = np.argpartition(hist_oc[0], len(hist_oc[0])-3)[-3:] # finding index of top few bins
   maxvals = np.partition(hist_oc[0], len(hist_oc[0])-3)[-3:]
   #mu = (hist_oc[1][midbin] + hist_oc[1][midbin+1])/2.
   # point between the top few bins, weighted av
   #mu = (hist_oc[1][maxinds[0]]*hist_oc[0][maxinds[0]] + hist_oc[1][maxinds[1]]*hist_oc[0][maxinds[1]] + hist_oc[1][maxinds[2]]*hist_oc[0][maxinds[2]]) / np.sum(hist_oc[0][maxinds])
   # or: weighted average of bins between maxinds
   iis = list(range(min(maxinds), max(maxinds)+1))
   mu = 0
   for ii in iis:
      mu += hist_oc[1][ii]*hist_oc[0][ii]
   mu /= np.sum(hist_oc[0][iis])
   rightside_dist = Signal[np.where(Signal>=mu)]
   leftside_dist = -(rightside_dist-mu) + mu
   fulldist = np.hstack([leftside_dist, rightside_dist])
   sd = np.std(fulldist)   
   # form centered distribution using truncation limits
   a = 0
   b = 1e32
   alpha = (a-mu)/sd
   beta = (b-mu)/sd
   dist = stats.norm(0, 1) # centred distribution from Z-scoring
   phi_beta = dist.pdf(beta)
   Phi_beta = dist.cdf(beta)
   phi_alpha = dist.pdf(alpha)
   Phi_alpha = dist.cdf(alpha)
   # calculate mean/sd of truncated distribution
   mu_trunc = mu - sd * (phi_beta - phi_alpha)/(Phi_beta - Phi_alpha)
   sd_trunc = sd * np.sqrt(1 - (beta*phi_beta - alpha*phi_alpha)/(Phi_beta - Phi_alpha) - ((phi_beta - phi_alpha)/(Phi_beta - Phi_alpha))**2)
   if (below): return mu_trunc - numSD*sd_trunc
   if (not below): return mu_trunc + numSD*sd_trunc

def determine_clones(cfl, clonal_mark_type, crypt_contours = 0):
   ## NEW METHOD OF CLONE FINDING (perhaps globally over a whole slide?):
   # - To find those bins that are `nuclear halo dropouts' (i.e. points 
   #  where we haven't got inside the halo, or where the crypt is missing
   #  part of the halo): scatter plot of each bin in each crypt, 
   #  out_bins_n against out_bins_c. Mean/Variance of main
   #  blob in x,y should allow us to find real clones as outliers in y
   #  but not in x.
   # - Then use the remaining bins in the same was as already, to calculate 
   #  signal width for each crypt and judge it as a clone or not.
   
   halo_n     = cfl['halo_n']
   halo_c     = cfl['halo_c']
   out_bins_n = cfl['out_bins_n']
   out_bins_c = cfl['out_bins_c']
   xy_coords  = cfl['xy_coords']
   in_bins_n  = cfl['in_bins_n']
   in_bins_c  = cfl['in_bins_c']
   content_n  = cfl['content_n']
   content_c  = cfl['content_c']
   globinds   = np.linspace(0, halo_n.shape[0]-1, halo_n.shape[0])
   globscores = np.zeros(halo_n.shape[0])

   # also -- use size distribution to first remove contours from the clone finding algorithm
   # (but don't remove them from the outputted crypts; use local and global indices)
   if not (crypt_contours==0):
      numrawcnts = len(crypt_contours)
      size = np.zeros(numrawcnts)
      for i in range(0,numrawcnts):
         size[i] = contour_Area(crypt_contours[i])
      goodsizeinds = np.where(size>=1)[0] # defunct

   halo_n_s1     = cfl['halo_n'][goodsizeinds]
   halo_c_s1     = cfl['halo_c'][goodsizeinds]
   out_bins_n_s1 = cfl['out_bins_n'][goodsizeinds,:]
   out_bins_c_s1 = cfl['out_bins_c'][goodsizeinds,:]
   xy_coords_s1  = cfl['xy_coords'][goodsizeinds,:]
   in_bins_n_s1  = cfl['in_bins_n'][goodsizeinds,:]
   in_bins_c_s1  = cfl['in_bins_c'][goodsizeinds,:]
   content_n_s1  = cfl['content_n'][goodsizeinds]
   content_c_s1  = cfl['content_c'][goodsizeinds]
   globinds_s1   = globinds[goodsizeinds]

   ## Finding bad bins in nuclear channel
   ###########################################
   numcnts = out_bins_n_s1.shape[0]
   numbins = out_bins_n_s1.shape[1]
   
   # nuclear channel signal and outlier lower/upper bound
   if (clonal_mark_type[2]=='N' or clonal_mark_type[2]=='B'): # nuclear or both, use outer signal
      Signal_out_n = out_bins_n_s1.ravel()
      outlier_below_n = find_outlier_truncated_normal(Signal_out_n, numSD = 2.)
   if (clonal_mark_type[2]=='L'): # lumen, use inner signal
      Signal_in_n = in_bins_n_s1.ravel()
      outlier_above_n = find_outlier_truncated_normal(Signal_in_n, below = False, numSD = 2.)
      
   # clonal channel mark-dependent signal and outlier lower/upper bound
   if (clonal_mark_type[2]=='N' or clonal_mark_type[2]=='B'): # nuclear or both, use outer signal
      Signal_out_c = out_bins_c_s1.ravel()
      if (clonal_mark_type[0]=='N'): # negative, use outlier below lower bound
         outlier_below_c = find_outlier_truncated_normal(Signal_out_c, numSD = 2.) # 2.25
      if (clonal_mark_type[0]=='P'): # positive, use outlier above upper bound
         outlier_above_c = find_outlier_truncated_normal(Signal_out_c, below = False, numSD = 2.) # 2.25
   if (clonal_mark_type[2]=='L'): # lumen, use inner signal
      Signal_in_c = in_bins_c_s1.ravel()
      if (clonal_mark_type[0]=='N'): # negative, use outlier below lower bound
         outlier_below_c = find_outlier_truncated_normal(Signal_in_c, numSD = 2.) # 2.25
      if (clonal_mark_type[0]=='P'): # positive, use outlier above upper bound
         outlier_above_c = find_outlier_truncated_normal(Signal_in_c, below = False, numSD = 2.) # 2.25
   
   ## Find good and bad bins using nuclear channel
   badbins = []
   goodbins = []
   allbins = list(range(0,numbins))

   if (clonal_mark_type[2]=='N' or clonal_mark_type[2]=='B'): # nuclear or both, use outer signal
      bins_c = out_bins_c_s1
      bins_n = out_bins_n_s1
   if (clonal_mark_type[2]=='L'): # lumen, use inner signal
      bins_c = in_bins_c_s1
      bins_n = in_bins_n_s1

   if (clonal_mark_type[0]=='N'): # negative, use outlier below lower bound 
      for i in range(0,numcnts):
         badbins.append([])
         goodbins.append([])
         for j in range(0,numbins):
            if (bins_c[i,j]<outlier_below_c and bins_n[i,j]<outlier_below_n):
               badbins[i].append(j)
            else: goodbins[i].append(j)
   if (clonal_mark_type[0]=='P'): # positive, use outlier above upper bound
      for i in range(0,numcnts):
         badbins.append([])
         goodbins.append([])
         for j in range(0,numbins):
            if (bins_c[i,j]>outlier_above_c and bins_n[i,j]<outlier_below_n):
               badbins[i].append(j)
            else: goodbins[i].append(j)   
      
   ## define flattened signals with bad bins removed for outlier calculation
   if (clonal_mark_type[2]=='N' or clonal_mark_type[2]=='B'): # nuclear or both, use outer signal
      flat_signal_oc = np.array([])
      flat_signal_on = np.array([])
      for k in range(numcnts):
         flat_signal_oc = np.hstack([flat_signal_oc, out_bins_c_s1[k, goodbins[k]]])
         flat_signal_on = np.hstack([flat_signal_on, out_bins_n_s1[k, goodbins[k]]])
   if (clonal_mark_type[2]=='L'): # lumen, use inner signal
      flat_signal_ic = np.array([])
      flat_signal_in = np.array([])
      for k in range(numcnts):
         flat_signal_ic = np.hstack([flat_signal_ic, in_bins_c_s1[k, goodbins[k]]])
         flat_signal_in = np.hstack([flat_signal_in, in_bins_n_s1[k, goodbins[k]]])

   ## Calculating global signal width with badbin dropout
   ###########################################
   clone_signal_width = np.zeros(numcnts)
   clone_signal_total = np.zeros(numcnts)
   if (clonal_mark_type[0]=='N'): # negative, use outlier below lower bound
      outlier = outlier_below_c
   if (clonal_mark_type[0]=='P'): # positive, use outlier above upper bound
      outlier = outlier_above_c
   if (clonal_mark_type[2]=='N' or clonal_mark_type[2]=='B'): # nuclear or both, use outer signal
      for k in range(numcnts):
         clone_signal_width[k], clone_signal_total[k] = signal_width_ndo(out_bins_c_s1[k,goodbins[k]], outlier, clonal_mark_type, numbins)
   if (clonal_mark_type[2]=='L'): # lumen, use inner signal
      for k in range(numcnts):
         clone_signal_width[k], clone_signal_total[k] = signal_width_ndo(in_bins_c_s1[k,goodbins[k]], outlier, clonal_mark_type, numbins)
     
   # pull out obvious clones above a hard threshold, remove from further knn analysis
   inds_clone_w = np.where(clone_signal_width>=0.45)[0]
   inds_clone_t = np.where(clone_signal_total>=0.5)[0]
   inds_clone = np.unique(np.hstack([inds_clone_w, inds_clone_t]))
   global_clone_inds = goodsizeinds[inds_clone]
   globscores[global_clone_inds] = clone_signal_total[inds_clone]
   goodsizeinds2 = np.setdiff1d(goodsizeinds, global_clone_inds)
   
   # subset further
   halo_n_s2           = cfl['halo_n'][goodsizeinds2]
   halo_c_s2           = cfl['halo_c'][goodsizeinds2]
   out_bins_n_s2       = cfl['out_bins_n'][goodsizeinds2,:]
   out_bins_c_s2       = cfl['out_bins_c'][goodsizeinds2,:]
   xy_coords_s2        = cfl['xy_coords'][goodsizeinds2,:]
   in_bins_n_s2        = cfl['in_bins_n'][goodsizeinds2,:]
   in_bins_c_s2        = cfl['in_bins_c'][goodsizeinds2,:]
   content_n_s2        = cfl['content_n'][goodsizeinds2]
   content_c_s2        = cfl['content_c'][goodsizeinds2]
   globinds_s2         = globinds[goodsizeinds2]
   # and goodbins!
   goodbinsinds = []
   for ii in goodsizeinds2:
      goodbinsinds.append(np.where(goodsizeinds==ii)[0][0])
   goodbins_s2 = list(np.asarray(goodbins)[goodbinsinds])
   
   ## Use clone content as extra check on clone status for (negative) non-nuclear stains
   ###########################################   
   #nonclone_above_content = find_outlier_truncated_normal(content_c, numSD = 1.25)
   #inds2 = np.where(content_c < nonclone_above_content)[0]
   #inds_clone = np.intersect1d(inds1,inds2)
   
   # convert global indices from R to subset indices for testing
#   Sinds = np.array([])
#   for i in Rinds:
#      Sinds = np.hstack([Sinds, np.where(globinds_s2==i)[0]])
#   Sinds = Sinds.astype(np.intp)
   
   ## Now use local knn comparisons to find less obvious clones and partials
   ###########################################
   # fix the below for all clonal marks!
   nn = 50
   nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(xy_coords_s2)
   distances, indices = nbrs.kneighbors(xy_coords_s2)
#   local_halo_c_zscores    = np.zeros(indices.shape[0])
#   local_content_c_zscores = np.zeros(indices.shape[0])
   local_bin_zscores       = np.zeros([indices.shape[0], numbins])
   num_outlier_bins        = np.zeros(indices.shape[0])

   if (clonal_mark_type[2]=='N' or clonal_mark_type[2]=='B'): # nuclear or both, use outer signal
      bins_c = out_bins_c_s2
   if (clonal_mark_type[2]=='L'): # lumen, use inner signal
      bins_c = in_bins_c_s2
   
   for i in range(0, indices.shape[0]):
      # find clone bin signal outlier from nbrs
      me = indices[i,0]
      my_nns = indices[i,1:]
      # Z-scores for crypt within knn population
      knn_bins = np.array([])
      for j in my_nns:
         knn_bins = np.hstack([knn_bins, bins_c[j, goodbins_s2[j]]])
      meanbins = np.mean(knn_bins)
      sdbins = np.std(knn_bins)
      for j in goodbins_s2[me]:
         local_bin_zscores[me, j] = (bins_c[me, j] - meanbins) / sdbins
      if (clonal_mark_type[0]=='N'): # negative, use outlier below lower bound
         num_outlier_bins[me] = np.where(local_bin_zscores[me,:] < -3.)[0].shape[0]
      if (clonal_mark_type[0]=='P'): # positive, use outlier above upper bound
         num_outlier_bins[me] = np.where(local_bin_zscores[me,:] > 3.)[0].shape[0]
         
      #local_content_c_zscores[me] = (content_c_s2[me] - np.mean(content_c_s2[my_nns])) / np.std(content_c_s2[my_nns])
      #local_halo_c_zscores[me]    = (halo_c_s2[me] - np.mean(halo_c_s2[my_nns])) / np.std(halo_c_s2[my_nns])

   new_inds_clone = np.where(num_outlier_bins>=2)[0]
   newglobinds = goodsizeinds2[new_inds_clone]
   globscores[newglobinds] = num_outlier_bins[new_inds_clone]/20.
   
   # checking for missed closed
   allneg = []
   for i in range(local_bin_zscores.shape[0]):
      if np.all(local_bin_zscores[i,:]<0):
         if (np.where(local_bin_zscores[i,:]<-2)[0].shape[0] > 0):
            allneg.append(i)
   missedinds = np.asarray(allneg)
   globmissedinds = goodsizeinds2[missedinds]
   globscores[globmissedinds] = 1./20. # set as base low score
   
   new_inds_clone = np.unique(np.hstack([new_inds_clone, missedinds]))     
   local_clone_inds = goodsizeinds2[new_inds_clone]
   all_clone_inds = np.hstack([global_clone_inds, local_clone_inds])
   return all_clone_inds, globscores[all_clone_inds]
              
def no_threshold_signal_collating(cnt_i, img_nuc, img_clone, nbins):
   # Find max halo contour in nuclear channel, and nucl/clone halo scores
   nucl_halo, clone_halo, output_cnt = max_halocnt_nc(cnt_i, img_nuc, img_clone)
   # Find clone/nucl content
   content_n, content_c = get_contents(cnt_i, img_nuc, img_clone)
   # Bin length of contour into ~20 bins; average pixel intensity in each bin in both nuclear and clonal channel
   av_sig_nucl = bin_intensities_flattened(output_cnt, img_nuc, nbins)
   av_sig_clone = bin_intensities_flattened(output_cnt, img_clone, nbins)
   # Save along with (x,y) coordinate of centre of crypt   
   M = cv2.moments(output_cnt)
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   centre_xy = (cX, cY)
   # Repeat for a contour that is slightly eroded to look for inside signal
   inner_cnt = extractInnerRingContour(cnt_i, img_nuc, 1)
   inav_sig_nucl = bin_intensities_flattened(inner_cnt, img_nuc, nbins)
   inav_sig_clone = bin_intensities_flattened(inner_cnt, img_clone, nbins)
   return nucl_halo, clone_halo, av_sig_nucl, av_sig_clone, centre_xy, inav_sig_nucl, inav_sig_clone, content_n, content_c#, output_cnt
   
def outlier_vec_calculator(av_sig_mat, numIQR = 1.25):
   big_outlier = np.zeros(av_sig_mat.shape[1])
   small_outlier = np.zeros(av_sig_mat.shape[1])
   av_small_out = np.ones(av_sig_mat.shape[1])
   av_big_out = np.ones(av_sig_mat.shape[1])
   for j in range(av_sig_mat.shape[1]):
      small_outlier[j] = np.percentile(av_sig_mat[:, j], 25)
      big_outlier[j] = tukey_upper_thresholdval(av_sig_mat[:, j], 75)
   av_small_out = av_small_out * (np.mean(small_outlier))
   av_big_out = av_big_out * (np.mean(big_outlier))
   return av_small_out , av_big_out



