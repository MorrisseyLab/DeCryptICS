#!/usr/bin/env python3
## process clones, make patches etc
import glob, os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from MiscFunctions import write_score_text_file, process_input_file
from just_read_svs import svs_file

def main():
   parser = argparse.ArgumentParser(description = 'This script takes as input a list of full paths (local or remote) to .svs files, '
                                                   'or a larger dataframe that contains such a list as a column. '
                                                   'The output is a processed dataframe of binary clone, partial and fufi designations for each crypt basd on probability thresholds. '
                                                   'Contour masks are created to load these outputs into QuPath as Detection Objects. ' )
    
   parser.add_argument('input_file', help = 'A file containing a list of full paths to slides (or array with column acting as said list). ')

   parser.add_argument('-r', action = 'store_true', 
                             default = False,
                             help = 'Forces repeat analysis of slides with existing processed output. '
                                    'Defaults to False if -r flag missing. ')
                                    
   parser.add_argument('-m', action = 'store_true', 
                             default = False,
                             help = 'Use manual threshold designation for each slide, with visualised output. '
                                    'Defaults to False if -m flag missing. ')

   parser.add_argument('-c',  action = 'store',
                              dest = 'p_c', 
                              default = 0.5, 
                              help = 'Mutant probability threshold. ')

   parser.add_argument('-p',  action = 'store',
                              dest = 'p_p', 
                              default = 0.5, 
                              help = 'Partial probability threshold. ')

   parser.add_argument('-f',  action = 'store',
                              dest = 'p_f', 
                              default = 0.5, 
                              help = 'FuFi probability threshold. ')
                              
   parser.add_argument('-g',  action = 'store',
                              dest = 'p_g', 
                              default = 0.01, 
                              help = 'Gland (crypt) probability threshold. ')
                                 
   args = parser.parse_args()
   ## check args
   print('Running with the following inputs:')
   print('input_file   = {!r}'.format(args.input_file))
   print('force_repeat = {!r}'.format(args.r))
   print('use_manual_thresholding = {!r}'.format(args.m))
   print('p_c = {!r}'.format(args.p_c))
   print('p_p = {!r}'.format(args.p_p))
   print('p_f = {!r}'.format(args.p_f))
   print('p_g = {!r}'.format(args.p_g))
   p_c = float(args.p_c)
   p_p = float(args.p_p)
   p_f = float(args.p_f)
   p_g = float(args.p_g)
   use_manual_thresholding = bool(args.m)
   r = bool(args.r)

   ## Find output folder
   base_path, folder_out, file_in, folders_to_analyse, full_paths = process_input_file(args.input_file)

   ## run processing
   for i in range(len(file_in)):
      if (os.path.isfile(folders_to_analyse[i]+'/processed_output.csv') and r==False):
         print('Passing on %s, output previously processed.' % full_paths[i])
         pass
      else:
         print('Processing %s.' % full_paths[i])
         try:
            process_events(folders_to_analyse[i], use_manual_thresholding, p_c, p_p, p_f, p_g)
         except:
            print('Passing due to error.')
         
   ## extract crypt counts from all analysed slides into base path
   extract_counts_csv(file_in, folder_out, base_path)

def process_events(output_folder, use_manual_thresholding=True, clone_thresh=0.5, partial_thresh=0.5, fufi_thresh=0.5, crypt_thresh=0.5):
   if use_manual_thresholding:
      file_name = output_folder.split('/Analysed_slides/')[0] + output_folder.split('/Analysed_')[-1][:-1] + '.svs'
      tilesize = 60
      um_per_pixel = 1.25
      svs = svs_file(file_name, tilesize, um_per_pixel)
      svs.load_events(output_folder)
      svs.initialize_event_tile_params(wh = 80, bfs = 2)
      # do clones      
      svs.set_event_type('p_clone')
      svs.plot_sampled_events()
      # do partials
      svs.set_event_type('p_partial')   
      svs.plot_sampled_events()
      # do fufis
      svs.set_event_type('p_fufi')
      svs.plot_sampled_events()
      # do crypts if present
      if 'p_crypt' in svs.raw_data.columns:
         svs.set_event_type('p_crypt')
         svs.plot_sampled_events()
      # create masks
      out_df = svs.raw_data
      clone_mask   = np.asarray(out_df['p_clone']   >= svs.clone_threshold).astype(np.int32)
      partial_mask = np.asarray(out_df['p_partial'] >= svs.partial_threshold).astype(np.int32)
      fufi_mask    = np.asarray(out_df['p_fufi']    >= svs.fufi_threshold).astype(np.int32)
      if 'p_crypt' in svs.raw_data.columns:
         crypt_mask = np.asarray(out_df['p_crypt'] >= svs.crypt_threshold).astype(np.int32)
   else:
      # load raw output
      out_df = pd.read_csv(output_folder + '/raw_crypt_output.csv')
      # get masks that can be used for loading contours in qupath, output these
      clone_mask   = np.asarray(out_df['p_clone']   >= clone_thresh).astype(np.int32)
      partial_mask = np.asarray(out_df['p_partial'] >= partial_thresh).astype(np.int32)
      fufi_mask    = np.asarray(out_df['p_fufi']    >= fufi_thresh).astype(np.int32)
      if 'p_crypt' in out_df.columns:
         crypt_mask = np.asarray(out_df['p_crypt'] >= crypt_thresh).astype(np.int32)
      
   clone_inds   = np.where(clone_mask==1)[0]
   # binarise output columns    
   out_df.loc[:,'p_clone']   = list(clone_mask)
   out_df.loc[:,'p_partial'] = list(partial_mask)
   out_df.loc[:,'p_fufi']    = list(fufi_mask)
   if 'p_crypt' in out_df.columns:
      out_df.loc[:,'p_crypt'] = list(crypt_mask)
      
   # Join patches as contours
   patch_indices = join_clones_if_close(out_df, clone_inds)
   out_df = get_crypt_patchsizes_and_ids(patch_indices, out_df)
   
   # save output, but remove .bacs first
   if os.path.exists(output_folder + '/clone_mask.bac'): os.remove(output_folder + '/clone_mask.bac')
   if os.path.exists(output_folder + '/partial_mask.bac'): os.remove(output_folder + '/partial_mask.bac')
   if os.path.exists(output_folder + '/fufi_mask.bac'): os.remove(output_folder + '/fufi_mask.bac')
   if os.path.exists(output_folder + '/crypt_mask.bac'): os.remove(output_folder + '/crypt_mask.bac')
   if os.path.exists(output_folder + '/processed_output.bac'): os.remove(output_folder + '/processed_output.bac')
   
   out_df.to_csv(output_folder + '/processed_output.csv', index=False)   
   write_score_text_file(clone_mask  , output_folder + '/clone_mask.txt')
   write_score_text_file(partial_mask, output_folder + '/partial_mask.txt')
   write_score_text_file(fufi_mask   , output_folder + '/fufi_mask.txt')
   if 'p_crypt' in out_df.columns:
      write_score_text_file(crypt_mask, output_folder + '/crypt_mask.txt')
   
def get_crypt_patchsizes_and_ids(patch_indices, out_df):
   # gives mutant patches of size > 1 a unique ID.  Single mutants have ID = 0
   patch_size = np.zeros(out_df.shape[0])
   patch_id   = np.zeros(out_df.shape[0])
   for i in range(1, len(patch_indices)+1):
      patch = patch_indices[i-1]
      for index in patch:
         patch_size[index] = len(patch)
         patch_id[index] = i
   out_df['patch_size'] = list(patch_size)
   out_df['patch_id'] = list(patch_id)
   return out_df
   
def add_nearby_clones(patch, indices, clone_inds, i, j):
   patch.append(i)
   for k in indices[i, 1:]:
      if k in clone_inds:
         if k not in patch:
            add_nearby_clones(patch, indices, clone_inds, k, j)
   return patch

def join_clones_if_close(out_df, clone_inds):
   nn = np.minimum(9, out_df.shape[0]-1)
   nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(out_df[['x','y']])
   distances, indices = nbrs.kneighbors(out_df[['x','y']])
   patches = []
   j = 0
   for i in clone_inds:
      patches.append([])
      patches[j] = add_nearby_clones(patches[j], indices, clone_inds, i, j)
      j += 1

   # remove length 1 patches and repeated indices
   cut_patches = []
   addflag = True
   for ll in patches:
      if len(ll)>1:
         addflag = True
         newpatch = set(ll)
         for pp in cut_patches:
            if (newpatch==pp):
               addflag = False
         if addflag == True:
            cut_patches.append(newpatch)

   # join any repeated subsets
   cut_patches2 = []
   used_patches = []
   for pp in range(len(cut_patches)):
      thispatch = cut_patches[pp]
      if np.any([thispatch.issubset(aset) for aset in cut_patches2]):
         # this patch is already part of a bigger patch
         continue
      used_patches.append(thispatch)
      # get quicker version of power set and check for all matching subsets
      allsubsets = list(thispatch)
      good_subsets = []
      for sbset in allsubsets:
         subset_bool = [set([sbset]).issubset(aset) for aset in cut_patches]
         good_subsets += list(np.where(subset_bool)[0])
      good_subsets = list(set(good_subsets))
      for jj in good_subsets:
         if cut_patches[jj] not in used_patches:
            thispatch |= cut_patches[jj]
      cut_patches2.append(thispatch)
      
   # turn into lists
   newpatchinds = []
   for patch in cut_patches2:
      patch = list(patch)
      newpatchinds.append(patch)
   return newpatchinds

def folder_from_image(image_num_str):
    return "/Analysed_"+str(image_num_str)+'/'

def extract_counts_csv(file_in, folder_out, base_path):
   contour_folders = [folder_from_image(im) for im in file_in]
   num = len(contour_folders)
   slide_counts = np.zeros([num, 8], dtype=object)
   slide_names = list(file_in)
   cols = ['Slide_ID', 'NCrypts', 'NFufis', 'NMutantCrypts', 'NClones', 'NMonoclonals', 'NPartials', 'NPatches', 'PatchSizes']
   for i in range(num):
      try:
         df = pd.read_csv(base_path+'/Analysed_slides/'+contour_folders[i]+"/processed_output.csv")
      except:
         print('File missing: %s' % (base_path+'/Analysed_slides/'+contour_folders[i]+"/processed_output.csv"))
         continue
      if 'p_crypt' in df.columns:
         # filter out non-crypts
         df = df[df['p_crypt']>0].reset_index().drop(['index'], axis=1)
      allclone_inds = np.where(np.asarray(df['p_clone'])==1)[0]
      partial_inds = np.where(np.asarray(df['p_partial'])==1)[0]
      monoclonal_inds = np.setdiff1d(allclone_inds, partial_inds)
      allmuts = np.hstack([partial_inds, monoclonal_inds])
      
      cryptcount = df.shape[0]
      fuficount = np.where(np.asarray(df['p_fufi'])==1)[0].shape[0]
      mutantcryptcount = allmuts.shape[0]
      monoclonalcount = monoclonal_inds.shape[0]
      partialcount = partial_inds.shape[0]
      
      # deal with patches
      try:
         unique_patches = np.vstack(list({tuple(row) for row in np.asarray(df[['patch_size', 'patch_id']].iloc[allmuts])})).astype(np.int32) # size, id
      except:
         unique_patches = np.zeros((1,2), dtype=np.int32)
      numpatches = np.max(unique_patches[:,1])
      patchsum = np.sum(unique_patches, axis=0)[0]
      patch_sizes = np.sort(unique_patches[unique_patches[:,0]>0, 0])
      patchsize_str = str(list(patch_sizes)).replace('[','').replace(']','').replace(' ', '')

      slide_counts[i,0] = cryptcount # 'NCrypts'
      slide_counts[i,1] = fuficount # 'NFufis',
      slide_counts[i,2] = mutantcryptcount # 'NMutantCrypts'
      slide_counts[i,3] = int(mutantcryptcount - patchsum + numpatches) # 'NClones'
      slide_counts[i,4] = monoclonalcount # 'NMonoclonals'
      slide_counts[i,5] = partialcount # 'NPartials'
      slide_counts[i,6] = numpatches # 'NPatches'
      slide_counts[i,7] = patchsize_str # 'PatchSizes'
   slidecounts_p = pd.DataFrame(slide_counts, columns=cols[1:])
   slidecounts_p = pd.concat([pd.DataFrame({'Slide_ID':slide_names}), slidecounts_p], axis=1)
   outname = "/slide_counts.csv"
   slidecounts_p.to_csv(folder_out + outname, sep=',', index=False)

if __name__=='__main__':
   main()
