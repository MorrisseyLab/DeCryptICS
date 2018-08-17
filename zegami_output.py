# -*- coding: utf-8 -*-
"""
Created on Thurs July 26 11:24:02 2018

@author: doran
"""

## Outputting data for Zugami

import cv2, os
from MiscFunctions import getROI_img_vips, plot_img, read_cnt_text_file
from GUI_ChooseROI_class      import getROI_svs

def zegami_output(crypt_contours, cfl, signal_width, local_scores, file_name, folder_to_analyse, save_images=False):
   ## Create output file structure
   zegami_folder = folder_to_analyse + "/zegami/"
   img_filenames = zegami_folder + "img_"
   metadata_filename = zegami_folder + "metadata.txt"
   try:
      os.mkdir(zegami_folder)
   except:
      pass   

   ## Choose subset of data (1000 data points here)
   numsubset = cfl['halo_n'].shape[0] #5000
   numdat = len(crypt_contours)
   if (numdat<=numsubset): inds = list(range(0,numdat))
   else:
      factor = int(floor(float(numdat)/float(numsubset)))
      inds = list(range(0, factor*numsubset, factor))
   #inds = list(range(0,numsubset)) # just take first numsubset
   
#   ## Create binary clone label
#   clone_poslabels = clone_inds[np.where(clone_inds<numsubset)[0]]
#   clone_labels = np.zeros(numsubset)
#   clone_labels[clone_poslabels] = 1
      
   ## Output the ROI of the subset of crypts
   if (save_images == True):
      obj_svs  = getROI_svs(file_name , get_roi_plot = False)
      i = 1
      for ii in inds:
         cnt_i = crypt_contours[ii]
         expand_box    = 35
         roi           = cv2.boundingRect(cnt_i)
         roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
         roi[roi<1]   = 0
         img_ROI       = getROI_img_vips(file_name, (roi[0],roi[1]), (roi[2],roi[3]))
         outfile = img_filenames + str(i) + ".png"
         cv2.imwrite(outfile, img_ROI)
         i += 1

   ## Form metadata into initial column with number of each crypt (1:n) and columns of data with headers
   halo_n           = cfl['halo_n']
   halo_c           = cfl['halo_c']
   xy_coords        = cfl['xy_coords']
   content_n        = cfl['content_n']
   content_c        = cfl['content_c']
   content_c_zscores = local_scores['content_c_zscore']
   halo_c_zscores = local_scores['halo_c_zscore']
   num_outlier_bins = local_scores['num_outlier_bins']
   local_signal_width = local_scores['local_signal_width']
   local_signal_total = local_scores['local_signal_total']
   size = np.zeros(halo_n.shape[0])
   for ii in inds:
      size[ii] = contour_Area(crypt_contours[ii])
   
   with open(metadata_filename, 'w') as fo:
      fo.write("ID\thalo_n\thalo_c\tcontent_n\tcontent_c\tx\ty\tsignal_width\tsize\tcontent_c_zscores_glob\thalo_c_zscores\tnum_outlier_bins\tlocal_signal_width\tlocal_signal_total\n")
      i = 1
      for ii in inds:
         outfile = "img_" + str(i)
         fo.write("%s\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\n" % (outfile, halo_n[ii], halo_c[ii], content_n[ii], content_c[ii], xy_coords[ii,0], xy_coords[ii,1], signal_width[ii], size[ii], content_c_zscores[ii], halo_c_zscores[ii], num_outlier_bins[ii], local_signal_width[ii], local_signal_total[ii]))
         i += 1
         
      
  

