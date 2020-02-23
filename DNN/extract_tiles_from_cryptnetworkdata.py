
import glob
import cv2
import numpy      as np
import openslide  as osl
import pandas     as pd
from MiscFunctions import getROI_img_osl, read_cnt_text_file, rescale_contours, mkdir_p,\
                          load_all_contours2, plot_img
from DNN_segment import get_tile_indices

ONLY_CLONES = True
tilesize = 512
overlap = 50
outfolder = "/home/doran/Work/py_code/DeCryptICS/DNN/input/hires"
img_outdir = outfolder + '/img/'
mask_outdir = outfolder + '/mask/'
mkdir_p(img_outdir)
mkdir_p(mask_outdir)

basefolder = "/home/doran/Work/images/Blocks/"
folders_meta = glob.glob(basefolder + "block_*/")
#basefolder = "/home/doran/Work/images/Leeds_May2019/splitbyKM/"
#folders_meta = glob.glob("/home/doran/Work/images/Leeds_May2019/curated_cryptdata/KM*/")
for ff in folders_meta:
   set_num = ff.split('/')[-2]
   folders_slides = glob.glob(basefolder + set_num + '/Analysed_slides/Analysed_*/')
   slide_data = pd.read_csv(basefolder + set_num + '/slide_info.csv')
   slide_data = slide_data.astype({'Image ID':'str'})
   for fsl in folders_slides:
      # get curated crypt net data, find image and get slide
      netdat = np.loadtxt(fsl + "crypt_network_data.txt")
      imname = fsl.split('/')[-2].split('Analysed_')[-1] + '.svs'
      imloc = basefolder + ff.split('/')[-2] + '/'
      impath = imloc + imname
      slide = osl.OpenSlide(impath)
      
      # define mark name
#      mark_letter = imname.split('_')[0][-1] # unique to the "KM**M_*****" file name 
#      if mark_letter=='S': mark = "STAG2"
#      if mark_letter=='M': mark = "MAOA"
#      if mark_letter=='K': mark = "KDM6A"      
      ind = np.where(slide_data['Image ID']==str(imname[:-4]))[0]
      try:
         mark = slide_data['mark'][ind[0]]
      except:
         print(".svs missing, skipping!")
      
      # find required downsample level
      mpp = float(slide.properties['openslide.mpp-x'])
      mpp_fin = 0.5035 # 2.0144 # ~ desired microns per pixel
      errlevel = 0.15 # 0.5 allowable distance from mpp_fin
      dsls = slide.level_downsamples
      mpp_test = []
      for lvl in dsls:
         mpp_test.append(lvl*mpp)
      mpp_test = np.asarray(mpp_test)
      choice = np.where(abs(mpp_test-mpp_fin)<errlevel)[0]
      # check if we need to manually downsample
      if (choice.shape[0]==1):
         print("Running with downsample stream 0")
         STREAM = 0
         dwnsmpl_lvl = choice[0]
      if (choice.shape[0]==0):
         print("Running with downsample stream 1")
         STREAM = 1
         # get level from which we can downsample to desired amount   
         dwnsmpl_lvl = np.where(abs(mpp_test-mpp_fin/2.) < errlevel)[0][0]

      # find scale
      scale = slide.level_downsamples[dwnsmpl_lvl]      
      if STREAM==1: scale *= 2

      # try to load contours
      try:
         cnts  = load_all_contours2(impath, scale)[0]
      except:
         print("Missing contour file, skipping.")
         continue

      # check if contour length consistent: may have been run again
      if not (len(cnts)==netdat.shape[0]):
         print("Contour length / network data shape mismatch, skipping.")
         continue
      
      imshape_xy = slide.level_dimensions[dwnsmpl_lvl]
#      # pre-load image
#      img_full = getROI_img_osl(impath, (0,0), slide.level_dimensions[dwnsmpl_lvl] , dwnsmpl_lvl)
#      if STREAM==1:
#         img_full = cv2.pyrDown(img_full)

      # draw contours to make mask
      clone_inds = np.where(netdat[:,3]>0)[0]
      big_mask = np.zeros([imshape_xy[1], imshape_xy[0]], dtype=np.uint8)
      for i in range(clone_inds.shape[0]):
         cv2.drawContours(big_mask, [cnts[clone_inds[i]]], 0, 255, -1)

      # define tiles
      all_indx = get_tile_indices(big_mask.shape[1::-1], overlap = overlap, SIZE = (tilsesize, tilsesize))
      x_tiles = len(all_indx)
      y_tiles = len(all_indx[0])
      
      # make output folder for this slide
      this_img_out = img_outdir + 'img_set_' + ff.split('/')[-2] + '_slide_' + imname[:-4]
      this_mask_out = mask_outdir + 'mask_set_' + ff.split('/')[-2] + '_slide_' + imname[:-4]

      # save tiles
      for i in range(x_tiles):
         for j in range(y_tiles):
            xy_vals = (int(all_indx[i][j][0]), int(all_indx[i][j][1]))
            wh_vals = (int(all_indx[i][j][2]), int(all_indx[i][j][3]))
            mask = big_mask[xy_vals[1]:(xy_vals[1]+wh_vals[1]), xy_vals[0]:(xy_vals[0]+wh_vals[0])]
            if cv2.countNonZero(mask)>25:
               img = getROI_img_osl(impath, (xy_vals[0],xy_vals[1]), (tilesize, tilesize) , dwnsmpl_lvl)
               outfile_img = this_img_out + '_x' + str(i) + '_y' + \
                                 str(j) + '_' + mark + '_T_clone.png'
               outfile_mask = this_mask_out + '_x' + str(i) + '_y' + \
                                 str(j) + '_' + mark + '_T_clone.png'
               cv2.imwrite(outfile_img, img)    
               cv2.imwrite(outfile_mask, mask)               

         
                  

            img = img_full[xy_vals[1]:(xy_vals[1]+wh_vals[1]), xy_vals[0]:(xy_vals[0]+wh_vals[0]), :]
            mask = big_mask[xy_vals[1]:(xy_vals[1]+wh_vals[1]), xy_vals[0]:(xy_vals[0]+wh_vals[0])]
            
            if cv2.countNonZero(mask)<300:
               if not ONLY_CLONES:
                  outfile_img = this_img_out + '_x' + str(i) + '_y' + \
                                    str(j) + '_' + mark + '_F_clone.png'
                  outfile_mask = this_mask_out + '_x' + str(i) + '_y' + \
                                    str(j) + '_' + mark + '_F_clone.png'
               else: continue
            else:
               outfile_img = this_img_out + '_x' + str(i) + '_y' + \
                                 str(j) + '_' + mark + '_T_clone.png'
               outfile_mask = this_mask_out + '_x' + str(i) + '_y' + \
                                 str(j) + '_' + mark + '_T_clone.png'
            cv2.imwrite(outfile_img, img)    
            cv2.imwrite(outfile_mask, mask)



