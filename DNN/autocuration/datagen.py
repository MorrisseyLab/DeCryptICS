import glob
import cv2
import os
import numpy      as np
import openslide  as osl
import pandas     as pd
from joblib import Parallel, delayed
import multiprocessing
from sklearn.neighbors import NearestNeighbors
from MiscFunctions import getROI_img_osl, read_cnt_text_file, rescale_contours, mkdir_p,\
                          load_all_contours2, plot_img, load_single_contour,\
                          rotate_contour_about_centre, add_offset

datapath = "./DNN/autocuration/data/"

def centred_tile(XY, tilesize, max_xy, edge_adjust = True):
   if type(tilesize)!=list and \
         type(tilesize)!=np.ndarray and \
            type(tilesize)!=tuple:
      tilesize = [tilesize,tilesize]
   x0y = XY[0] - tilesize[0]/2.
   xy0 = XY[1] - tilesize[1]/2.
   if edge_adjust:
      if (x0y < 0):
         x0y = 0
      if (xy0 < 0):
         xy0 = 0
      xNy = x0y + tilesize[0]
      if (xNy >= max_xy[0]):
         x0y = x0y - (xNy - max_xy[0] + 1)
         xNy = x0y + tilesize[0]
      xyN = xy0 + tilesize[1]
      if (xyN >= max_xy[1]):
         xy0 = xy0 - (xyN - max_xy[1] + 1)
         xyN = xy0 + tilesize[1]
   return np.array([x0y, xy0])

def pad_edge_image(img_m_out, SIZE, xy, full_img_shape):
   SIZE = int(SIZE)
   if (img_m_out.shape[1]<2*SIZE):
      pad_dir = np.array([full_img_shape[1]-xy[0], xy[0]])
      pad = np.zeros((img_m_out.shape[0], 2*SIZE-img_m_out.shape[1], 3), dtype=np.uint8)
      if (np.argmin(pad_dir)==0):
         img_m_out = np.hstack([img_m_out, pad])
      if (np.argmin(pad_dir)==1):
         img_m_out = np.hstack([pad, img_m_out])
   if (img_m_out.shape[0]<2*SIZE):
      pad_dir = np.array([full_img_shape[0]-xy[1], xy[1]])
      pad = np.zeros((2*SIZE-img_m_out.shape[0], img_m_out.shape[1], 3), dtype=np.uint8)
      if (np.argmin(pad_dir)==0):       
         img_m_out = np.vstack([img_m_out, pad])
      if (np.argmin(pad_dir)==1):
         img_m_out = np.vstack([pad, img_m_out])
   return img_m_out

def pad_edge_image_keep_tl_padding(img_m_out, SIZE, xy, full_img_shape):
   pad_l = 0
   pad_t = 0
   SIZE = int(SIZE)
   if (img_m_out.shape[1]<2*SIZE):
      pad_dir = np.array([full_img_shape[1]-xy[0], xy[0]])
      pad = np.zeros((img_m_out.shape[0], 2*SIZE-img_m_out.shape[1], 3), dtype=np.uint8)
      if (np.argmin(pad_dir)==0):
         img_m_out = np.hstack([img_m_out, pad])
      if (np.argmin(pad_dir)==1):
         img_m_out = np.hstack([pad, img_m_out])
         pad_l = pad.shape[1]
   if (img_m_out.shape[0]<2*SIZE):
      pad_dir = np.array([full_img_shape[0]-xy[1], xy[1]])
      pad = np.zeros((2*SIZE-img_m_out.shape[0], img_m_out.shape[1], 3), dtype=np.uint8)
      if (np.argmin(pad_dir)==0):
         img_m_out = np.vstack([img_m_out, pad])         
      if (np.argmin(pad_dir)==1):
         img_m_out = np.vstack([pad, img_m_out])
         pad_t = pad.shape[0]
   return img_m_out, pad_l, pad_t

def pad_image(img_hi_bb, DNN_size):
   if (img_hi_bb.shape[0]>=img_hi_bb.shape[1]):
      img_hi_bb = image_resize(img_hi_bb, height = DNN_size)
      if (img_hi_bb.shape[1]!=DNN_size):
         pad = np.zeros((DNN_size, (DNN_size-img_hi_bb.shape[1])//2, 3), dtype=np.uint8)
         img_hi_bb = np.hstack([pad, img_hi_bb, pad])
         img_hi_bb = cv2.resize(img_hi_bb, (DNN_size, DNN_size), interpolation = cv2.INTER_AREA)
   else:
      img_hi_bb = image_resize(img_hi_bb, width = DNN_size)
      if (img_hi_bb.shape[0]!=DNN_size):
         pad = np.zeros(((DNN_size-img_hi_bb.shape[0])//2, DNN_size, 3), dtype=np.uint8)
         img_hi_bb = np.vstack([pad, img_hi_bb, pad])
         img_hi_bb = cv2.resize(img_hi_bb, (DNN_size, DNN_size), interpolation = cv2.INTER_AREA)
   return img_hi_bb
   
def pad_image_keeppadding(img_hi_bb, DNN_size):
   pad_lr = 0
   pad_tb = 0
   if (img_hi_bb.shape[0]>=img_hi_bb.shape[1]):
      img_hi_bb, r = image_resize_keepscaling(img_hi_bb, height = DNN_size)
      if (img_hi_bb.shape[1]!=DNN_size):
         pad = np.zeros((DNN_size, (DNN_size-img_hi_bb.shape[1])//2, 3), dtype=np.uint8)
         img_hi_bb = np.hstack([pad, img_hi_bb, pad])
         img_hi_bb = cv2.resize(img_hi_bb, (DNN_size, DNN_size), interpolation = cv2.INTER_AREA)
         pad_lr = pad.shape[1]         
   else:
      img_hi_bb, r = image_resize_keepscaling(img_hi_bb, width = DNN_size)
      if (img_hi_bb.shape[0]!=DNN_size):
         pad = np.zeros(((DNN_size-img_hi_bb.shape[0])//2, DNN_size, 3), dtype=np.uint8)
         img_hi_bb = np.vstack([pad, img_hi_bb, pad])
         img_hi_bb = cv2.resize(img_hi_bb, (DNN_size, DNN_size), interpolation = cv2.INTER_AREA)
         pad_lr = pad.shape[0]         
   return img_hi_bb, r, pad_lr, pad_tb

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized
    
def image_resize_keepscaling(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image, 1.
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized, r

def pull_centered_img(XY, imgpath, tilesize, ROTATE=False, RT_m=0, dwnsample_lvl=1):
   # get slide maximum dims
   slide = osl.OpenSlide(imgpath)
   maxdims = slide.level_dimensions[dwnsample_lvl]
   # Fix tiles to specific DNN size
   xy_m = XY/slide.level_downsamples[dwnsample_lvl]
   if ROTATE==False:
      xy_vals_m_ds_out = centred_tile(xy_m, tilesize, maxdims, edge_adjust = False)
      # pull out images
      Lx = np.maximum(0,int(np.around(xy_vals_m_ds_out[0])))
      Rx = int(np.around((xy_vals_m_ds_out[0]+tilesize)))
      Ty = np.maximum(0,int(np.around(xy_vals_m_ds_out[1])))
      By = int(np.around((xy_vals_m_ds_out[1]+tilesize)))
      img = getROI_img_osl(imgpath, (Lx,Ty), (tilesize, tilesize), dwnsample_lvl)
      # pad if required
      img = pad_edge_image(img.copy(), tilesize//2, xy_m, maxdims)
      return img
   else:
      xy_vals_m_ds_out = centred_tile(xy_m, 2*tilesize, maxdims, edge_adjust = False)
      # pull out images
      Lx = np.maximum(0,int(np.around(xy_vals_m_ds_out[0])))
      Rx = int(np.around((xy_vals_m_ds_out[0]+2*tilesize)))
      Ty = np.maximum(0,int(np.around(xy_vals_m_ds_out[1])))
      By = int(np.around((xy_vals_m_ds_out[1]+2*tilesize)))
      img = getROI_img_osl(imgpath, (Lx,Ty), (2*tilesize, 2*tilesize), dwnsample_lvl)
      # pad if required
      img = pad_edge_image(img.copy(), tilesize, xy_m, maxdims)
      img = cv2.warpAffine(img.copy(), RT_m[:2,:], img.shape[1::-1])
      img = img[(tilesize//2):-(tilesize//2), (tilesize//2):-(tilesize//2), :]
      return img

def pull_crypt(XY, imgpath, cntpath, ind_m, size_small, 
               ROTATE=False, RT_m=0, dwnsample_lvl=0):
   # get slide maximum dims
   slide = osl.OpenSlide(imgpath)
   maxdims = slide.level_dimensions[dwnsample_lvl]
   scale = slide.level_downsamples[dwnsample_lvl]
   # get contour for crypt img
   cnt = load_single_contour(cntpath, cnt_id = ind_m)
   # create initial bounding box to get size_big requirements
   lt = int(-60./scale)
   rb = int(120./scale)
   bb_pad = np.array([lt, lt, rb, rb], dtype=np.int32)
   bb_m = np.asarray(cv2.boundingRect(cnt[0])) + bb_pad
   bb_m[bb_m<0] = 0
   size_big = int(1.5*np.max(bb_m[-2:]))
   # create centred tile
   xy_vals_m_ds_in2 = centred_tile(XY, size_big, maxdims, edge_adjust = False)
   cnt_sub = add_offset(cnt.copy(), -xy_vals_m_ds_in2)
   Lx = np.maximum(0,int(np.around(xy_vals_m_ds_in2[0])))
   Ty = np.maximum(0,int(np.around(xy_vals_m_ds_in2[1])))
   # pull out image and rotate if required
   img = getROI_img_osl(imgpath, (Lx,Ty), (size_big, size_big), dwnsample_lvl)
   if ROTATE==True:
      img = cv2.warpAffine(img.copy(), RT_m[:2,:], img.shape[1::-1])
      cnt_sub = [rotate_contour_about_centre(cnt_sub[0], RT_m)]
   # create padded bounded box
   bb_m = np.asarray(cv2.boundingRect(cnt_sub[0])) + bb_pad
   bb_m[bb_m<0] = 0
   img_m_cr = img[bb_m[1]:(bb_m[1]+bb_m[3]), bb_m[0]:(bb_m[0]+bb_m[2]), :]
   img_m_cr = pad_image(img_m_cr, size_small)
   return img_m_cr

def read_data(read_new = False, read_negative=False):
   if (os.path.exists(datapath+"positive_data.npy") and 
       read_new==False and read_negative==False):
      positive_data = np.load(datapath+"positive_data.npy", allow_pickle=True)
      return positive_data
   if (os.path.exists(datapath+"negative_data.npy") and
       os.path.exists(datapath+"positive_data.npy") and 
       read_new==False and read_negative==True):
      positive_data = np.load(datapath+"positive_data.npy", allow_pickle=True)
      negative_data = np.load(datapath+"negative_data.npy", allow_pickle=True)
      return positive_data, negative_data
   elif (read_new==True):
      positive_data = np.empty((0,7), dtype=object)
      negative_data = np.empty((0,7), dtype=object)

      basefolder = "/home/doran/Work/images/Leeds_May2019/splitbyKM/"
      folders_meta = glob.glob("/home/doran/Work/images/Leeds_May2019/curated_cryptdata/KM*/")
      for ff in folders_meta:
         set_num = ff.split('/')[-2]
         folders_curated = glob.glob(ff + 'Analysed_slides/Analysed_*')
         basefolder_contours = basefolder + set_num + '/Analysed_slides/'
         slide_data = pd.read_csv(basefolder + set_num + '/slide_info.csv')
         slide_data = slide_data.astype({'Image ID':'str'})
         for fsl in folders_curated:
            # get curated crypt net data, find image and get slide
            netdat = np.loadtxt(fsl + '/crypt_network_data.txt')
            clone_inds = np.where(netdat[:,3]>0)[0]
            numclones = clone_inds.shape[0]
            imname = fsl.split('Analysed_')[-1].split('/')[0]
            imloc = basefolder + set_num + '/'
            impath = imloc + imname + '.svs'
            cnts_m_in = load_all_contours2(impath)[0]
            if not (len(cnts_m_in)==netdat.shape[0]): continue
            contour_path = basefolder_contours + 'Analysed_' + imname + '/crypt_contours.txt'
            try:
               mark = slide_data['mark'].iloc[np.where(slide_data['Image ID']==imname)[0][0]]
            except: continue
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

         ## output a subset of negative crypts
         mult = negative_data.shape[0] / positive_data.shape[0]
         idx = np.random.choice(negative_data.shape[0], 
                                size=int(mult / 5)*positive_data.shape[0],
                                replace=False)                             
         np.save(datapath+"positive_data.npy", positive_data)
         np.save(datapath+"negative_data.npy", negative_data[np.sort(idx), :])
      return positive_data, negative_data
   else:
      print("Data not loaded -- set read_new=True to reload.")
      return 0, 0

def read_slide_data(fsl, datapath):
   slide_data = np.empty((0,4), dtype=np.float32)
   # get curated crypt net data, find image and get slide
   netdat = np.loadtxt(fsl + '/crypt_network_data.txt')
   clone_inds = np.where(netdat[:,3]>0)[0]
   numclones = clone_inds.shape[0]
   imname = fsl.split('Analysed_')[-1].split('/')[0]
   imloc = fsl.split('Analysed_slides')[0]
   impath = imloc + imname + '.svs'
   cnts_m_in = load_all_contours2(impath)[0]
   if not (len(cnts_m_in)==netdat.shape[0]):
      print("contours and data not the same size!")
      return(0)
   try:
      mark = slide_info['mark'].iloc[np.where(slide_info['Image ID']==imname)[0][0]]
   except:
      mark = 'unknown'
   ## add positive data
   if numclones>0:
      this_netdat = netdat[clone_inds,:4]
      this_netdat[:,2] = clone_inds
      slide_data = np.row_stack([slide_data , this_netdat])
   ## add negative data
   this_negdat = netdat[np.setdiff1d(range(netdat.shape[0]), clone_inds),:4]
   this_negdat[:,2] = np.setdiff1d(range(netdat.shape[0]), clone_inds)
   numnegs = this_negdat.shape[0]
   slide_data = np.row_stack([slide_data , this_negdat])
   ## output slide data
   np.save(datapath + imname + '_data.npy', slide_data)
   ## output contours as numpy array
   cnts_m_in = np.asarray(cnts_m_in)
   np.save(datapath + imname + '_cnts.npy', cnts_m_in)
   return slide_data, cnts_m_in, impath 

def read_training_data():      
      basefolder = "/home/doran/Work/images/Leeds_May2019/splitbyKM/"
      folders_meta = glob.glob("/home/doran/Work/images/Leeds_May2019/curated_cryptdata/KM*/")
      for ff in folders_meta:
         set_num = ff.split('/')[-2]
         folders_curated = glob.glob(ff + 'Analysed_slides/Analysed_*')
         basefolder_contours = basefolder + set_num + '/Analysed_slides/'
         slide_info = pd.read_csv(basefolder + set_num + '/slide_info.csv')
         slide_info = slide_info.astype({'Image ID':'str'})
         for fsl in folders_curated:
            read_slide_data(fsl, datapath)

def pull_crypt_from_cnt(XY, imgpath, cnt, size_small, 
                        ROTATE=False, RT_m=0, dwnsample_lvl=0):
   cv2.setNumThreads(0)
   if type(cnt)!=list: cnt = [cnt]
   # get slide maximum dims
   slide = osl.OpenSlide(imgpath)
   maxdims = slide.level_dimensions[dwnsample_lvl]
   scale = slide.level_downsamples[dwnsample_lvl]
   xy_m = XY/scale
   if dwnsample_lvl>0:
      cnt = rescale_contours(cnt.copy(), 1./scale)
   # create initial bounding box to get size_big requirements
   lt = int(-60/scale) # add to left and top boundaries
   rb = int(120/scale) # add to width and height
   bb_pad = np.array([lt, lt, rb, rb], dtype=np.int32)
   bb_m = np.asarray(cv2.boundingRect(cnt[0])) + bb_pad
   bb_m[bb_m<0] = 0
   size_big = int(1.5*np.max(bb_m[-2:]))
   # create centred tile
   xy_vals_m_ds_in2 = centred_tile(xy_m, size_big, maxdims, edge_adjust = False)
   cnt_sub = add_offset(cnt.copy(), -xy_vals_m_ds_in2)
   Lx = np.maximum(0,int(np.around(xy_vals_m_ds_in2[0])))
   Ty = np.maximum(0,int(np.around(xy_vals_m_ds_in2[1])))
   # pull out image and rotate if required
   img = getROI_img_osl(imgpath, (Lx,Ty), (size_big, size_big), dwnsample_lvl)
   if ROTATE==True:
      img = cv2.warpAffine(img.copy(), RT_m[:2,:], img.shape[1::-1])
      cnt_sub = [rotate_contour_about_centre(cnt_sub[0], RT_m)]
   # create padded bounded box
   bb_m = np.asarray(cv2.boundingRect(cnt_sub[0])) + bb_pad
   bb_m[bb_m<0] = 0
   img_m_cr = img[bb_m[1]:(bb_m[1]+bb_m[3]), bb_m[0]:(bb_m[0]+bb_m[2]), :]
   img_m_cr = pad_image(img_m_cr, size_small)
   return img_m_cr
 
def pull_crypt_from_cnt_keepbbox(XY, imgpath, cnt, size_small, 
                                 ROTATE=False, RT_m=0, dwnsample_lvl=0):
   cv2.setNumThreads(0)
   if type(cnt)!=list: cnt = [cnt]
   # get slide maximum dims
   slide = osl.OpenSlide(imgpath)
   maxdims = slide.level_dimensions[dwnsample_lvl]
   scale = slide.level_downsamples[dwnsample_lvl]
   xy_m = XY/scale
   if dwnsample_lvl>0:
      cnt = rescale_contours(cnt.copy(), 1./scale)
   # create initial bounding box to get size_big requirements
   lt = int(-60/scale) # add to left and top boundaries
   rb = int(120/scale) # add to width and height
   bb_pad = np.array([lt, lt, rb, rb], dtype=np.int32)
   bb_m = np.asarray(cv2.boundingRect(cnt[0])) + bb_pad
   bb_m[bb_m<0] = 0
   size_big = int(1.5*np.max(bb_m[-2:]))
   # create centred tile
   xy_vals_m_ds_in2 = centred_tile(xy_m, size_big, maxdims, edge_adjust = False)
   cnt_sub = add_offset(cnt.copy(), -xy_vals_m_ds_in2)
   Lx = np.maximum(0,int(np.around(xy_vals_m_ds_in2[0])))
   Ty = np.maximum(0,int(np.around(xy_vals_m_ds_in2[1])))
   # pull out image and rotate if required
   img = getROI_img_osl(imgpath, (Lx,Ty), (size_big, size_big), dwnsample_lvl)
   if ROTATE==True:
      img = cv2.warpAffine(img.copy(), RT_m[:2,:], img.shape[1::-1])
      cnt_sub = [rotate_contour_about_centre(cnt_sub[0], RT_m)]
   # create padded bounded box
   bb_m = np.asarray(cv2.boundingRect(cnt_sub[0])) + bb_pad
   bb_m[bb_m<0] = 0
   img_m_cr = img[bb_m[1]:(bb_m[1]+bb_m[3]), bb_m[0]:(bb_m[0]+bb_m[2]), :]
   img_m_cr, r, pad_lr, pad_tb = pad_image_keeppadding(img_m_cr, size_small)
   # rescale for image resize
   cnt_sub2 = add_offset(cnt.copy(), -(xy_vals_m_ds_in2-bb_m[:2]))
   lt = int(-40/scale) # add to left and top boundaries
   rb = int(80/scale) # add to width and height
   bb_pad2 = np.array([lt, lt, rb, rb], dtype=np.int32)
   bb_out = np.asarray(cv2.boundingRect(cnt_sub2[0])) + bb_pad2
   bb_out = bb_out * r
   # trim for padding
   bb_out = np.array([bb_out[0]+pad_lr, bb_out[1]+pad_tb, bb_out[2]-pad_lr, bb_out[3]-pad_tb])
   return img_m_cr, bb_out
   
def create_nbr_stack(XY, imgpath, cnts, slide_data, ind_m, scr_size=50, nn=30, sampsize=200, multicore=True):
   ## take 20 or 30 closest crypts, and put the centre crypt, ind_m, first
   upr = ind_m + sampsize
   lwr = ind_m - sampsize
   if ind_m<sampsize:
      upr = sampsize - (ind_m-sampsize)
      lwr = 0
   elif (ind_m+sampsize)>slide_data.shape[0]:
      if (ind_m<sampsize):
         pass
      else:
         upr = slide_data.shape[0]
         lwr = ind_m - sampsize - (slide_data.shape[0]-ind_m)
   sub_data = slide_data[np.bitwise_and(slide_data[:,2]<=upr, slide_data[:,2]>=lwr),:]
   nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(sub_data[:,:2])
   _, knn_indices = nbrs.kneighbors(XY[np.newaxis,:])
   this_ind = np.where(sub_data[:,2]==ind_m)[0][0]
   in_inds = np.setdiff1d(knn_indices[0,:], this_ind)
   in_inds = np.hstack([this_ind, in_inds])
   sub_data = sub_data[in_inds,:]
   sldinds = sub_data[:,2].astype(np.int32)
   ## pull crypts and return stack
   crypt_stack = np.zeros((scr_size, scr_size, 3*nn), dtype=np.uint8)
   if multicore:
      cv2.setNumThreads(0)
      num_cores = multiprocessing.cpu_count()
      results = Parallel(n_jobs=num_cores)(delayed(pull_crypt_from_cnt)(sub_data[ii,:2], imgpath, 
                                                   cnts[int(sub_data[ii,2])], scr_size, ROTATE=False, 
                                                   RT_m=0, dwnsample_lvl=1) for ii in range(in_inds.shape[0]))
      for ii in range(in_inds.shape[0]):
         crypt_stack[:,:,(3*ii):(3*(ii+1))] = results[ii]
   else:
      for ii in range(in_inds.shape[0]):
         tind = int(sub_data[ii,2])
         crypt_stack[:,:,(3*ii):(3*(ii+1))] = pull_crypt_from_cnt(sub_data[ii,:2], imgpath, 
                                                                  cnts[tind], scr_size, ROTATE=False, 
                                                                  RT_m=0, dwnsample_lvl=1)
   return crypt_stack

def pull_centered_img_keeporigincoords(XY, imgpath, tilesize, dwnsample_lvl=1):
   # get slide maximum dims
   slide = osl.OpenSlide(imgpath)
   maxdims = slide.level_dimensions[dwnsample_lvl]
   # Fix tiles to specific DNN size
   xy_m = XY/slide.level_downsamples[dwnsample_lvl]
   xy_vals_m_ds_out = centred_tile(xy_m, tilesize, maxdims, edge_adjust = False)
   # pull out images
   Lx = np.maximum(0,int(np.around(xy_vals_m_ds_out[0])))
   Rx = int(np.around((xy_vals_m_ds_out[0]+tilesize)))
   Ty = np.maximum(0,int(np.around(xy_vals_m_ds_out[1])))
   By = int(np.around((xy_vals_m_ds_out[1]+tilesize)))
   img = getROI_img_osl(imgpath, (Lx,Ty), (tilesize, tilesize), dwnsample_lvl)
   # pad if required 
   img, pad_l, pad_t = pad_edge_image_keep_tl_padding(img.copy(), tilesize/2., xy_m, maxdims)
   return img, (Lx,Ty), (pad_l,pad_t)

def get_bounding_box(cnt, imgpath, xy_lt, pad_lt, dwnsample_lvl=0):
   if type(cnt)!=list: cnt = [cnt]
   # rescale contour
   slide = osl.OpenSlide(imgpath)
   scale = slide.level_downsamples[dwnsample_lvl]
   if dwnsample_lvl>0:
      cnt = rescale_contours(cnt.copy(), 1./scale)
   # recenter contour
   xy_rc = np.array([xy_lt[0]-pad_lt[0], xy_lt[1]-pad_lt[1]])
   cnt_sub = add_offset(cnt.copy(), -xy_rc)
   # create bounding box
   lt = int(-60/scale) # add to left and top boundaries
   rb = int(120/scale) # add to width and height
   bb_pad = np.array([lt, lt, rb, rb], dtype=np.int32)
   bb_m = np.asarray(cv2.boundingRect(cnt_sub[0])) + bb_pad
   bb_m[bb_m<0] = 0
   return bb_m

### Maybe save all slide info for each slide separately, (clone and wt together),
# then just use all positive clone data as the input, loading the pertinent slide
# data when needed to find neighbours? (Then load a negative from same slide too)
# (check speed of this)

## choose a positive data point as "base"
#data = positive_data[0,:]
#XY = data[:2]
#imgpath = data[5]
#cntpath = data[6]
#ind_m = int(data[2])
#clone_bool = int(data[3])

### load the relevant slide data
#thisname = datapath + imgpath.split('/')[-1].split('.svs')[0]
#slide_data = np.load(thisname + '_data.npy')
#cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)
   
## then stack up, and use 2D convolutions through to create a encoded vector, and subtract from the high resolution vector.
## (or use 3D convolutions through an extra dimension)

#####################
## load a slide and get the bounding box of the central crypt
#data = positive_data[0,:]
#XY = data[:2]
#imgpath = data[5]
#cntpath = data[6]
#ind_m = int(data[2])
#clone_bool = int(data[3])

#thisname = datapath + imgpath.split('/')[-1].split('.svs')[0]
#slide_data = np.load(thisname + '_data.npy')
#cnts = np.load(thisname + '_cnts.npy', allow_pickle=True)

#dwnsamp = 1
#img, xy, pd = pull_centered_img_keeporigincoords(XY, imgpath, tilesize, dwnsample_lvl=dwnsamp)
#bbox = get_bounding_box(cnts[ind_m], xy, pd, dwnsample_lvl=dwnsamp)
## recast as (xmin, ymin, xmax, ymax) as fractions of the image size
#bbox_fracs = np.array([[bbox[0]/float(img.shape[1]),
#                        bbox[1]/float(img.shape[0]),
#                        (bbox[0] + bbox[2])/float(img.shape[1]),
#                        (bbox[1] + bbox[3])/float(img.shape[0])]])



