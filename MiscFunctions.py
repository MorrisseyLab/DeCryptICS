from __future__ import division
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pyvips
import openslide as osl
from openslide_python_fix import _load_image_lessthan_2_29, _load_image_morethan_2_29
from pathlib import Path

def mkdir_p(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def contour_mean_Area(cnt, img):
    # Get mean colour of object
    roi           = cv2.boundingRect(cnt)
    Start_ij_ROI  = np.array(roi)[0:2] # get x,y of bounding box
    cnt_roi       = cnt - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill     = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill, [cnt_roi], 0, 255, -1) ## Get mask
    mean_col_ii   = cv2.mean(img_ROI, mask_fill)[0]/255.
    return(mean_col_ii)

def bbox_y1_x1_y2_x2(cnti):
    bb_cv2 = cv2.boundingRect(cnti)
    # x,y,w,h -> y1, x1, y2, x2
    return np.array([bb_cv2[1], bb_cv2[0], bb_cv2[1] + bb_cv2[3], bb_cv2[0] + bb_cv2[2]])
    
def box_overlap(A,B):
    return (A[0] < B[2] and A[2] > B[0] and A[1] < B[3] and A[3] > B[1])

def process_input_file(input_file):
   input_file = os.path.abspath(input_file)
   linux_test = len(input_file.split('/'))
   windows_test = len(input_file.split('\\'))
   if (linux_test==1 and windows_test>1):
      base_path = '\\' + os.path.join(*input_file.split('\\')[:-1]) + '\\'
   if (linux_test>1 and windows_test==1):
      base_path = '/' + os.path.join(*input_file.split('/')[:-1]) + '/'
   if (linux_test==1 and windows_test==1):
      base_path = os.getcwd() + '/'
 
   ## Read input file
   ftype = input_file.split('.')[-1]   
   
   ext1 = "svs"; ext2 = "svs"
   if (input_file.split('_')[-1].split('.')[0] == "tif"):
      ext1 = "tif"; ext2 = "tiff"
   if (input_file.split('_')[-1].split('.')[0] == "png"):
      ext1 = "png"; ext2 = "png"
   if (input_file.split('_')[-1].split('.')[0] == "jpg"):
      ext1 = "jpg"; ext2 = "jpeg"

   # check if we loaded a value as a header
   if (ftype=="csv"):      a = pd.read_csv(input_file)
   elif (ftype[:2]=="xl"): a = pd.read_excel(input_file)
   else:                   a = pd.read_table(input_file)
   heads = list(a.columns.values)
   img_sum = 0
   for hh in heads:
      if (hh.split('.')[-1]==ext1 or hh.split('.')[-1]==ext2): img_sum += 1
   if img_sum>0:
      if (ftype=="csv"):      a = pd.read_csv(input_file  , header=None)
      elif (ftype[:2]=="xl"): a = pd.read_excel(input_file, header=None)
      else:                   a = pd.read_table(input_file, header=None)
   a = np.asarray(a).reshape(a.shape)
   
   # extract file paths
   ncols = a.shape[1]   
   for i in range(ncols):
      if type(a[0,i]) == str:
         if (a[0,i].split('.')[-1]==ext1 or a[0,i].split('.')[-1]==ext2):
            pathind = i
   full_paths = list(a[:,pathind]) # take correct column
#   clonal_mark_list = list(a[:,pathind+1])

   linux_test = len(full_paths[0].split('/'))
   windows_test = len(full_paths[0].split('\\'))
   if (linux_test==1 and windows_test>1):
      file_in = [name.split("\\")[-1].split(".")[0] for name in full_paths]
   if (linux_test>1 and windows_test==1):
      file_in = [name.split("/")[-1].split(".")[0] for name in full_paths]
   
   ## Define file structures
   folder_out = base_path + '/' + "Analysed_slides/" 
   mkdir_p(folder_out)
   folders_to_analyse = [folder_out+fldr for fldr in ["Analysed_"+fnum+"/" for fnum in file_in]]
   return base_path, folder_out, file_in, folders_to_analyse, full_paths

def centred_tile(XY, tilesize, max_xy, edge_adjust = True):
   x0y = XY[0] - tilesize/2.
   xy0 = XY[1] - tilesize/2.
   if edge_adjust:
      if (x0y < 0):
         x0y = 0
      if (xy0 < 0):
         xy0 = 0
      xNy = x0y + tilesize
      if (xNy >= max_xy[0]):
         x0y = x0y - (xNy - max_xy[0] + 1)
         xNy = x0y + tilesize
      xyN = xy0 + tilesize
      if (xyN >= max_xy[1]):
         xy0 = xy0 - (xyN - max_xy[1] + 1)
         xyN = xy0 + tilesize
   return np.array([x0y, xy0])

def add_offset(contour_list, xy_offset):
   cnts = contour_list.copy()
   cnt_list_out = []    
   for l in range(len(cnts)):
      shape_prior = cnts[l].shape
      cnt_l = cnts[l].reshape((cnts[l].shape[0], cnts[l].shape[2]))
      new_cnt_l = cnt_l + np.round(xy_offset).astype(np.int32)
      new_cnt_l = new_cnt_l.reshape(shape_prior)
      cnt_list_out.append(new_cnt_l) 
   return cnt_list_out  

def rescale_contours(contour_list, scaling_val):
    cnt_list_out = []
    for cc in contour_list:
      a = (cc*scaling_val).astype(np.int32)
      cnt_list_out.append(a)
    return cnt_list_out

def contour_EccMajorMinorAxis(cnt):
    try:    
        # Get mean colour of object
        _, axes,_ = cv2.fitEllipse(cnt)
    
        # length of MAJOR and minor axis
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)    
        # eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
        eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
    except:
        eccentricity = 0
        majoraxis_length = 0
        minoraxis_length = 0
    return eccentricity, majoraxis_length, minoraxis_length

def contour_Area(cnt):
    if len(cnt) == 1: return 1        
    return(cv2.contourArea(cnt))

def write_cnt_text_file(cnt_list, file_name):
    with open(file_name, 'w') as ff:
        for cnt_i in cnt_list:
            ff.write(','.join(['%f' % num for num in cnt_i[:,0,0]])+"\n")
            ff.write(','.join(['%f' % num for num in cnt_i[:,0,1]])+"\n")
            
def write_score_text_file(clone_scores, file_name):
    with open(file_name, 'w') as ff:
        for score in clone_scores:
            ff.write("%f\n" % score)
            
#def write_cnt_hdf5(cnt_list, cnt_file_name):
#   with h5py.File(cnt_file_name, 'w', libver='latest') as f:  # use 'latest' for performance
#      for idx, arr in enumerate(cnt_list):
#         dset = f.create_dataset(str(idx), shape=arr.shape, data=arr, chunks=arr.shape,
#                                 compression='gzip', compression_opts=9)

def write_clone_image_snips(folder_to_analyse, file_name, clone_contours, scaling_val):
   imgout = folder_to_analyse + "/clone_images/"
   mkdir_p(imgout)
   #smallcnts = rescale_contours(clone_contours, 1./scaling_val) #for level=1
   i = 0
   for cc in clone_contours: # smallcnts here if level=1
      expand_box    = 60*scaling_val #remove scaling val here if level=1
      roi           = cv2.boundingRect(cc)
      roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box), dtype=np.uint64)
      roi[roi<1]   = 0
      img_ROI      = getROI_img_osl(file_name, (roi[0],roi[1]), (roi[2],roi[3]), level = 0) # or level=1?
      outfile      = "/clone_" + str(i) + ".png"
      cv2.imwrite(imgout + outfile, img_ROI)
      i += 1
   return True
            
def read_cnt_text_file(file_name):
    with open(file_name, 'r') as ff:
        contours = ff.readlines()
    numcnts = len(contours)//2
    cnts_out = []
    for i in range(numcnts):
        i *= 2 # x
        j = i+1 # y        
        lx = contours[i][:-1].split(',')
        lx = [int(float(x)) for x in lx]
        ly = contours[j][:-1].split(',')
        ly = [int(float(x)) for x in ly]
        numpnts = len(lx)
        a = np.zeros([numpnts,1,2], dtype=np.int32)
        for k in range(numpnts):
            a[k,0,0] = lx[k]
            a[k,0,1] = ly[k]
        cnts_out.append(a)
    return cnts_out

def load_all_contours2(filename_list, scales = None, H_mats = None):
   cntsout = []
   if not (type(filename_list)==list): filename_list = [filename_list]
   if (scales is not None):
      if not (type(scales)==list): scales = [scales]
   if (H_mats is not None):
      if not (type(H_mats)==list): H_mats = [H_mats]
   for ff in range(len(filename_list)):
      cnts_m = read_cnt_text_file(filename_list[ff][:-len(filename_list[ff].split('/')[-1])] + 'Analysed_slides/Analysed_' + filename_list[ff].split('/')[-1].split('.')[0] + '/crypt_contours.txt')
      if (scales is not None):
         cnts_m = rescale_contours(cnts_m, 1./scales[ff])
      if (H_mats is not None): 
         cnts_m = rotate_contours(cnts_m, H_mats[ff])
      cntsout.append(cnts_m)
   return cntsout

def load_single_contour(filename, cnt_id = 0, scale = None, H_mat = None):   
   with open(filename) as fp:
      for i, line in enumerate(fp):
         if i == (cnt_id*2):
            x = line
         elif i == (cnt_id*2+1):
            y = line
         elif i > (cnt_id*2+1):
            break
   cnt_out = []
   lx = x[:-1].split(',')
   lx = [int(float(cx)) for cx in lx]
   ly = y[:-1].split(',')
   ly = [int(float(cy)) for cy in ly]
   numpnts = len(lx)
   a = np.zeros([numpnts,1,2], dtype=np.int32)
   for k in range(numpnts):
      a[k,0,0] = lx[k]
      a[k,0,1] = ly[k]
   if (scale is not None):
      a = rescale_contours(a, 1./scale)
   if (H_mat is not None): 
      a = rotate_contours(a, H_mat)
   cnt_out.append(a)
   return cnt_out

def rotate_contours(cnts, H_mat):
   new_cnts = []
   for l in range(len(cnts)):
      shape_prior = cnts[l].shape
      cnt_l = cnts[l].reshape((cnts[l].shape[0],cnts[l].shape[2]))
      new_cnt_l = np.around(transform_XY(cnt_l, H_mat)).astype(np.int32)
      new_cnt_l = new_cnt_l.reshape(shape_prior)
      new_cnts.append(new_cnt_l)
   return new_cnts

def contour_xy(cnt, reverse = False):
    m_ij   = cv2.moments(cnt)
    if m_ij['m00']==0:
        pos_xy = (np.mean(cnt[:,0,0]), np.mean(cnt[:,0,1]))
    else:
        pos_xy = (int(m_ij['m10']/m_ij['m00']), int(m_ij['m01']/m_ij['m00']))
    if reverse:
        pos_xy = (pos_xy[1],pos_xy[0])
    return(pos_xy)

def transform_XY(XY, H_mat):
   XYZ = np.ones((3, XY.shape[0]))
   XYZ[:2,:] = XY.T
   new_XYZ = np.dot(H_mat, XYZ)
   return new_XYZ[:2,:].T

def RotationTranslationMatrix(theta, xt, yt, midpointxy, opencv=True):
   # rotate by rot_angle theta (-180,180] and then translate by vector (xt, yt)
   theta_rad = theta/180. * np.pi
   if opencv==False:
      RT = np.array([[np.cos(theta_rad), -np.sin(theta_rad), xt],
                     [np.sin(theta_rad), np.cos(theta_rad) , yt],
                     [0            , 0             , 1 ]])
   else:
      RT = np.array([[np.cos(theta_rad) , np.sin(theta_rad), xt],
                     [-np.sin(theta_rad), np.cos(theta_rad), yt],
                     [0            , 0             , 1 ]])
   # alter rotation axis from origin to centre of image
   cx_f = midpointxy[0]
   cy_f = midpointxy[1]
   rot_mat = RT.copy()
   rot_mat[:2, 2] = np.array([0,0]) # extract pure rotation
   extra_translate = (np.around(np.dot(rot_mat, np.array([cx_f, cy_f, 1])))).astype(np.int32)
   RT[:2,2] = RT[:2,2] - extra_translate[:2] + np.array([cx_f, cy_f])
   return RT

def rotate_contour_about_centre(cnt, H_mat, centre=None):
   shape_prior = cnt.shape
   cnt = cnt.reshape((cnt.shape[0],cnt.shape[2]))
   if centre==None:
      try:
         centre = contour_xy(cnt)
      except:
         centre = np.mean(cnt, axis=0)
   theta = np.angle(H_mat[0,0] + 1j*H_mat[0,1], deg=True)
   RT = RotationTranslationMatrix(theta, 0, 0, centre)   
   new_cnt = np.around(transform_XY(cnt, RT)).astype(np.int32)
   new_cnt = new_cnt.reshape(shape_prior)
   return new_cnt

def simplify_contours(cnt_list):
    new_cnts = []
    for i in cnt_list:
        approx = cv2.approxPolyDP(i, 0.01*cv2.arcLength(i,True), True)
        if (len(approx)<10):
            new_cnts.append(i)
        else:
            new_cnts.append(approx)
    return new_cnts

def plot_histogram(x, bins=50, norm_it = False):
    hist, bins = np.histogram(x, bins=bins, normed = norm_it)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()  

def thresh_img(img_deconv2, channel, thesh):
    _, img_threshed = cv2.threshold(img_deconv2[:,:,channel], thesh, 255, cv2.THRESH_BINARY)        
    if img_threshed.dtype != np.uint8: img_threshed = img_threshed.astype('uint8', copy=False) 
    return(img_threshed)

def binaryTo3C(img):
    return cv2.merge((img, img ,img))

## For memory management use just in place ops
def transform_OD(img):
    OD_data = img+np.float32(1.0)
    OD_data /= 256. ## In place
    OD_data = cv2.log(OD_data, OD_data) ## In place
    OD_data *= -1. ## In place
    return(OD_data)
    
def col_deconvol(img, deconv_mat):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False)
    img_deconv = cv2.transform(OD_data, deconv_mat)
    ## Convert to 8 bits
    img_deconv = np.clip(img_deconv, 0, 1, out=img_deconv)
    img_deconv *= 255
    img_deconv = img_deconv.astype('uint8', copy=False) 
    return(img_deconv)

def col_deconvol_blur_clone(img, deconv_mat, size_blur):
    # deconvolution
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)
    ## blur
    nucl_blur = cv2.GaussianBlur(img_deconv[:,:,0], size_blur1, 0)
    clone_blur      = cv2.GaussianBlur(img_deconv[:,:,1], size_blur, 0)
    ## convert to 8 bits
    clone_blur = np.clip(clone_blur, 0, 1, out=clone_blur)
    clone_blur *= 255
    clone_blur = clone_blur.astype('uint8', copy=False) 
    return clone_blur

## If you convert to 8bit before blurring you lose resolution
def col_deconvol_and_blur2(img, deconv_mat, size_blur1, size_blur2):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)    
    nucl_blur  = cv2.GaussianBlur(img_deconv[:,:,0], size_blur1, 0)
    clone_blur = cv2.GaussianBlur(img_deconv[:,:,1], size_blur2, 0)
    return nucl_blur, clone_blur

def col_deconvol_and_blur3(img, deconv_mat, size_blur1, size_blur2):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)    
    nucl_blur  = cv2.GaussianBlur(img_deconv[:,:,0], size_blur1, 0)
    clone_blur = cv2.GaussianBlur(img_deconv[:,:,1], size_blur2, 0)
    background_blur = cv2.GaussianBlur(img_deconv[:,:,2], size_blur1, 0)
    return nucl_blur, clone_blur, background_blur

## If you convert to 8bit before blurring you lose resolution
def col_deconvol_and_blur(img, deconv_mat, size_blur1, size_blur2, size_blur3):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)    
    nucl_blur_small = cv2.GaussianBlur(img_deconv[:,:,0], size_blur1, 0)
    nucl_blur_large = cv2.GaussianBlur(img_deconv[:,:,0], size_blur2, 0)
    clone_blur      = cv2.GaussianBlur(img_deconv[:,:,1], size_blur3, 0)
    
    ## Convert to 8 bits
    nucl_blur_small = np.clip(nucl_blur_small, 0, 1, out=nucl_blur_small)
    nucl_blur_small *= 255
    nucl_blur_small = nucl_blur_small.astype('uint8', copy=False) 

    ## Convert to 8 bits
    nucl_blur_large = np.clip(nucl_blur_large, 0, 1, out=nucl_blur_large)
    nucl_blur_large *= 255
    nucl_blur_large = nucl_blur_large.astype('uint8', copy=False) 

    ## Convert to 8 bits
    clone_blur = np.clip(clone_blur, 0, 1, out=clone_blur)
    clone_blur *= 255
    clone_blur = clone_blur.astype('uint8', copy=False) 
    
    return nucl_blur_small, nucl_blur_large, clone_blur

def col_deconvol_32(img, deconv_mat):
    OD_data    = transform_OD(img)
    deconv_mat = deconv_mat.astype('float32', copy=False) 
    img_deconv = cv2.transform(OD_data, deconv_mat)
    return(img_deconv)

def plot_img(list_to_plot, nrow = 1, nameWindow = 'Plots', NewWindow = True, hold_plot = True):
    num_images = len(list_to_plot)
    num_cols   = int(num_images/nrow)
    if num_images%nrow != 0:
        raise(UserWarning, "If more than one row make sure there are enough images!")
    if NewWindow:
        screen_res = 1600, 1000
        cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nameWindow, screen_res[0], screen_res[1])
    if isinstance(list_to_plot, tuple) == 0: 
        vis = list_to_plot
    else:
        last_val = num_cols 
        vis      = np.concatenate(list_to_plot[0:last_val], axis=1)
        for row_i in range(1, nrow):
            first_val = last_val
            last_val  = first_val + num_cols
#            print (first_val,last_val)
            vis_aux   = np.concatenate(list_to_plot[first_val:last_val], axis=1)
            vis       = np.concatenate((vis, vis_aux), axis=0)        
    cv2.imshow(nameWindow, vis)
    if(hold_plot):
        0xFF & cv2.waitKey(0)
        cv2.destroyWindow(nameWindow)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def make8bit(img_in):
    # Saturate values
    img_deconv_255 = 255*np.clip(img_in, 0, 1)
    # Covert to 8 bit
    img_deconv_255 = img_deconv_255.astype('uint8', copy=False) 
    return(img_deconv_255)

## Correct width and height for cropping so that it never overshoots the 
## size of the image
def correct_wh(max_vals, xy_vals, wh_vals):
    final_x = xy_vals[0] + wh_vals[0]
    final_y = xy_vals[1] + wh_vals[1]
    new_wh_x = wh_vals[0]
    new_wh_y = wh_vals[1]
    if final_x > max_vals[0] : new_wh_x = max_vals[0] - xy_vals[0]
    if final_y > max_vals[1] : new_wh_y = max_vals[1] - xy_vals[1]    
    return int(new_wh_x), int(new_wh_y)

# def getROI_img_vips(file_name, x_y, w_h, level = 0):
#     vim           = pyvips.Image.openslideload(file_name, level = level)   #openslideload
#     max_vals      = (vim.width, vim.height)
#     wh_vals_final = correct_wh(max_vals, x_y, w_h) ## Correct rounding errors 
#     area          = vim.extract_area(x_y[0], x_y[1], wh_vals_final[0], wh_vals_final[1])
#     size          = (area.width, area.height)
#     data          = area.write_to_memory()
#     new_img       = np.fromstring(data, dtype=np.uint8).reshape(size[1], size[0], 4)  ## Remove alpha channel  
#     new_img       = cv2.cvtColor(new_img[:,:,0:3], cv2.COLOR_RGB2BGR)
#     return new_img

def getROI_img_vips(file_name, x_y, w_h, level = 0):
    vim           = pyvips.Image.openslideload(file_name, level = level)   #openslideload
    max_vals      = (vim.width, vim.height)
    wh_vals_final = correct_wh(max_vals, x_y, w_h) ## Correct rounding errors 
    area          = vim.crop(x_y[0], x_y[1], wh_vals_final[0], wh_vals_final[1])   
    new_img       = np.ndarray(buffer=area.write_to_memory(),
                       dtype=np.uint8,
                       shape=[area.height, area.width, area.bands])
    new_img       = cv2.cvtColor(new_img[:,:,0:3], cv2.COLOR_RGB2BGR)
    return new_img


def getROI_img_osl(file_name, x_y, w_h, level = 0):
    vim           = osl.OpenSlide(file_name)
    max_vals      = vim.level_dimensions[level]
    wh_vals_final = correct_wh(max_vals, x_y, w_h) ## Correct rounding errors    
    newxy = tuple([int(vim.level_downsamples[level]*f) for f in x_y])
    # Check which _load_image() function to use depending on the size of the region.
    if (wh_vals_final[0] * wh_vals_final[1]) >= 2**29:
        osl.lowlevel._load_image = _load_image_morethan_2_29
    else:
        osl.lowlevel._load_image = _load_image_lessthan_2_29    
    new_img = np.array(vim.read_region(location = newxy, level = level, size = wh_vals_final))
    new_img = cv2.cvtColor(new_img[:,:,0:3], cv2.COLOR_RGB2BGR)
    return new_img
    
def getROI_img(file_name, x_y, w_h, level = 0, lib = 'vips'):
   if lib=='osl':
      return getROI_img_osl(file_name, x_y, w_h, level = level)
   elif lib=='vips':
      return getROI_img_vips(file_name, x_y, w_h, level = level)
   else:
      print("choose a valid image library: lib = 'vips' or 'osl'")
      return 1
    
def getIndexesTileImage(max_vals, scalingVal, ROI_crop, max_num_pix  = 10000): # 22000
    ## Index stuff to tile image
    start_indx      = (int(ROI_crop[0][0]*scalingVal), int(ROI_crop[0][1]*scalingVal))
    full_delta_s    = (int(ROI_crop[1][0]*scalingVal) - start_indx[0], int(ROI_crop[1][1]*scalingVal) - start_indx[1])
    last_x          = start_indx[0] + full_delta_s[0]
    last_y          = start_indx[1] + full_delta_s[1]
    ## Make sure you don't overshoot the image
    if last_x>max_vals[0]: last_x = max_vals[0] # x -> cols
    if last_y>max_vals[1]: last_y = max_vals[1] # y -> rows
    last_indx       = (last_x, last_y)
    num_tiles       = [int(np.ceil(float(full_delta_s[0])/max_num_pix)), int(np.ceil(float(full_delta_s[1])/max_num_pix))]
    if num_tiles[0] == 0: num_tiles[0] = 1
    if num_tiles[1] == 0: num_tiles[1] = 1
        
    delta_x         = np.ceil(full_delta_s[0]/num_tiles[0])
    delta_y         = np.ceil(full_delta_s[1]/num_tiles[1])
        
    overlap = 175
    if (delta_x<overlap):
        overlap = delta_x//2
    if (delta_y<overlap):
        overlap = delta_y//2
    all_indx = []
    for i in range(num_tiles[0]):
        if (i==0):
            x0      = start_indx[0]
            width_i = delta_x 
        else:
            x0 = start_indx[0] + i*delta_x - overlap
            width_i = delta_x + overlap
        if (i == (num_tiles[0]-1)): width_i = last_indx[0] - x0
        all_indx.append([])
        for j in range(num_tiles[1]):
            if (j==0):    
                y0       = start_indx[1] + j*delta_y
                height_j = delta_y
            else:
                y0       = start_indx[1] + j*delta_y - overlap
                height_j = delta_y + overlap
            if j == (num_tiles[1]-1): height_j = last_indx[1] - y0
            all_indx[i].append((x0, y0, width_i, height_j))
    return all_indx

def plotImageAndFit(indx_True, indx_on, crypt_cnt_raw, img, indx_subset = None):
    if indx_subset is None: 
        indx_subset = np.ones(len(crypt_cnt_raw),dtype= np.bool)
    if indx_True is None: 
        indx_True = np.ones(len(crypt_cnt_raw),dtype= np.bool)
        
    ## Try to plot results on image
    crypt_cnt_subset = [cnt_i for is_crypt, cnt_i in zip(indx_subset, crypt_cnt_raw) if is_crypt]
    crypt_cnt_mine   = [cnt_i for is_crypt, cnt_i in zip(indx_on, crypt_cnt_subset) if is_crypt]
    crypt_cnt_EM     = [cnt_i for is_crypt, cnt_i in zip(indx_True, crypt_cnt_subset) if is_crypt]
    img_plot         = img.copy()
    #cv2.drawContours(img_plot,  crypts_erode_cnt, -1, (255,0,0), 7)    
    cv2.drawContours(img_plot,   crypt_cnt_EM, -1, (  0,  0, 255), 12) 
    cv2.drawContours(img_plot, crypt_cnt_mine, -1, (255,  0,   0),  6) 
    plot_img(img_plot, hold_plot=True)

