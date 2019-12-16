# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:43:39 2015

@author: edward
"""

import cv2
import numpy as np
from MiscFunctions import plot_img
from sklearn.neighbors import NearestNeighbors

## Define standard structuring elements
st_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
st_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
st_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) 
st_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def get_centre_coords(cnts):
   numcnts = len(cnts)
   xy_coords = np.zeros([numcnts, 2], dtype=np.int32)
   for i in range(numcnts):
      M = cv2.moments(cnts[i])
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      xy_coords[i,0] = cX
      xy_coords[i,1] = cY
   return xy_coords

def find_if_close(xy_1, xy_2, max_distance):   
    dist = np.linalg.norm(xy_1-xy_2)    
    if abs(dist) < max_distance:
        return True
    else:
        return False

## Modified version of 
## http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
def joinContoursIfClose(contours, max_distance = 400):
    cnt_xy   = np.array([contour_xy(cnt_i) for cnt_i in contours])
    num_cnt  = len(contours)
    clusters = np.arange(num_cnt)    
    for indx1 in range(num_cnt-1):
        cnt_xy_1 = cnt_xy[indx1,:]
        for indx2 in range(indx1+1, num_cnt):
            cnt_xy_2 = cnt_xy[indx2,:]
            is_close = find_if_close(cnt_xy_1, cnt_xy_2, max_distance)
            if is_close:
                val = min(clusters[indx1], clusters[indx2])
                clusters[indx2] = clusters[indx1] = val    
    cnt_joined = []
    maximum = int(clusters.max())+1
    for i in range(maximum):
        pos = np.where(clusters==i)[0]
        if pos.size != 0:
            cont = np.vstack([contours[i] for i in pos])
            hull = cv2.convexHull(cont)
            cnt_joined.append(hull)
    return cnt_joined

def add_nearby_clones(patch, indices, clone_inds, i, j):
   patch.append(i)
   for k in indices[i, 1:]:
      if k in clone_inds:
         if k not in patch:
            add_nearby_clones(patch, indices, clone_inds, k, j)
   return patch

def joinContoursIfClose_OnlyKeepPatches(crypt_contours, crypt_dict, clone_inds):
   nn = np.minimum(9, len(crypt_contours)-1)
   nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(crypt_dict['crypt_xy'])
   distances, indices = nbrs.kneighbors(crypt_dict['crypt_xy'])    
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
      used_patches.append(thispatch)
      subset_bool = [thispatch.issubset(aset) for aset in cut_patches]
      good_subsets = np.where(subset_bool)[0]
      for jj in good_subsets:
         if cut_patches[jj] not in used_patches:
            thispatch |= cut_patches[jj]
      cut_patches2.append(thispatch)
   
   # check lengths and occurrences of indices
#   allinds = []
#   for pp in cut_patches2:
#      for ind in pp: allinds.append(ind)
    
   # old; broken
   # join any repeated subsets
#   cut_patches2 = []
#   j = 0
#   joined_patch_ids = []
#   for s in cut_patches:
#      if j not in joined_patch_ids:
#         curr_set = s.copy()
#         joined_patch_ids.append(j)
#         for ind in s:
#            for i in range(j+1, len(cut_patches)):
#               s2 = cut_patches[i]
#               if ind in s2:
#                  curr_set |= s2
#                  joined_patch_ids.append(i)
#         cut_patches2.append(curr_set)
#         j += 1
   patch_size = [len(s) for s in cut_patches2]
   cnt_joined = []
   for patch in cut_patches2:
      cont = np.vstack([np.array(crypt_contours[i]) for i in patch])
      hull = cv2.convexHull(cont)
      cnt_joined.append(hull)
   newpatchinds = []
   for patch in cut_patches2:
      patch = list(patch)
      newpatchinds.append(patch)
   return cnt_joined, patch_size, newpatchinds


## Modified version of 
## http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
#    cnt_xy   = np.array([contour_xy(cnt_i) for cnt_i in contours])
#    num_cnt  = len(contours)
#    if (num_cnt<2):
#      return []
#    clusters = np.arange(num_cnt)    
#    for indx1 in range(num_cnt-1):
#        cnt_xy_1 = cnt_xy[indx1,:]
#        for indx2 in range(indx1+1, num_cnt):
#            cnt_xy_2 = cnt_xy[indx2,:]
#            is_close = find_if_close(cnt_xy_1, cnt_xy_2, max_distance)
#            if is_close:
#                val = min(clusters[indx1], clusters[indx2])
#                clusters[indx2] = val
#                clusters[indx1] = val    
#    cnt_joined = []
#    maximum = int(clusters.max())+1
#    for i in range(maximum):
#        pos = np.where(clusters==i)[0]
#        if (pos.size > 1):
#            cont = np.vstack(contours[i] for i in pos)
#            hull = cv2.convexHull(cont)
#            cnt_joined.append(hull)
#    return cnt_joined

## Giving cv2.drawContours all contours is slower than looping 
def drawAllCont(img_new, all_cnt, cnt_num, col, line_width):
    all_cnt_loop = all_cnt
    if cnt_num != -1: all_cnt_loop = [all_cnt[cnt_num]]
    for cnt_i in all_cnt_loop:
        cv2.drawContours(img_new, [cnt_i], 0, col, line_width)
    return img_new

def filterList(list_i, index_keep):
    return [elem_i for elem_i,indx_i in zip(list_i, index_keep) if indx_i]

def formatString(np_vector):
    str_out   = "["
    np_vector = np.round(np_vector, 2)
    for elem_i in np_vector:
        str_out += (" " + str(elem_i) + ",")
    ## Replace last coma by finishing bracket
    str_out = str_out[0:-1] + "]"
    return str_out  
    
def plotCntAndFeat(img, cnt_list, feat_mat, indx_use = None):
    if indx_use is None: indx_use = np.zeros(len(cnt_list))
    if (indx_use==[]): return 0 # D.K. if all contours are thrown away
    img_plot  = img.copy()
    for ii in range(len(cnt_list)):
        cnt = cnt_list[ii]
        feat_ii = formatString(feat_mat[ii,:]) # str(np.round(feat_mat[ii,:], 2))
        col_use = (200, 255, 0)
        if indx_use[ii]: col_use = (100, 0, 255) #0.427    
        cv2.drawContours(img_plot, [cnt], 0, col_use, 6)
        cv2.putText(img_plot, feat_ii, tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 120, 0), 1)
    plot_img(img_plot, hold_plot = True)
    
def plotThrownCnts(img, cnt_list, feat_mat, indx_use):
    cnt_list = filterList(cnt_list, np.invert(indx_use))
    img_plot  = img.copy()
    for ii in range(len(cnt_list)):
        cnt = cnt_list[ii]
        feat_ii = formatString(feat_mat[ii,:])
        col_use = (200, 255, 0)  
        cv2.drawContours(img_plot, [cnt], 0, col_use, 6)
        cv2.putText(img_plot, feat_ii, tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 120, 0), 1)
    plot_img(img_plot, hold_plot = True)

def plotCnt(img, cnt_list, indx_use = None):
    if indx_use is None: indx_use = np.zeros(len(cnt_list))
    if (indx_use==[]): return 0 # D.K. if all contours are thrown away
    img_plot  = img.copy()
    for ii in range(len(cnt_list)):
        cnt = cnt_list[ii]
        col_use = (50, 255, 50)
        if indx_use[ii]: col_use = (100, 0, 255)
        cv2.drawContours(img_plot, [cnt], 0, col_use, 6)
    plot_img(img_plot, hold_plot = True)

def drawConts2col(indx_on, crypt_cnt_raw, img):
    ## Try to plot results on image
    crypt_cnt_true   = [cnt_i for is_crypt, cnt_i in zip(indx_on, crypt_cnt_raw) if is_crypt]
    crypt_cnt_false  = [cnt_i for is_crypt, cnt_i in zip(indx_on, crypt_cnt_raw) if not is_crypt]
    img_plot         = img.copy()
    drawAllCont(img_plot,   crypt_cnt_false, -1, (  0,  0, 255), 12) #cv2.drawContours(img_plot,   crypt_cnt_false, -1, (  0,  0, 255), 12) 
    drawAllCont(img_plot, crypt_cnt_true, -1, (255,  0,   0),  6) #cv2.drawContours(img_plot, crypt_cnt_true, -1, (255,  0,   0),  6) 
    return img_plot

def contour_MajorMinorAxis(cnt):
    # Get mean colour of object
    _, axes,_ = cv2.fitEllipse(cnt)  
    # length of MAJOR and minor axis
    majoraxis_length = max(axes)
    minoraxis_length = min(axes)    
    return majoraxis_length, minoraxis_length

## Taken from http://opencvpython.blogspot.co.uk/2012/04/contour-features.html
def contour_eccentricity(cnt):
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
    return(eccentricity)
    
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

# Retrieve the contour of white blobs (with no holes in them)
def getContourWhiteBlobs(bin_img, maxSize=1e9, minSize = 100):
    crypt_cnt_plot_raw, h_cnt_info  = cv2.findContours(bin_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    if len(crypt_cnt_plot_raw) < 1: return crypt_cnt_plot_raw ## If no contours return empty
    is_inner = h_cnt_info[0,:,2] == -1
    crypt_cnt_final = [i for i,keep_me in zip(crypt_cnt_plot_raw, is_inner) if keep_me and 
        contour_Area(i) > minSize                and 
        bin_img[contour_xy(i,reverse = True)]!=0 and
        contour_Area(i) < maxSize]
    return crypt_cnt_final

# Retrieve the contour of white halos (in uninverted binary image)
def getContourWhiteHalos(bin_img, throw_outer = True):
    crypt_cnt_plot_raw, h_cnt_info  = cv2.findContours(bin_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    numcnts = len(crypt_cnt_plot_raw)
    if (numcnts < 1 or (not throw_outer)):
        return crypt_cnt_plot_raw ## If no contours return empty/return all
    else:
        is_inner = h_cnt_info[0,:,2] == -1
        # don't keep the outer contours of the hierarchy (throw outer ring)
        crypt_cnt_final = [i for i,keep_me in zip(crypt_cnt_plot_raw, is_inner) if keep_me]
        return crypt_cnt_final
    
def redrawBinaryFromCnts(bin_img, contours):
    img_new         = np.zeros(bin_img.shape, dtype = np.uint8)
    drawAllCont(img_new, contours, -1, 255, -1)
    return(img_new)
    
## Correct halo if object is intersected by background and at least of a certain value
def getCorrectionHalo(cnt_i, background_img, allHalo_i):
    min_halo          = 0.7
    allHalo_corrected = allHalo_i
    if contour_sum_Area(cnt_i, background_img)!=0 and allHalo_i > min_halo:
       # Get mean colour of object
        MA,ma             = contour_MajorMinorAxis(cnt_i)
        allHalo_corrected = allHalo_i*(1. + ma/(2.*MA+2.*ma))
        if allHalo_corrected > 0.9: allHalo_corrected = 0.9
    return allHalo_corrected

def contour_max_Halo(cnt_i, img1, debug_plot = False):
    # Max and min halo size to calculate
    start_diff = 1 # min diff to check 
    end_diff   = 8 # end_diff -1 max diff to check
    # Expand box
    expand_box    = 50
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_roi       = cnt_i - Start_ij_ROI # chnage coords to start from x,y
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1) ## Get mask
     
    max_dilations      = 20
    img_plot           = img_ROI.copy()
    sum_dilations      = np.zeros(max_dilations+1)
    areas_dilations    = np.zeros(max_dilations+1)
    # Area and sum pre-dilations
    areas_dilations[0] = cv2.countNonZero(mask_fill1)
    sum_dilations[0]   = cv2.mean(img_ROI, mask_fill1)[0]/255. * areas_dilations[0]
    for i in range(1, max_dilations+1):        
        mask_fill1          = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = 1)
        areas_dilations[i]  = cv2.countNonZero(mask_fill1)
        sum_dilations[i]    = cv2.mean(img_ROI, mask_fill1)[0]/255. * areas_dilations[i]
        if debug_plot: 
            cnt_aux1, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
            drawAllCont(img_plot, cnt_aux1, -1, 150, 1) #cv2.drawContours(img_plot, cnt_aux1, -1, 150, 1) ## Get mask
    if debug_plot:
        plot_img((img_ROI, img_plot), nrow = 1)
        0xFF & cv2.waitKey()
        cv2.destroyAllWindows()
    num_diffs  = end_diff-start_diff
    max_each = np.zeros(num_diffs)
    indices = []
    for diff_size, ii in zip(range(start_diff,end_diff), range(num_diffs)):
        indx_1    = range(diff_size,len(sum_dilations))
        indx_2    = range(0,len(sum_dilations)-diff_size)
        halo_mean = (sum_dilations[indx_1] - sum_dilations[indx_2])/(areas_dilations[indx_1] - areas_dilations[indx_2])
        max_each[ii] = np.max(halo_mean)
        maxindx = np.where(halo_mean==max_each[ii])[0][0]        
        middle_contour_number = (indx_1[maxindx]+indx_2[maxindx])/2.
        indices.append(middle_contour_number)         
    maxhalo = np.max(max_each)
    maxindx_global = np.where(max_each==maxhalo)[0][0]
    maxmiddlecontour = int(np.ceil(indices[maxindx_global]))
    mid_halo_cnt = extractHaloContour(cnt_roi, img_ROI, maxmiddlecontour)
#    plotCnt(img_plot, mid_halo_cnt)
    mid_halo_cnt = mid_halo_cnt + Start_ij_ROI # re-shift to full-image coords
    output_cnt = np.zeros([mid_halo_cnt.shape[1], 1, mid_halo_cnt.shape[3]], dtype=np.int32)
    for ii in range(mid_halo_cnt.shape[1]):
        output_cnt[ii, 0, :] = mid_halo_cnt[0, ii, 0, :]
    return maxhalo, output_cnt

def inner_signal(cnt_i, img1, debug_plot = False):
    # Max and min halo size to calculate
    start_diff = 1 # min diff to check 
    end_diff   = 8 # end_diff -1 max diff to check
    # Expand box
    expand_box    = 20
    roi           = cv2.boundingRect(cnt_i)            
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_roi       = cnt_i - Start_ij_ROI # chnage coords to start from x,y
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1) ## Get mask
     
    max_erosions       = 10
    img_plot           = img_ROI.copy()
    sum_dilations      = np.zeros(max_erosions+1)
    areas_dilations    = np.zeros(max_erosions+1)
    areas_dilations[i]  = cv2.countNonZero(mask_fill1)
    sum_dilations[i]    = cv2.mean(img_ROI, mask_fill1)[0]/255. * areas_dilations[i]
    # Area and sum pre-dilations
    areas_dilations[0] = cv2.countNonZero(mask_fill1)
    sum_dilations[0]   = cv2.mean(img_ROI, mask_fill1)[0]/255. * areas_dilations[0]
    for i in range(1, max_erosions+1):        
        mask_fill1          = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = 1)

        if debug_plot: 
            cnt_aux1, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
            drawAllCont(img_plot, cnt_aux1, -1, 150, 1) #cv2.drawContours(img_plot, cnt_aux1, -1, 150, 1) ## Get mask
    if debug_plot:
        plot_img((img_ROI, img_plot), nrow = 1)
        0xFF & cv2.waitKey()
        cv2.destroyAllWindows()
    num_diffs  = end_diff-start_diff
    max_each = np.zeros(num_diffs)
    indices = []
    for diff_size, ii in zip(range(start_diff,end_diff), range(num_diffs)):
        indx_1    = range(diff_size,len(sum_dilations))
        indx_2    = range(0,len(sum_dilations)-diff_size)
        halo_mean = (sum_dilations[indx_1] - sum_dilations[indx_2])/(areas_dilations[indx_1] - areas_dilations[indx_2])
        max_each[ii] = np.max(halo_mean)
        maxindx = np.where(halo_mean==max_each[ii])[0][0]        
        middle_contour_number = (indx_1[maxindx]+indx_2[maxindx])/2.
        indices.append(middle_contour_number)         
    maxhalo = np.max(max_each)
    maxindx_global = np.where(max_each==maxhalo)[0][0]
    maxmiddlecontour = int(np.ceil(indices[maxindx_global]))
    mid_halo_cnt = extractHaloContour(cnt_roi, img_ROI, maxmiddlecontour)
#    plotCnt(img_plot, mid_halo_cnt)
    mid_halo_cnt = mid_halo_cnt + Start_ij_ROI # re-shift to full-image coords
    output_cnt = np.zeros([mid_halo_cnt.shape[1], 1, mid_halo_cnt.shape[3]], dtype=np.int32)
    for ii in range(mid_halo_cnt.shape[1]):
        output_cnt[ii, 0, :] = mid_halo_cnt[0, ii, 0, :]
    return maxhalo, output_cnt

def extractHaloContour(cnt_roi, img_ROI, num_dilations):
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1)
    mask_fill1          = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = num_dilations)
    halo_cnt, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    return halo_cnt

def calc_Halo_Gap(cnt_i, img1):
    roi           = cv2.boundingRect(cnt_i)
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_i       = cnt_i - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] # note here the use of y coord first!
    maxgap = 0
    gap = 0
    #totgap = 0
    num_pixels = cnt_i.shape[0]
    last_point = cnt_i[0].shape[0] - 1
    curpoint = 0
    endongap = False
    for xy_i in cnt_i[:,0,:]:        
        x = xy_i[0]
        y = xy_i[1]        
        curpoint += 1
        if (img_ROI[y,x]==0): # note here the use of y coord first!
            gap += 1 # count zeros
            #totgap += 1
            if (gap > maxgap): maxgap = gap
            if (curpoint==last_point):
                endongap = True            
        else:
            if (gap > maxgap): maxgap = gap # update max if required
            gap = 0 # reset gap counter
    if (endongap):
        curpoint = 0
        while(gap>0):            
            x = cnt_i[curpoint,0,0]
            y = cnt_i[curpoint,0,1]
            curpoint += 1
            if (img_ROI[y,x]==0): # note here the use of y coord first!
                gap += 1 # count zeros
                if (gap > maxgap): maxgap = gap
                if (curpoint==last_point):
                    break # to stop going round in circles!
            else:
                if (gap > maxgap): maxgap = gap # update max if required
                gap = 0 # reset gap counter
    return(float(maxgap)/num_pixels)                  

def remove_wiggles(cnt_i, img1, indx_i):
    # Get region of interest and shift coords to 0,0
    roi           = cv2.boundingRect(cnt_i)
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_roi       = cnt_i - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] # note here the use of y coord first!    
    # Create binary mask for contour
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1) ## Get mask
    # Get area of original contour
    area = contour_Area(cnt_roi)
    # Define the size of opening/erosion that it can take
    radius_approx  = np.sqrt(float(area)/np.pi)
    max_eros_level = int(radius_approx/3.)//2
    if (max_eros_level==0):
        return cnt_i
    # Do openings/erosions
    if (indx_i==0):
        mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_ERODE, st_3, iterations = max_eros_level + 1)
        mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_OPEN, st_3, iterations = max_eros_level)
    if (indx_i==1):
        mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_ERODE, st_3, iterations = max_eros_level)
        mask_fill1 = cv2.morphologyEx(mask_fill1, cv2.MORPH_OPEN, st_3, iterations = max_eros_level)
    # Extract new contours from ROI
    new_cnt, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # If more than one contour now, take largest; if none return original contour
    numcnts = len(new_cnt)
    if (numcnts==0):
        return cnt_i
    elif (numcnts==1):
        return np.asarray(new_cnt[0]) + Start_ij_ROI
    else:        
        areas = []
        for i in range(numcnts):
            areas.append(contour_Area(new_cnt[i]))
        maxarea = np.where(areas==np.max(areas))[0][0]
        return np.asarray(new_cnt[maxarea]) + Start_ij_ROI    

def contour_mean_Halo2(cnt, img, halo_dilations = 3, debug_plot = False):
    # Get mean colour of object
    roi           = cv2.boundingRect(cnt)            
    # Expand box
    expand_box    = 30 + halo_dilations*5
    roi = np.array((roi[0]-expand_box, roi[1]-expand_box,  roi[2]+2*expand_box, roi[3]+2*expand_box))
    roi[roi <1]   = 0
    Start_ij_ROI  = roi[0:2] # get x,y of bounding box
    cnt_roi       = cnt - Start_ij_ROI # chnage coords to start from x,y
    img_ROI       = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]   
    mask_fill1    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill1, [cnt_roi], 0, 255, -1) ## Get mask
    mask_fill1   = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = 4)
    mask_fill2   = cv2.morphologyEx(mask_fill1, cv2.MORPH_DILATE, st_5, iterations = halo_dilations)
    mask_fill    = mask_fill2 - mask_fill1       
    mean_col_ii  = cv2.mean(img_ROI, mask_fill)[0]/255.
    # Drwa stuff for debug
    if(debug_plot):
        cnt_aux1, _ = cv2.findContours(mask_fill1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
        cnt_aux2, _ = cv2.findContours(mask_fill2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:] 
        img_plot = img_ROI.copy()
        cv2.drawContours(img_plot, [cnt_aux1[0], cnt_aux2[0]], -1, 150, 1) ## Get mask
        cv2.putText(img_plot,"Mean =" + str(round(mean_col_ii, 2)), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 150)
        plot_img((img_ROI, img_plot), nrow = 2, nameWindow = 'test1', NewWindow = True)
        0xFF & cv2.waitKey()
        cv2.destroyAllWindows()
    return(mean_col_ii)

#Plot regardless if 1 0 mask or contours
def plotSegmented(segmented, img):
    img_plot = img.copy()
    if(type(segmented)==list):
        drawAllCont(img_plot, segmented, -1, 255, 3) #cv2.drawContours(img_plot, segmented, -1, 255, 3)
    else:
        contours_new, _ = cv2.findContours(segmented.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
        drawAllCont(img_plot, contours_new, -1, 255, 3) #cv2.drawContours(img_plot, contours_new, -1, 255, 3)   
    plot_img(img_plot, hold_plot=True)
    return(img_plot)
    
    
def get_coords(binary_img):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    areas_all       = [contour_xy(i) for i in contours_raw]
    return(areas_all)    
        

def find_Contour_inside(cnt_i, cnt_search, ret_index = False):
    pos_xy = contour_xy(cnt_i)
#    qq = [cnt for cnt in cnt_search if cv2.pointPolygonTest(cnt, pos_xy, False)!=-1]
    indx_qq = [i for i in range(len(cnt_search)) if cv2.pointPolygonTest(cnt_search[i], pos_xy, False)!=-1]
    if len(indx_qq) == 0: 
        return_val = []
        indx_qq    = [[]] # So that an empty indx can be returned if ret_index = True
    else:
        return_val = cnt_search[indx_qq[0]]
    if ret_index:
        return_val = return_val, indx_qq[0]
    return return_val
        

def getMaximumInts(binary_img, img):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    max_Ints_all       = [contour_max_Area(i, img) for i in contours_raw]
    return(max_Ints_all)    


def getMeanInts(binary_img, img):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    max_Ints_all       = [contour_mean_Area(i, img) for i in contours_raw]
    return(max_Ints_all)    


def getPercentileInts(binary_img, img, perc):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    max_Ints_all       = [contour_Percentile_Area(i, img, perc) for i in contours_raw]
    return(max_Ints_all)    


def getfracAreaNucl(binary_img, nucl_bin_img):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    areas_all       = [contour_mean_Area(i, nucl_bin_img) for i in contours_raw]
    return(areas_all)    

def getAreas(binary_img):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    areas_all       = [contour_Area(i) for i in contours_raw]
    return(areas_all)    

def filterSmallArea_outer(binary_img, thresh_size):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    validCont       = [i for i in contours_raw if contour_Area(i) > thresh_size]
    img_new         = np.zeros(binary_img.shape, dtype = np.uint8)
    drawAllCont(img_new, validCont, -1, 255, -1) #cv2.drawContours(img_new, validCont, -1, 255, -1)
    return(img_new)
    
def filterLargeArea_outer(binary_img, thresh_size):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]     
    validCont       = [i for i in contours_raw if contour_Area(i) < thresh_size]
    img_new         = np.zeros(binary_img.shape, dtype = np.uint8)
    drawAllCont(img_new, validCont, -1, 255, -1) #cv2.drawContours(img_new, validCont, -1, 255, -1)
    return(img_new)

def filterStains(binary_img, img_for_mean, thresh_size):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]  
    validCont       = [i for i in contours_raw if contour_mean_Area(i, img_for_mean) < thresh_size]
    img_new         = np.zeros(binary_img.shape, dtype = np.uint8)
    drawAllCont(img_new, validCont, -1, 255, -1) #cv2.drawContours(img_new, validCont, -1, 255, -1)
    return(img_new)


def contour_xy(cnt, reverse = False):
    m_ij   = cv2.moments(cnt)
    pos_xy = (int(m_ij['m10']/m_ij['m00']), int(m_ij['m01']/m_ij['m00']))
    if reverse:
        pos_xy = (pos_xy[1],pos_xy[0])
    return(pos_xy)    

def contour_Area(cnt):
    if len(cnt) == 1: return 1        
    return(cv2.contourArea(cnt))

def contour_solidity(cnt):
    # Get mean colour of object
    area      = cv2.contourArea(cnt)
    hull      = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity  = float(area)/hull_area
    return(solidity)

def contour_sum_Area(cnt, img):
    # Get mean colour of object
    roi           = cv2.boundingRect(cnt)
    Start_ij_ROI  = np.array(roi)[0:2] # get x,y of bounding box
    cnt_roi       = cnt - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill     = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill, [cnt_roi], 0, 255, -1) ## Get mask    
    sum_col_ii   = cv2.sumElems(cv2.bitwise_and(img_ROI, mask_fill))[0]
    return(sum_col_ii)

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

def contour_max_Area(cnt, img):
    # Get max colour of object
    roi          = cv2.boundingRect(cnt)
    Start_ij_ROI = np.array(roi)[0:2] # get x,y of bounding box
    cnt_roi      = cnt - Start_ij_ROI # change coords to start from x,y
    img_ROI      = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill, [cnt_roi], 0, 255, -1) ## Get mask    
    max_val_ii   = np.max(cv2.bitwise_and(img_ROI, mask_fill))
    return(max_val_ii)

def contour_Percentile_Area(cnt, img, perc):
    # Get max colour of object
    roi          = cv2.boundingRect(cnt)
    Start_ij_ROI = np.array(roi)[0:2] # get x,y of bounding box
    cnt_roi      = cnt - Start_ij_ROI # change coords to start from x,y
    img_ROI      = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill    = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill, [cnt_roi], 0, 255, -1) ## Get mask    
    max_val_ii   = np.percentile(cv2.bitwise_and(img_ROI, mask_fill), perc)
    return(max_val_ii)
  
def filterSmallArea(binary_img, thresh_size):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) [-2:]  
    img_new         = binary_img.copy()
    for cnt_i in contours_raw:
        if contour_Area(cnt_i) < thresh_size:
                cv2.drawContours(img_new, [cnt_i], -1, 0, -1)
    return(img_new)

def contour_var_Area(cnt, img):
    # Get mean colour of object
    roi           = cv2.boundingRect(cnt)
    Start_ij_ROI  = np.array(roi)[0:2] # get x,y of bounding box
    cnt_roi       = cnt - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill     = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill, [cnt_roi], 0, 1, -1) ## Get mask    
    var_col_ii    = np.var(cv2.bitwise_and(img_ROI, mask_fill).ravel())
#    var_col_ii    = np.var(img_ROI[mask_fill.ravel()].ravel())
    return(var_col_ii)

def contour_entropy_Area(cnt, img):
    # Get mean colour of object
    roi           = cv2.boundingRect(cnt)
    Start_ij_ROI  = np.array(roi)[0:2] # get x,y of bounding box
    cnt_roi       = cnt - Start_ij_ROI # change coords to start from x,y
    img_ROI       = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_fill     = np.zeros(img_ROI.shape[0:2], np.uint8)
    cv2.drawContours(mask_fill, [cnt_roi], 0, 1, -1) ## Get mask    
    entropy_ii    = claudes_entropy(img_ROI, mask_fill)
    return(entropy_ii)

## from http://stackoverflow.com/questions/16647116/faster-way-to-analyze-each-sub-window-in-an-image
def claudes_entropy(img, mask):
    hist = cv2.calcHist([img],[0],mask,[256],[0,256])
    hist = hist.ravel()/hist.sum()
    logs = np.log2(hist+0.00001)
    entropy = -1 * (hist*logs).sum()
    return entropy  


def filterContains(binary_img, img_for_mean, thresh_size):
    contours_raw, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [-2:] 
    validCont       = [i for i in contours_raw if contour_mean_Area(i, img_for_mean) > thresh_size]
    img_new         = np.zeros(binary_img.shape, dtype = np.uint8)
    drawAllCont(img_new, validCont, -1, 255, -1) #cv2.drawContours(img_new, validCont, -1, 255, -1)
    return(img_new)

