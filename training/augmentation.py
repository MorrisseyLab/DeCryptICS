import cv2
import numpy as np
import random
from skimage.transform import warp, AffineTransform

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

def random_flip(image, mask_list, u=0.25):
   u01 = np.random.random()
   rem = (1.-u)/3.
   if u01 > u:
      if u01 < (u+rem): # horizontal flip
         image = cv2.flip(image, 1)
         for m in range(len(mask_list)):
            mask_list[m] = cv2.flip(mask_list[m], 1)
      elif u01 < (u+2*rem): # vertical flip
         image = cv2.flip(image, 0)
         for m in range(len(mask_list)):
            mask_list[m] = cv2.flip(mask_list[m], 0)
      else: # flip on both axes
         image = cv2.flip(image, -1)
         for m in range(len(mask_list)):
            mask_list[m] = cv2.flip(mask_list[m], -1) 
   return image, mask_list
    
def fix_mask(mask):
   mask[mask < 122] = 0
   mask[mask >= 122] = 255
   return mask
    
def fix_masks(mask_list):
   for m in range(len(mask_list)):
      mask_list[m] = fix_mask(mask_list[m])
   return mask_list

def random_affine(image, mask_list, u=0.5, scale_lims = (0.7,1.3), 
                  rot_lims = (-np.pi, np.pi)):
   u01 = np.random.random()
   if u01 > u:
      image = image.copy()
      h, w = image.shape[:2]
      tform = AffineTransform(scale=random.uniform(scale_lims[0], scale_lims[1]),
                              rotation=random.uniform(rot_lims[0], rot_lims[1]),
                              shear=0, translation=(0,0))
      angle = tform.rotation * (-1)
      translate = tuple(tform.translation)
      shear = tform.shear * (-1)
      scale = tform.scale[0]
      midxy = (w/2., h/2.)
      # alter rotation axis from origin to centre of image
      cx_f = midxy[0]
      cy_f = midxy[1]
      rot_mat = tform.params.copy()
      # extract pure rotation
      rot_mat[:2, 2] = np.array([0,0])
      rot_mat[0, 1] = -rot_mat[1, 0]
      rot_mat[1, 1] = rot_mat[0, 0]   
      extra_translate = (np.around(np.dot(rot_mat, np.array([cx_f, cy_f, 1])))).astype(np.int32)
      tform.params[:2,2] = tform.params[:2,2] - extra_translate[:2] + np.array([cx_f, cy_f])

      image = cv2.warpPerspective(image, tform.params, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))  
      for m in range(len(mask_list)):
         mask_list[m] = cv2.warpPerspective(mask_list[m], tform.params, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))
   return image, mask_list
   
def random_perspective(image, mask_list, u=0.5, scale = 0.025):
   u01 = np.random.random()
   if u01 > u:
      image = image.copy()
      h, w = image.shape[:2]
      ## choose 4 starting points in square around the centre
      ## in top-left, top-right, bottom-right, bottom-left order
      rect = np.zeros((4,2), dtype=np.float32)
      rect[0,0] = 1./3.; rect[0,1] = 1./3.
      rect[1,0] = 1./3.; rect[1,1] = 2./3.
      rect[2,0] = 2./3.; rect[2,1] = 1./3.
      rect[3,0] = 2./3.; rect[3,1] = 2./3.
      rect[:,0] = rect[:,0] * h
      rect[:,1] = rect[:,1] * w
      
      ## randomise destination points
      dst = np.zeros((4,2), dtype=np.float32)
      dst[0,:] = np.random.normal(loc = np.array([1./3., 1./3.]), scale = scale)
      dst[1,:] = np.random.normal(loc = np.array([1./3., 2./3.]), scale = scale)
      dst[2,:] = np.random.normal(loc = np.array([2./3., 1./3.]), scale = scale)
      dst[3,:] = np.random.normal(loc = np.array([2./3., 2./3.]), scale = scale)
      dst[:,0] = dst[:,0] * h
      dst[:,1] = dst[:,1] * w

      M = cv2.getPerspectiveTransform(rect, dst)     
      image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))
      for m in range(len(mask_list)):
         mask_list[m] = cv2.warpPerspective(mask_list[m], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))
   return image, mask_list   

def random_affine_each_elem(image_list, u=0.5, scale_lims = (0.7,1.3), 
                            rot_lims = (-np.pi, np.pi)):
   for m in range(len(image_list)):
      u01 = np.random.random()
      if u01 > u:
         image = image_list[m].copy()
         h, w = image.shape[:2]
         tform = AffineTransform(scale=random.uniform(scale_lims[0], scale_lims[1]),
                                 rotation=random.uniform(rot_lims[0], rot_lims[1]),
                                 shear=0, translation=(0,0))
         angle = tform.rotation * (-1)
         translate = tuple(tform.translation)
         shear = tform.shear * (-1)
         scale = tform.scale[0]
         midxy = (w/2., h/2.)
         # alter rotation axis from origin to centre of image
         cx_f = midxy[0]
         cy_f = midxy[1]
         rot_mat = tform.params.copy()
         # extract pure rotation
         rot_mat[:2, 2] = np.array([0,0])
         rot_mat[0, 1] = -rot_mat[1, 0]
         rot_mat[1, 1] = rot_mat[0, 0]   
         extra_translate = (np.around(np.dot(rot_mat, np.array([cx_f, cy_f, 1])))).astype(np.int32)
         tform.params[:2,2] = tform.params[:2,2] - extra_translate[:2] + np.array([cx_f, cy_f])

         image = cv2.warpPerspective(image, tform.params, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))
         image_list[m] = image
   return image_list

def random_flip_each_elem(image_list, u=0.5):
   for m in range(len(image_list)):
      u01 = np.random.random()
      if u01 > u:
         image = image_list[m].copy()
         rem = (1.-u)/3.
         if u01 < (u+rem): # horizontal flip
            image = cv2.flip(image, 1)
         elif u01 < (u+2*rem): # vertical flip
            image = cv2.flip(image, 0)
         else: # flip on both axes
            image = cv2.flip(image, -1)
         image_list[m] = image
   return image_list

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

#def random_shift_scale_rotate(image, mask_list,
#                  shift_limit=(-0.05, 0.05),
#                  scale_limit=(-0.1, 0.1),
#                  borderMode=cv2.BORDER_CONSTANT, u=0.5):
#   if np.random.random() < u:
#      height, width, channel = image.shape

#      angle = np.random.choice([90., -90., 180.])
#      scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
#      dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
#      dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

#      cc = np.math.cos(angle / 180. * np.math.pi)
#      ss = np.math.sin(angle / 180. * np.math.pi)
#      rotate_matrix = np.array([[cc, -ss], [ss, cc]])

#      box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
#      box1 = box0 - np.array([width / 2, height / 2])
#      box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

#      box0 = box0.astype(np.float32)
#      box1 = box1.astype(np.float32)
#      mat = cv2.getPerspectiveTransform(box0, box1)
#      image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
#                                  borderMode=borderMode, borderValue=(0, 0, 0,))
#      for m in range(len(mask_list)):
#         mask_list[m] = cv2.warpPerspective(mask_list[m], mat, (width, height), flags=cv2.INTER_LINEAR, 
#                                            borderMode=borderMode, borderValue=(0,0,0,))
#   return image, mask_list

#def affine_matrix(midpointxy, theta=0, scale=1, shear=0, xyt=(0,0), opencv=True):
#   # theta, shear angle in radians
#   if opencv==False:
#      RT = np.array([[scale*np.cos(theta), -scale*np.sin(theta + shear), xyt[0]],
#                     [scale*np.sin(theta), scale*np.cos(theta + shear) , xyt[1]],
#                     [0                  , 0                           , 1 ]])
#   else:
#      RT = np.array([[scale*np.cos(theta) , scale*np.sin(theta + shear), xyt[0]],
#                     [-scale*np.sin(theta), scale*np.cos(theta + shear), xyt[1]],
#                     [0                   , 0                          , 1 ]])
#   # alter rotation axis from origin to centre of image
#   cx_f = midpointxy[0]
#   cy_f = midpointxy[1]
#   rot_mat = RT.copy()
#   # extract pure rotation
#   rot_mat[:2, 2] = np.array([0,0])
#   rot_mat[0, 1] = -rot_mat[1, 0]
#   rot_mat[1, 1] = rot_mat[0, 0]   
#   extra_translate = (np.around(np.dot(rot_mat, np.array([cx_f, cy_f, 1])))).astype(np.int32)
#   RT[:2,2] = RT[:2,2] - extra_translate[:2] + np.array([cx_f, cy_f])
#   return RT
