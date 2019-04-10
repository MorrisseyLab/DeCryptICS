#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:24:35 2018

@author: doran
"""
from scipy.linalg import eig
import numpy as np
from MiscFunctions import transform_OD
import math

def pack_rgb(img):
   rgb_packed = []
   for i in range(img.shape[0]):
      for j in range(img.shape[1]):
         rgb_packed.append((img[i,j,0], img[i,j,1], img[i,j,2]))
   return rgb_packed

def covariance(x, y):		
   n = len(x)
   if not (n == len(y)):
	   print("Cannot compute covariance - array lengths are not the same")
	   return 0
   xMean = 0
   for v in x: xMean += float(v)/n
   yMean = 0
   for v in y: yMean += float(v)/n
   result = 0
   for i in range(n):
      xDev = x[i] - xMean
      yDev = y[i] - yMean
      result += xDev * yDev / n
   return result

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def estimateStains(img, deconv_mat_ref):
   ''' Adapted from the Java implementation in QuPath, by Pete Bankhead:
   https://github.com/qupath/qupath/blob/master/qupath-core-processing-awt/src/main/java/qupath/lib/algorithms/color/EstimateStainVectors.java '''
   # play with these 
   maxStain = 1.5 # 1 default
   minStain = 0.01 # 0.05 default
   ignorePercentage = 1.
   alpha = ignorePercentage / 100.

   rgb = pack_rgb(img)
   ## Find optical density vectors
   timg = transform_OD(img)
   red = timg[:,:,0].ravel()
   green = timg[:,:,1].ravel()
   blue = timg[:,:,2].ravel()

   sqrt3 = 1./math.sqrt(3)
   grayThreshold = math.cos(0.15)

   ## Loop through and discard pixels that are too faintly or densely stained
   keepCount = 0;
   maxStainSq = maxStain*maxStain;
   for i in range(len(red)):
      r = red[i]
      g = green[i]
      b = blue[i]
      magSquared = r*r + g*g + b*b
      if (magSquared > maxStainSq or r < minStain or g < minStain or b < minStain):
         continue
      ## Update the arrays
      red[keepCount] = r
      green[keepCount] = g
      blue[keepCount] = b
      rgb[keepCount] = rgb[i]
      keepCount += 1

   if (keepCount <= 1):
      print("Not enough pixels remain after applying stain thresholds!")

   ## Trim the arrays
   if (keepCount < len(red)):
      red = red[:keepCount]
      green = green[:keepCount]
      blue = blue[:keepCount]
      rgb = rgb[:keepCount]

   cov = np.zeros([3,3])
   cov[0,0] = covariance(red, red)
   cov[1,1] = covariance(green, green)
   cov[2,2] = covariance(blue, blue)
   cov[0,1] = covariance(red, green)
   cov[0,2] = covariance(red, blue)
   cov[1,2] = covariance(green, blue)
   cov[2,1] = cov[1,2]
   cov[2,0] = cov[0,2]
   cov[1,0] = cov[0,1]

   eigen = np.linalg.eig(cov)
   eigenValues = eigen[0]
   eigenOrder = np.argsort(eigen[0])
   eigen1 = eigen[1][eigenOrder[2],:]
   eigen2 = eigen[1][eigenOrder[1],:]

   ## Calculate polar angles
   phi = np.zeros([keepCount])
   for i in range(keepCount):
      r = red[i]
      g = green[i]
      b = blue[i]
      phi[i] = np.arctan2(r*eigen1[0] + g*eigen1[1] + b*eigen1[2], r*eigen2[0] + g*eigen2[1] + b*eigen2[2])

   ## Select stain vectors from data		
   inds = np.argsort(phi)
   ind1 = inds[int(alpha * keepCount + .5)];
   ind2 = inds[int((1 - alpha) * keepCount + .5)];

   ## Create new stain vectors
   s1 = unit_vector(np.array([red[ind1], green[ind1], blue[ind1]]))
   s2 = unit_vector(np.array([red[ind2], green[ind2], blue[ind2]]))
   s3 = unit_vector(np.cross(s1,s2))
	
   ## Check we've got the closest match - if not, switch the order
   ## (requires an approximate reference deconvolution matrix)
   dcmi = np.linalg.inv(deconv_mat_ref)
   dcm = dcmi.T
   stain_orig_1 = dcm[0,:]
   stain_orig_2 = dcm[1,:]
   angle12 = angle_between(s1, stain_orig_2)
   angle11 = angle_between(s1, stain_orig_1)
   angle22 = angle_between(s2, stain_orig_2)
   angle21 = angle_between(s2, stain_orig_1)
   if (min(angle12, angle21) < min(angle11, angle22)):
      s1 = unit_vector(np.array([red[ind2], green[ind2], blue[ind2]]))
      s2 = unit_vector(np.array([red[ind1], green[ind1], blue[ind1]]))
      
   C = np.array([s1,s2,s3])
   D = np.linalg.inv(C.T)
   #imgd = cv2.transform(timg, D)
   return D
	

