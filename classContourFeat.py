# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:54:19 2015

@author: edward
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
#from cnt_Feature_Functions import *
from cnt_Feature_Functions import st_5, getCorrectionHalo, contour_Area
from cnt_Feature_Functions import contour_max_Halo, contour_solidity, contour_sum_Area
from cnt_Feature_Functions import contour_mean_Area, contour_MajorMinorAxis, contour_eccentricity
from cnt_Feature_Functions import contour_var_Area, contour_entropy_Area, calc_Halo_Gap


class getAllFeatures:
    def __init__(self, crypt_cnt, nuclei_ch_raw, background_img, smallBlur_img_nuc):
                        
        ## Get rid of contours with less than 5 points which will 
        ## crash elipse fitting (artefacts in the image can produce squares!!)
        if (np.any( np.asarray(list(map(len, crypt_cnt))) < 5)): 
            raise ValueError('A contour has fewer than 5 points')

        HaloAnalysis      = [    contour_max_Halo(i, nuclei_ch_raw) for i in crypt_cnt]
#        HaloGapCalc       = [ calc_Halo_Gap(h[1], nuclei_ch_raw) for h in HaloAnalysis]
        allHalo           = [                               h[0] for h in HaloAnalysis]
        allHaloGap        = [ calc_Halo_Gap(h[1], nuclei_ch_raw) for h in HaloAnalysis]
#        allHalototGap     = [                               j[1] for j in HaloGapCalc ]
        
        allSizes          = [                       contour_Area(i) for i in crypt_cnt]
#        allHalo           = [    contour_max_Halo(i, nuclei_ch_raw) for i in crypt_cnt]
        allSolid          = [                   contour_solidity(i) for i in crypt_cnt]
        allSumNucl        = [    contour_sum_Area(i, nuclei_ch_raw) for i in crypt_cnt]
        allMeanNucl       = [   contour_mean_Area(i, nuclei_ch_raw) for i in crypt_cnt]
        allMajorAxis      = [          contour_MajorMinorAxis(i)[0] for i in crypt_cnt]
        allMinorAxis      = [          contour_MajorMinorAxis(i)[1] for i in crypt_cnt]
        allEcc            = [               contour_eccentricity(i) for i in crypt_cnt]
        allVar            = [contour_var_Area(i, smallBlur_img_nuc) for i in crypt_cnt]
        allEntrop         = [contour_entropy_Area(i, smallBlur_img_nuc) for i in crypt_cnt]

        ## Correct halo close to border (U shape objects with no halo at the top)
        background_img    = cv2.morphologyEx( background_img, cv2.MORPH_DILATE,  st_5, iterations = 1)
        allHalo_corrected = [getCorrectionHalo(i, background_img, allHalo_i) for allHalo_i,i in zip(allHalo, crypt_cnt)]
        
        self.numContours       = len(allSizes)
        self.allSizes          = np.array(allSizes)
        self.allHalo_old       = np.array(allHalo)
        self.allHalo           = np.array(allHalo_corrected)
        self.allHaloGap        = np.array(allHaloGap)
#        self.allHalototGap     = np.array(allHalototGap)
        self.allSolid          = np.array(allSolid)
        self.allSumNucl        = np.array(allSumNucl)
        self.allMeanNucl       = np.array(allMeanNucl)
        self.allMajorAxis      = np.array(allMajorAxis)
        self.allMinorAxis      = np.array(allMinorAxis)
        self.allEcc            = np.array(allEcc)
        self.allVar            = np.array(allVar)
        self.allEntrop         = np.array(allEntrop)

    def plotFeatures(self, classVec = None):  
        if classVec is None: classVec = np.zeros(self.numContours)    
        classes_unique = np.unique(classVec)
        for class_i in classes_unique:
            indx_use = (classVec == class_i)
            plt.subplot(2, 4, 1)
            plt.plot(self.allHalo[indx_use], self.allMeanNucl[indx_use], "o")
            plt.title('Halo vs MeanNucl')
            plt.xlabel('Halo');plt.ylabel('MeanNucl')
            
            plt.subplot(2, 4, 2)
#            plt.plot(self.allSizes[indx_use], self.allEntrop[indx_use], "o")
#            plt.title('Area vs Entropy')
#            plt.xlabel('Area');plt.ylabel('Sum')
            plt.plot(self.allHaloGap[indx_use], self.allSizes[indx_use], "o") #allSumNucl
            plt.title('HaloGap vs Area')
            plt.xlabel('HaloGap');plt.ylabel('Area')
            
            plt.subplot(2, 4, 3)
            plt.plot(self.allSizes[indx_use], self.allMeanNucl[indx_use], "o") #allSumNucl
            plt.title('Area vs Mean')
            plt.xlabel('Area');plt.ylabel('Mean')
                       
            plt.subplot(2, 4, 4)
            plt.plot(self.allSizes[indx_use], 1-self.allHalo[indx_use], "o") #allSumNucl
            plt.title('Halo vs Area')
            plt.xlabel('Area');plt.ylabel('1-Halo')

            plt.subplot(2, 4, 5)
#            plt.plot(self.allSizes[indx_use], self.allVar[indx_use], "o") #allSumNucl
#            plt.title('Variance vs Size')
#            plt.xlabel('Area');plt.ylabel('Variance')
            plt.plot(self.allHalo[indx_use], self.allHaloGap[indx_use], "o") #allSumNucl
            plt.title('Halo vs HaloGap')
            plt.xlabel('Halo');plt.ylabel('Halo Gap')

            plt.subplot(2, 4, 6)
            plt.plot(self.allSizes[indx_use], self.allEcc[indx_use], "o") #allSumNucl
            plt.title('Area vs Eccentricity')
            plt.xlabel('Area');plt.ylabel('Eccentricity')

            plt.subplot(2, 4, 7)
            plt.plot(self.allHalo[indx_use], self.allSumNucl[indx_use], "o")
            plt.title('Halo vs SumNucl')
            plt.xlabel('Halo');plt.ylabel('SumNucl')
#            plt.plot(self.allHalototGap[indx_use], self.allHalomaxGap[indx_use], "o")
#            plt.title('HalototGap vs HalomaxGap')
#            plt.xlabel('HalototGap');plt.ylabel('HalomaxGap')

            plt.subplot(2, 4, 8)
            plt.plot(self.allSizes[indx_use], self.allSolid[indx_use], "o") #allSumNucl
            plt.title('Area vs Solid')
            plt.xlabel('Area');plt.ylabel('Solid')
            
        plt.show()


    def plot_histogram(self, x, bins_i):
        hist, bins_j = np.histogram(x, bins=bins_i)
        width      = 0.7 * (bins_j[1] - bins_j[0])
        center     = (bins_j[:-1] + bins_j[1:]) / 2
        plt.bar(center, hist, align='center', width=width)

    def plotHistograms(self, classVec = None):  
        if classVec is None: classVec = np.zeros(self.numContours)
        classes_unique = np.unique(classVec)
        num_classes    = len(classes_unique)
        for ii in range(num_classes):
#            print ii 
            class_i  = classes_unique[ii]
            indx_use = (classVec == class_i)

            plt.subplot(num_classes, 8, 1+ii) 
            self.plot_histogram(self.allSizes[indx_use], 30)
            plt.title('Area')
            
            plt.subplot(num_classes, 8, 2+ii)
            self.plot_histogram(self.allHalo[indx_use], 30)
            plt.title('Halo corrected')
            
            plt.subplot(num_classes, 8, 3+ii)
            self.plot_histogram(self.allSolid[indx_use],30)
            plt.title('Solidity')
    
            plt.subplot(num_classes, 8, 4+ii)
            self.plot_histogram(self.allSumNucl[indx_use], 30)
            plt.title('Sum Nuclei')
    
            plt.subplot(num_classes, 8, 5+ii)
            self.plot_histogram(self.allMeanNucl[indx_use], 30)
            plt.title('Mean Nuclei')
    
            plt.subplot(num_classes, 8, 6+ii)
            self.plot_histogram(self.allMajorAxis[indx_use], 30)
            plt.title('Major Axis')
    
            plt.subplot(num_classes, 8, 7+ii)
            self.plot_histogram(self.allMinorAxis[indx_use], 30)
            plt.title('Minor Axis')
    
            plt.subplot(num_classes, 8, 8+ii)
            self.plot_histogram(self.allEcc[indx_use],30)
            plt.title('Eccentricity')
        plt.show()
       
    def plotCorrectedHalo(self):  
        plt.subplot(1,1,1)
        plt.plot(self.allHalo_old, self.allHalo, "o")
        plt.title('allHalo vs allHalo_corrected')
        plt.xlabel('allHalo');plt.ylabel('allHalo_corrected')
        plt.show()

