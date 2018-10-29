# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:48:24 2015

@author: edward
"""
from MiscFunctions import getROI_img_osl
import openslide as osl
import cv2
import numpy as np
import pickle, os


#obj_svs.roi1
class getROI_svs:
    def __init__(self, svs_filename, minSize = 50, get_roi_plot = True):    
        self.drawing      = False # true if mouse is pressed
        self.svs_filename = svs_filename
        slide             = osl.OpenSlide(svs_filename)
        ## Find level of reasonable size to plot
        self.dims_slides   = slide.level_dimensions #np.asarray(slide.level_dimensions)
        smallImage         = len(self.dims_slides) - 1 
        self.roi_full_thmb = [(0, 0), slide.level_dimensions[smallImage]] ## Start wi
        self.roi1          = [(0, 0), slide.level_dimensions[smallImage]] ## Start wi       
        self.scalingVal    = slide.level_downsamples[smallImage]
        self.img_zoom      = getROI_img_osl(svs_filename, (0,0), slide.level_dimensions[smallImage], level = smallImage)
#        self.img_zoom      = slide.read_region((0,0), smallImage, slide.level_dimensions[smallImage])
#        self.img_zoom      = cv2.cvtColor(np.asarray(self.img_zoom)[:,:,0:3], cv2.COLOR_RGB2BGR)
        self.chosenROI = []
        if (get_roi_plot):
            self.chosenROI     = self.getROI('Zoom', 5, minSize)
        
    def getROI_img(self):
        ## Zoom in and choose colours
#        ROI_zoom      = self.getROI('Zoom', 5, minSize)
        start_indx    = (int(self.chosenROI[0][0]*self.scalingVal), int(self.chosenROI[0][1]*self.scalingVal))
        delta_s       = (int(self.chosenROI[1][0]*self.scalingVal) - start_indx[0], int(self.chosenROI[1][1]*self.scalingVal) - start_indx[1])
        img_big       = getROI_img_osl(self.svs_filename, start_indx, delta_s)
        return(img_big)
        
    def getROI(self, win_name, size_line, minSize):
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) #WINDOW_AUTOSIZE)#
        cv2.setMouseCallback(win_name, self.drawRect, [win_name, size_line, minSize])
        cv2.imshow(win_name, self.img_zoom)
        k = cv2.waitKey() & 0xFF
        cv2.destroyWindow(win_name)
        ROI_chosen = self.correct_ROI_order(self.roi1)
        self.chosenROI = ROI_chosen
        return(ROI_chosen)

    def correct_ROI_order(self, roi):
        x1_x2 = sorted([roi[0][0], roi[1][0]])
        y1_y2 = sorted([roi[0][1], roi[1][1]])
        roi = ((x1_x2[0],y1_y2[0]),(x1_x2[1],y1_y2[1]))
        return(roi)
        
    # mouse callback function
    def drawRect(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi1[0] = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.roi1[1] = x,y
                    
                img_plot = self.img_zoom.copy()
                cv2.rectangle(img_plot,self.roi1[0], self.roi1[1], (255,   0,   0), param[1])
                cv2.imshow(param[0], img_plot)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi1[1] = x,y
            if abs(self.roi1[0][0] - self.roi1[1][0]) < 10 or abs(self.roi1[0][1] - self.roi1[1][1]) < 10:
                self.roi1 = [(self.roi1[0][0]-param[2],self.roi1[0][1]-param[2]),(self.roi1[1][0]+param[2],self.roi1[1][1]+param[2])]
                img_plot = self.img_zoom.copy()
                cv2.rectangle(img_plot,self.roi1[0], self.roi1[1], (255,   0,   0), param[1])
                cv2.imshow(param[0], img_plot)


