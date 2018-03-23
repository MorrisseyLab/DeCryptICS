# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:04:30 2016

@author: edward
"""
import numpy as np
import cv2

## Clone list has info from all tiles and as such has entries that give x0,y0 
## of the tile, this has to be added to all x,y of clones below it
def parseCloneList(cluster_clones_all, list_order, maxSize): 
    num_clones = len(cluster_clones_all)
    if maxSize == -1: maxSize = num_clones
    list_out = []
    for ii in range(num_clones):
        line_i   = cluster_clones_all[ii]
        ## Check if new offset
        if len(line_i) == 2:
            x_offset = int(line_i[0]); y_offset = int(line_i[1])
        else:
            ## x, y , clustNum, '[f1,..,fn]'
            x                    = x_offset  + int(line_i[0])
            y                    = y_offset  + int(line_i[1])
            list_out.append((x, y, line_i[3]))            
    ## Reorder -------------
    final_list_out = []
    for ii in range(len(list_out)):
        index_use = list_order[ii]
        line_i   = list_out[index_use]
        ## x, y , clustNum, '[f1,..,fn]'
        x                    = str(int(line_i[0]))
        y                    = str(int(line_i[1]))
        clust_num            = str(ii + 1)
        final_list_out.append(",".join([x, y, clust_num, "\'" + line_i[2] + "\'"]))
    # Truncate to max size if it was given
    final_list_out = final_list_out[0:maxSize]    
    return final_list_out



# Mat with [Clonefrac, maxIntensity, clust_size]
def makeScoringMat(img_all_clones):
    num_clones      = len(img_all_clones)
    clone_score_mat = np.zeros((num_clones, 3))
    for i in range(num_clones):
        cluster_size         = img_all_clones[i][3].count(",") + 1
        clone_score_mat[i,:] = [int(img_all_clones[i][1]), img_all_clones[i][2], cluster_size]
    # Normalise max intensity
    clone_score_mat[:,0] = clone_score_mat[:,0]/(1.*np.max(clone_score_mat[:,0]))
    return(clone_score_mat)


def annotateImage(img_all_clones, indx_i, ii):
#    print ii, indx_i
    clone_i    = img_all_clones[indx_i][0].copy()
    clust_info = img_all_clones[indx_i][3]
    x_start    = int(clone_i.shape[1]/2)
    str_write  = str(ii) + "- " + clust_info
    cv2.putText(clone_i, str_write, (x_start,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
    return clone_i

def orderAndAnnotateImage(img_all_clones, order_clone, maxSize):
    size_use        = np.min([maxSize, len(order_clone)])
    order_clone_use = order_clone[0:size_use]
    imgs_ordered    = [annotateImage(img_all_clones, indx_i, ii + 1) for ii, indx_i in zip(range(size_use), 
                                                                                      order_clone_use)]
    return np.vstack(imgs_ordered)

def saveCloneInfo(file_save_txt, cluster_clones_all, list_order, maxSize):
    # Reformat clone list
    new_clone_list = parseCloneList(cluster_clones_all, list_order, maxSize)
    # make_write_string ===========================================================
    string_out_clones = '\n'.join(new_clone_list)
    f = open(file_save_txt, "w"); f.write(string_out_clones);f.close()        
    

def orderClones_Annotate_WriteImg(folder_name, img_all_clones, cluster_clones_all, frac_tot_crypts_wclones, maxSize = 200):  
    file_save_txt = folder_name + '/cluster_clones_all.csv'
    file_save_img = folder_name + '/all_clones.jpg'    
    num_clones    = len(img_all_clones)
    
    # If MPAS hom
    if frac_tot_crypts_wclones > 0.25:
        saveCloneInfo(file_save_txt, cluster_clones_all, list_order = range(num_clones), maxSize = -1)
        return(0)  
        
    if frac_tot_crypts_wclones == 0:
        file_save_txt = folder_name + '/cluster_clones_all.csv'
        f = open(file_save_txt, "w"); f.write("(0,0)");f.close()        
        return(0)          
    ##  Re-order by score and truncate 
    # Make scoring mat
    score_mat     = makeScoringMat(img_all_clones)
    # Calculate single score from mat   
    score_weights = np.array([1, 1, 0.02])/2.02 # Patch-size should have a small effect (0.02)
    score_clone   = np.dot(score_mat, score_weights)     
    # Reorder and renumber list annotate images
    order_clone            = np.argsort(score_clone)[::-1] # index  stuff reverses order
    clone_images_reordered = orderAndAnnotateImage(img_all_clones, order_clone, maxSize)        
    cv2.imwrite(file_save_img, clone_images_reordered)
    saveCloneInfo(file_save_txt, cluster_clones_all, list_order = order_clone, maxSize = maxSize)
    
