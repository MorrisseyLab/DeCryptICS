#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:43:46 2018

@author: doran
"""
import numpy as np
from matplotlib import pyplot as plt
#import warnings
# D.K. these imports don't work
#from .GibbsDistributionClass.GibbsNormalClass import Gaussian_Component_Gibbs
#from .GibbsDistributionClass.GibbsExponentialClass import Exponential_Component_Gibbs
# 


## Yup, I do a lot of R coding ...
def cbind(list_to_bind):
    xx_all = np.concatenate(list_to_bind)  
    xx_all = xx_all.reshape((len(list_to_bind[0]),len(list_to_bind)), order='F')
    return xx_all

def rbind(list_to_bind):
    xx_all = np.concatenate(list_to_bind )
    xx_all = xx_all.reshape((len(list_to_bind), len(list_to_bind[0])), order='C')
    return xx_all
    


def predictVals2(means_all, data_1, data_2, Area):
#    means_all = np.mean(mcmc_out,0)
    [mean_x_c1_d1, prec_x_c1_d1, mean_x_c1_d2,  prec_x_c1_d2,
     mean_x_c1_d3, prec_x_c1_d3, mean_x_c2_d1,  prec_x_c2_d1,
     mean_x_c2_d2, prec_x_c2_d2, lamb_x_c2_d3,         alpha] = means_all
    # fix parameters to get loglik
    # D.K. c1 and c2 are true and false (classifications), d1,d2,d3 are data types
    Loglik_c1_d1 = getGaussLike(data_1, mean_x_c1_d1, prec_x_c1_d1, Funnel = None) # Halo
    Loglik_c1_d2 = getGaussLike(data_2, mean_x_c1_d2, prec_x_c1_d2, Funnel = Area) # Nuclear #, num_sd_penOutliers = +5.
    Loglik_c1_d3 = getGaussLike(Area,   mean_x_c1_d3, prec_x_c1_d3, Funnel = None) # Area
    Loglik_c2_d1 = getGaussLike(data_1, mean_x_c2_d1, prec_x_c2_d1, Funnel = None) # Halo #, num_sd_penOutliers = -2.
    Loglik_c2_d2 = getGaussLike(data_2, mean_x_c2_d2, prec_x_c2_d2, Funnel = Area) # Nuclear      
    Loglik_c2_d3 = getExpLike(Area, lamb_x_c2_d3)                                # Area

    ## Make Loglik_c2_d3 = Loglik_c1_d3 if  area > mu +2sd, so that it won't contribute to total lik
    indx_large = Area > (mean_x_c1_d3 + 2./np.sqrt(prec_x_c1_d3))
    Loglik_c2_d3[indx_large] = Loglik_c1_d3[indx_large]
    
    log_lik1 = Loglik_c1_d1 + Loglik_c1_d2 + Loglik_c1_d3
    log_lik2 = Loglik_c2_d1 + Loglik_c2_d2 + Loglik_c2_d3
    
    
    
    probs_full = getProbs(log_lik1, log_lik2, alpha)
    probs_full_correct = correctProbsOutliers(probs_full, data_2, mean_x_c1_d2, prec_x_c1_d2*Area, 5)
    probs_full_correct = correctProbsOutliers(probs_full, data_1, mean_x_c1_d1, prec_x_c1_d1, -5)
    
    probs_d1   = getProbs(Loglik_c1_d1, Loglik_c2_d1, alpha)
    probs_d2   = getProbs(Loglik_c1_d2, Loglik_c2_d2, alpha)
    probs_d3   = getProbs(Loglik_c1_d3, Loglik_c2_d3, alpha)    
    probs_all  = rbind((probs_full_correct, probs_d1, probs_d2, probs_d3))
    return probs_full_correct, probs_all


def correctProbsOutliers(probs_full, x_vec, mu, lamb, num_sd_penOutliers):
    if num_sd_penOutliers is not None:
        if num_sd_penOutliers<0:
            num_sd_penOutliers         = np.abs(num_sd_penOutliers)        
            limit_x                    = mu - num_sd_penOutliers/np.sqrt(lamb)
            probs_full[x_vec<limit_x] = 0.
        else:
            limit_x                    = mu + num_sd_penOutliers/np.sqrt(lamb)            
            probs_full[x_vec>limit_x] = 0.
    return probs_full

def getGaussLike(x_vec, mu, lamb, Funnel, num_sd_penOutliers = None): 
    if Funnel is None: Funnel = np.ones(len(x_vec), np.double)
    lamb_A      = lamb*Funnel    
    const_val   = -0.5*np.log(2.*np.pi)
    log_lik_vec = const_val + 0.5*np.log(lamb_A) - 0.5*lamb_A*(x_vec - mu)**2
    ## If active, heavily penalise any data a certain number of sd above or below mu
    log_lik_vec = penaliseOutliers(log_lik_vec, x_vec, mu, lamb_A, num_sd_penOutliers)
    return log_lik_vec

def penaliseOutliers(log_lik_vec, x_vec, mu, lamb, num_sd_penOutliers):
    if num_sd_penOutliers is not None:
        if num_sd_penOutliers<0:
            num_sd_penOutliers         = np.abs(num_sd_penOutliers)        
            limit_x                    = mu - num_sd_penOutliers/np.sqrt(lamb)
            log_lik_vec[x_vec<limit_x] = 50.
        else:
            limit_x                    = mu + num_sd_penOutliers/np.sqrt(lamb)            
            log_lik_vec[x_vec>limit_x] = -1e100
    return log_lik_vec


def plot_MeanAnd2SD(mean_i, prec_i, area_x, isFunnel, color = "k"):        
    if isFunnel: 
        area2 = area_x
    else:
        area2 = np.ones(len(area_x))
    y_max = mean_i + 2./np.sqrt(area2*prec_i)
    y_min = mean_i - 2./np.sqrt(area2*prec_i)
    plt.plot(area_x, y_max, color = color, linestyle = '--', linewidth=2)
    plt.plot(area_x, y_min, color = color, linestyle = '--', linewidth=2)

def scatterWithCols(xx, yy, cols, indx_vals):
    for class_i in [0,1]:
        indx_use = (class_i == indx_vals)
        plt.scatter(xx[indx_use], yy[indx_use], color = cols[class_i])
    
def plotSizeDistribution(mean_x_c1_d3, prec_x_c1_d3, lamb_x_c2_d3, alpha, Area):
    area_vec = np.arange(0, np.max(Area), np.max(Area)/1000)
    y_exp    = (1.-alpha)*np.exp(getExpLike(area_vec, lamb_x_c2_d3))
    y_gauss  = alpha*np.exp(getGaussLike(area_vec, mean_x_c1_d3, prec_x_c1_d3, Funnel = None))
    plt.plot(area_vec, y_exp, color = "r", linestyle = '--', linewidth=2)
    plt.plot(area_vec, y_gauss, color = "b", linestyle = '--', linewidth=2)

def getExpLike(data_x, Explambda): #mu, lamb, A_vec, x_vec
    log_lik_vec = np.log(Explambda) - Explambda*data_x
    return log_lik_vec


def getProbs(log_lik1, log_lik2, alpha):
#    exp_neg_lik = np.exp(log_lik1-log_lik2)
#    probs_1 = exp_neg_lik*alpha/(exp_neg_lik*alpha + (1.- alpha))
    exp_lik = np.exp(log_lik2-log_lik1)
    probs_1 = alpha/(alpha + (1.- alpha)*exp_lik)
    return probs_1


def plotMAP_fit2(means_all, data_1, data_2, Area, indx_cluster):
    [mean_x_c1_d1, prec_x_c1_d1, mean_x_c1_d2,  prec_x_c1_d2,
     mean_x_c1_d3, prec_x_c1_d3, mean_x_c2_d1,  prec_x_c2_d1,
     mean_x_c2_d2, prec_x_c2_d2, lamb_x_c2_d3,         alpha] = means_all

    area_x = np.arange(start = 1, stop = np.max(Area), step = 10)   
    plt.subplot(231)
    scatterWithCols(Area, data_1, ["r", "b"], indx_cluster) #plt.scatter(Area, data_1)
    plot_MeanAnd2SD(mean_x_c1_d1, prec_x_c1_d1, area_x, isFunnel = False, color = "b")
    plot_MeanAnd2SD(mean_x_c2_d1, prec_x_c2_d1, area_x, isFunnel = False, color = "r")
    plt.ylim((np.min(data_1), np.max(data_1)));plt.xlim((0, np.max(Area)))
    plt.subplot(232)
    scatterWithCols(Area, data_2, ["r", "b"], indx_cluster) #plt.scatter(Area, data_2)
    plot_MeanAnd2SD(mean_x_c1_d2, prec_x_c1_d2, area_x, isFunnel = True, color = "b")
    plot_MeanAnd2SD(mean_x_c2_d2, prec_x_c2_d2, area_x, isFunnel = True, color = "r")
    plt.ylim((np.min(data_2), np.max(data_2)));plt.xlim((0, np.max(Area)))
    plt.subplot(233)
    scatterWithCols(data_1, data_2, ["r", "b"], indx_cluster) # plt.scatter(data_1, data_2)
    plt.ylim((0, 1));plt.xlim((0, 1))
    plt.subplot(234)
    plt.hist(Area, 50, normed = True, color = "lightblue")   
    plotSizeDistribution(mean_x_c1_d3, prec_x_c1_d3, lamb_x_c2_d3, alpha, Area)
    try:
        plt.subplot(235)
        plt.hist(Area[indx_cluster], 30 , normed = True, color = "lightblue")
        plt.plot(area_x, np.exp(getGaussLike(area_x, mean_x_c1_d3, prec_x_c1_d3, Funnel = None)), color = "b", linestyle = '--', linewidth=2)
        plt.subplot(236)
        plt.hist(Area[np.logical_not(indx_cluster)], 50, normed = True, color = "lightblue")
        plt.plot(area_x, np.exp(getExpLike(area_x, lamb_x_c2_d3)), color = "r", linestyle = '--', linewidth=2)
    except:
        print("Only one cluster")
    plt.show()



run_i  = qq_both[3]
mean_i = run_i[5]
#mean_i[11] = 0.5 # artificially set alpha low (not needed really)
#mean_i = np.mean(mcmc_out, 0)
feat_cnt_nuc_m = getAllFeatures(run_i[1], nuclei_ch_raw, backgrd, smallBlur_img_nuc)
#feat_cnt_nuc_m = feat_cnt_nuc2
probs_full, probs_all = predictVals2(mean_i, feat_cnt_nuc_m.allHalo, feat_cnt_nuc_m.allMeanNucl, feat_cnt_nuc_m.allSizes-np.min(feat_cnt_nuc_m.allSizes)+1)
plotMAP_fit2(mean_i, feat_cnt_nuc_m.allHalo, feat_cnt_nuc_m.allMeanNucl, feat_cnt_nuc_m.allSizes-np.min(feat_cnt_nuc_m.allSizes)+1,probs_full>0.5)

cc = cbind((feat_cnt_nuc_m.allHalo, feat_cnt_nuc_m.allMeanNucl, feat_cnt_nuc_m.allSizes, probs_all[0,:], probs_all[1,:], probs_all[2,:], probs_all[3,:]))



