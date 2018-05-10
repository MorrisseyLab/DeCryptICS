# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:54:05 2015

@author: edward
"""
from numpy.random import beta
import numpy as np
from matplotlib import pyplot as plt
from numba import jit, double, int_
from numba.types import Tuple
from random import random as runiform 
from math import exp
from sklearn.cluster import KMeans
#import warnings
# D.K. these imports don't work
from .GibbsDistributionClass.GibbsNormalClass import Gaussian_Component_Gibbs
from .GibbsDistributionClass.GibbsExponentialClass import Exponential_Component_Gibbs

# =============================================================================
# from .GibbsNormalClass import Gaussian_Component_Gibbs
# from .GibbsDistributionClass.GibbsExponentialClass import Exponential_Component_Gibbs
# from .GibbsDistributionClass.GibbsExponentialClass import Exponential_Component_Gibbs
# =============================================================================

## Silence overflow warnings
np.seterr(over='ignore')

def initGammas_kmeans(xx_filt):
    k2    = KMeans(n_clusters=2)
    fit_k = k2.fit(xx_filt[:,0:2])
    
    if fit_k.cluster_centers_[0,0] - fit_k.cluster_centers_[1,0] >0: 
        gammas = 1-fit_k.predict(xx_filt[:,0:2])
    else:
        gammas = fit_k.predict(xx_filt[:,0:2])
    gammas = gammas.astype(np.double, copy = False)
    return gammas

@jit("double[:](double[:], double[:], double)", nopython=True)
def updateGammas(log_lik1, log_lik2, alpha): 
    num_data   = len(log_lik1)
    new_gammas = np.zeros(num_data, dtype= double)
    for i in range(len(log_lik1)):
        probs_1    = alpha/(alpha + (1.- alpha)*exp(log_lik2[i]-log_lik1[i]))
        if runiform() < probs_1: new_gammas[i] = 1.
        # Else leave as is given that we've initialised to zero
    return new_gammas

@jit(nopython=True)
def updateAlpha(gamma_i, prior_a, prior_b):
    sum_gammas  = np.sum(gamma_i) #Sum_nb(gamma_i) #np.sum(gamma_i)
    new_alpha   = beta(prior_a + sum_gammas, prior_b + len(gamma_i) - sum_gammas)
    return new_alpha

#@jit('Tuple((double[:], double[:,:]))(double[:,:], double[:], int_, int_)') #, nopython=True)
@jit(nopython=True)
def GibbsMixFunnel2Clusters3d_2(xx_filt, gamma_i, num_iter, thin):
    x_halo    = xx_filt[:,0]
    y_nucl    = xx_filt[:,1] 
    Size_vec  = xx_filt[:,2]
    noFunnel  = np.ones(len(x_halo))
    prior_a = 2.
    prior_b = 0.01
        
    ## Initialse gaussian for gibbs
    ## Compnent 1, data 1:3
    gauss_c1_d1 = Gaussian_Component_Gibbs(x_halo, noFunnel, 0.8, 25., prior_a, prior_b, 0.9)    
    gauss_c1_d2 = Gaussian_Component_Gibbs(y_nucl, Size_vec, 0.02, 50., 9., 15., 0.01)   
    gauss_c1_d3 = Gaussian_Component_Gibbs(Size_vec, noFunnel, 15000., 4e-8, 0.2, 7e-6, 40000.)                                        

    ## Compnent 2, data 1:3
    gauss_c2_d1 = Gaussian_Component_Gibbs(x_halo, noFunnel, 0.5, 40., prior_a, prior_b, 0.3)                                                                              
    gauss_c2_d2 = Gaussian_Component_Gibbs(y_nucl, Size_vec, 0.1, 40., 2.5, 15., 0.7)                                   
    ## Could improve this with a gamma with shape fixed to 0.5
    exp_c2_d3 = Exponential_Component_Gibbs(Size_vec, 0.0625, 6.25e-10, 0.001)
    
    data_size  = len(gauss_c1_d1.x_vec_full)
    mcmc_out   = np.zeros((num_iter,12))
    alpha      = 0.5
    gamma_all  = np.zeros(data_size)
    
    for i in range(num_iter):
        for j in range(thin):
             not_gamma_i = 1. - gamma_i
             # Update parameters
             gauss_c1_d1.update_params(gamma_i)
             gauss_c1_d2.update_params(gamma_i)
             gauss_c1_d3.update_params(gamma_i)
             gauss_c2_d1.update_params(not_gamma_i)
             gauss_c2_d2.update_params(not_gamma_i)
             exp_c2_d3.update_params(not_gamma_i)
             # Calculate log likelihod
             log_like1 = gauss_c1_d1.logLike_allData() + gauss_c1_d2.logLike_allData() + gauss_c1_d3.logLike_allData() 
             log_like2 = gauss_c2_d1.logLike_allData() + gauss_c2_d2.logLike_allData() + exp_c2_d3.logLike_allData()
             # Sample new gammas
             gamma_i   = updateGammas(log_like1, log_like2, alpha)
             #if fix_gammas is not None: gamma_i = fix_gammas # For debugging
             # Sample mixture coefficient
             alpha     = updateAlpha(gamma_i, 2., 2.)                              
        gamma_all += gamma_i
        mcmc_out[i, :] = [gauss_c1_d1.mu, gauss_c1_d1.lamb, 
                         gauss_c1_d2.mu, gauss_c1_d2.lamb, 
                         gauss_c1_d3.mu, gauss_c1_d3.lamb, 
                         gauss_c2_d1.mu, gauss_c2_d1.lamb, 
                         gauss_c2_d2.mu, gauss_c2_d2.lamb, 
                         exp_c2_d3.Explambda, alpha]
    return gamma_all, mcmc_out

def plotChains(mcmc_out):
    plt.subplot(3,4,1)
    plt.plot(mcmc_out[:,0])
    plt.subplot(3,4,2)
    plt.plot(mcmc_out[:,1])
    plt.subplot(3,4,3)
    plt.plot(mcmc_out[:,2])
    plt.subplot(3,4,4)
    plt.plot(mcmc_out[:,3])
    plt.subplot(3,4,5)
    plt.plot(mcmc_out[:,4])
    plt.subplot(3,4,6)
    plt.plot(mcmc_out[:,5])
    plt.subplot(3,4,7)
    plt.plot(mcmc_out[:,6])
    plt.subplot(3,4,8)
    plt.plot(mcmc_out[:,7])
    plt.subplot(3,4,9)
    plt.plot(mcmc_out[:,8])
    plt.subplot(3,4,10)
    plt.plot(mcmc_out[:,9])
    plt.subplot(3,4,11)
    plt.plot(mcmc_out[:,10])
    plt.subplot(3,4,12)
    plt.plot(mcmc_out[:,11])
    plt.show()

def plotPosteriors(mcmc_out):
    plt.subplot(3,4,1)
    plt.hist(mcmc_out[:,0], 50)
    plt.title('Mean c1 d1')
    plt.subplot(3,4,2)
    plt.hist(mcmc_out[:,1], 50)
    plt.title('Precision c1 d1', color = 'r')
    plt.subplot(3,4,3)
    plt.hist(mcmc_out[:,2], 50)
    plt.title('Mean c1 d2')
    plt.subplot(3,4,4)
    plt.hist(mcmc_out[:,3], 50)
    plt.title('Precision c1 d2', color = 'r')
    plt.subplot(3,4,5)
    plt.hist(mcmc_out[:,4], 50)
    plt.title('Mean c1 d3')
    plt.subplot(3,4,6)
    plt.hist(mcmc_out[:,5], 50)
    plt.title('Precision c1 d3', color = 'r')    
    plt.subplot(3,4,7)
    plt.hist(mcmc_out[:,6], 50)
    plt.title('Mean c2 d1')
    plt.subplot(3,4,8)
    plt.hist(mcmc_out[:,7], 50)
    plt.title('Precision c2 d1', color = 'r')
    plt.subplot(3,4,9)
    plt.hist(mcmc_out[:,8], 50)
    plt.title('Mean c2 d2')
    plt.subplot(3,4,10)
    plt.hist(mcmc_out[:,9], 50)
    plt.title('Precision c2 d2', color = 'r')
    plt.subplot(3,4,11)
    plt.hist(mcmc_out[:,10], 50)
    plt.title('Precision c2 d3', color = 'r')
    plt.subplot(3,4,12)
    plt.hist(mcmc_out[:,11], 50)
    plt.title('Alpha')
    plt.show()

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

def plotMAP_fit(mcmc_out, data_1, data_2, Area, indx_cluster):
    means_all = np.mean(mcmc_out,0)
    [mean_x_c1_d1, prec_x_c1_d1, mean_x_c1_d2, prec_x_c1_d2,
     mean_x_c1_d3, prec_x_c1_d3, mean_x_c2_d1, prec_x_c2_d1,
     mean_x_c2_d2, prec_x_c2_d2, lamb_x_c2_d3,        alpha] = means_all
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

def penaliseOutliers(log_lik_vec, x_vec, mu, lamb, num_sd_penOutliers):
    if num_sd_penOutliers is not None:
        if num_sd_penOutliers<0:
            num_sd_penOutliers         = np.abs(num_sd_penOutliers)        
            limit_x                    = mu - num_sd_penOutliers/np.sqrt(lamb)
            log_lik_vec[x_vec<limit_x] = 999999.
        else:
            limit_x                    = mu + num_sd_penOutliers/np.sqrt(lamb)            
            log_lik_vec[x_vec>limit_x] = 999999.
    return log_lik_vec

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
    log_lik_vec = penaliseOutliers(log_lik_vec, x_vec, mu, lamb, num_sd_penOutliers)
    return log_lik_vec

def getExpLike(data_x, Explambda): #mu, lamb, A_vec, x_vec
    log_lik_vec = np.log(Explambda) - Explambda*data_x
    return log_lik_vec


def getProbs(log_lik1, log_lik2, alpha):
    probs_1 = alpha/(alpha + (1.- alpha)*np.exp(log_lik2-log_lik1))
    return probs_1

## Heavily penalise outliers for nuclear and halo
## Zero contribution from area for outliers => no contribution of mix model 
## when beyond distribution of gaussian
def predictVals(mcmc_out, data_1, data_2, Area):
    means_all = np.mean(mcmc_out,0)
    [mean_x_c1_d1, prec_x_c1_d1, mean_x_c1_d2,  prec_x_c1_d2,
     mean_x_c1_d3, prec_x_c1_d3, mean_x_c2_d1,  prec_x_c2_d1,
     mean_x_c2_d2, prec_x_c2_d2, lamb_x_c2_d3,         alpha] = means_all
    # fix parameters to get loglik
    # D.K. c1 and c2 are true and false (classifications), d1,d2,d3 are data types
    Loglik_c1_d1 = getGaussLike(data_1, mean_x_c1_d1, prec_x_c1_d1, Funnel = None) # Halo
    Loglik_c1_d2 = getGaussLike(data_2, mean_x_c1_d2, prec_x_c1_d2, Funnel = Area) # Nuclear
    Loglik_c1_d3 = getGaussLike(Area,   mean_x_c1_d3, prec_x_c1_d3, Funnel = None) # Area
    Loglik_c2_d1 = getGaussLike(data_1, mean_x_c2_d1, prec_x_c2_d1, Funnel = None)#, num_sd_penOutliers = -2.) # Halo
    Loglik_c2_d2 = getGaussLike(data_2, mean_x_c2_d2, prec_x_c2_d2, Funnel = Area)#, num_sd_penOutliers = +2.2) # Nuclear      
    Loglik_c2_d3 = getExpLike(Area, lamb_x_c2_d3)                                # Area

    ## Make Loglik_c2_d3 = Loglik_c1_d3 if  area > mu +2sd, so that it won't contribute to total lik
    indx_large = Area > (mean_x_c1_d3 + 2./np.sqrt(prec_x_c1_d3))
    Loglik_c2_d3[indx_large] = Loglik_c1_d3[indx_large]
    
    log_lik1 = Loglik_c1_d1 + Loglik_c1_d2 + Loglik_c1_d3
    log_lik2 = Loglik_c2_d1 + Loglik_c2_d2 + Loglik_c2_d3
    
    probs_full = getProbs(log_lik1, log_lik2, alpha)
    # correct for mad outliers
    probs_full_correct = correctProbsOutliers(probs_full, data_2, mean_x_c1_d2, prec_x_c1_d2*Area, 5)
    probs_full_correct = correctProbsOutliers(probs_full_correct, data_1, mean_x_c1_d1, prec_x_c1_d1, -5)
    
    probs_d1   = getProbs(Loglik_c1_d1, Loglik_c2_d1, alpha)
    probs_d2   = getProbs(Loglik_c1_d2, Loglik_c2_d2, alpha)
    probs_d3   = getProbs(Loglik_c1_d3, Loglik_c2_d3, alpha)    
    probs_all  = rbind((probs_full_correct, probs_d1, probs_d2, probs_d3))
#    if mean_x_c1_d1 < mean_x_c2_d1: 
#        print("Distributions have swapped place!! making probs  = 1 - probs")
#        probs_full = 1.- probs_full
#        probs_all  = 1.- probs_all
    return probs_full_correct, probs_all
    
def indx_filterData(x_all, perc_halo = 0, perc_nucl = 100, perc_area = 100, 
                    hard_lim_nucl = 0.5, hard_lim_halo = 0.3):
                        
    keep_this_halo = np.percentile( x_all[:,0], perc_halo) #10
    keep_this_nucl = np.percentile( x_all[:,1], perc_nucl) #90) 
    keep_this_area = np.percentile( x_all[:,2], perc_area) #90)
    
    if keep_this_halo < hard_lim_halo: keep_this_halo = hard_lim_halo
    if keep_this_nucl > hard_lim_nucl: keep_this_nucl = hard_lim_nucl
        
    indx_use_these   = np.logical_and( x_all[:,0] >= keep_this_halo, 
                                       x_all[:,1] <= keep_this_nucl)
    indx_use_these   = np.logical_and(indx_use_these, 
                                      x_all[:,2] <= keep_this_area)  
    return indx_use_these

## Yup, I do a lot of R coding ...
def cbind(list_to_bind):
    xx_all = np.concatenate(list_to_bind)  
    xx_all = xx_all.reshape((len(list_to_bind[0]),len(list_to_bind)), order='F')
    return xx_all

def rbind(list_to_bind):
    xx_all = np.concatenate(list_to_bind )
    xx_all = xx_all.reshape((len(list_to_bind), len(list_to_bind[0])), order='C')
    return xx_all
    
def compareToGoldStandard(indx_on, indx_True):
    FP = np.bitwise_and(indx_on, np.logical_not(indx_True))
    TP = np.bitwise_and(indx_on, indx_True)
    FN = np.bitwise_and(np.logical_not(indx_on), indx_True)
    TN = np.bitwise_and(np.logical_not(indx_on), np.logical_not(indx_True))
    print("TP = " + str(np.sum(TP)) + "  FP = " + str(np.sum(FP)))
    print("TN = " + str(np.sum(TN)) + "  FN = " + str(np.sum(FN)))
    return TP,FP,TN,FN

def calcuateGammaParameters(mean_x, sd_x, sdUnits = True):
    if sdUnits:
        mean_x = 1./mean_x**2
        sd_x   = 1./sd_x**2
    b = mean_x/sd_x**2
    a = mean_x*b
    return a, b
    
