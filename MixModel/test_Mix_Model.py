# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:06:27 2015

@author: edward
"""
import numpy as np
import matplotlib.pyplot as plt
#from mixModelThreeDataTypes_funcs import *
from mixModelThreeDataTypes_funcs import cbind, initGammas_kmeans, GibbsMixFunnel2Clusters3d_2, predictVals, plotMAP_fit #indx_filterData, 
from mixModelThreeDataTypes_funcs import plotChains, plotPosteriors

from numpy.random import normal, exponential


def mixGibbs(allHalo, allMeanNucl, allSizes, num_iter, thin, plot_me = False): 
    ## Shift areas so as to start at size 1 
    xx_all          = cbind((allHalo, allMeanNucl, allSizes))
#    indx_use_these  = indx_filterData(xx_all, perc_halo = 5, perc_nucl = 95, perc_area = 95, hard_lim_nucl = 0.5, hard_lim_halo = 0.3) # Remove outliers
#    # If nothing left after filtering return empty    
#    if  np.sum(indx_use_these) == 0:
#        return [],[],[]
    xx_filt         = xx_all.copy() #[indx_use_these,:]
    gamma_i    = initGammas_kmeans(xx_filt)

    gamma_all, mcmc_out =  GibbsMixFunnel2Clusters3d_2(xx_filt, gamma_i,  num_iter, thin)
    p_full, p_all = predictVals(mcmc_out, xx_all[:,0], xx_all[:,1], xx_all[:,2])
    if plot_me:
        plt.figure() 
        plotChains(mcmc_out)
        plt.figure() 
        plotPosteriors(mcmc_out)
#        cluster_probs = gamma_all/num_iter  
#        indx_on       = cluster_probs > 0.5
        plt.figure() 
        plotMAP_fit(mcmc_out, xx_all[:,0], xx_all[:,1], xx_all[:,2], p_full > 0.5) #indx_True)
    return p_full > 0.5 , p_all, np.mean(mcmc_out,0)


## Simulate data (first N data points are of cluster 1 [true vals] and second N data points come from 2 [False Vals]) ==================================
dataSet_size  = 400
#Areas 
area_sim      = np.concatenate((     normal(  20000,  5000, dataSet_size),
                                  exponential(1/0.0001,  dataSet_size))) 
#Mean 
Mean_sim      = np.concatenate((     normal(  0.03,   4/np.sqrt(area_sim[0:dataSet_size])),
                                     normal(  0.1,    4/np.sqrt(area_sim[dataSet_size:]))))
                                     
#Halo
Halo_sim      = np.concatenate((     normal(  0.75,   0.1, dataSet_size),
                                     normal(  0.55,   0.1, dataSet_size)))

                              
#start_time = timeit.default_timer()
out_vals = mixGibbs(Halo_sim, Mean_sim, area_sim, 2000, 5) #, plot_me = True)


#print(timeit.default_timer() - start_time)
# First half should all be true 
print("TP:" + str(np.sum(out_vals[0][0:dataSet_size])) + " out of " + str(dataSet_size))
print("FP:" + str(np.sum(out_vals[0][dataSet_size:]))  + " out of " + str(dataSet_size))


np.savetxt('area.out', area_sim, delimiter=',')
np.savetxt('mean.out', Mean_sim, delimiter=',')
np.savetxt('Halo.out', Halo_sim, delimiter=',')

kk_init = initGammas_kmeans(cbind((Halo_sim, Mean_sim, area_sim)))
np.savetxt('Init_gamma.out', kk_init, delimiter=',')



