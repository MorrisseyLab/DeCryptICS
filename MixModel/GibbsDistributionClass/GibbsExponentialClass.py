# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:49:04 2015

@author: edward
"""
from numpy.random import gamma
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma as gamma_fun
from numba import jitclass, jit, double, boolean
from math import log


spec = dict([
    ('prior_a', double),          # an array field
    ('prior_b', double),          # an array field
    ('x_vec_full', double[:]),          # an array field
    ('Explambda', double),          # an array field
])


@jitclass(spec)
class Exponential_Component_Gibbs(object):
    def __init__(self, x_vec, prior_a, prior_b, init_Explambda = None):
        if init_Explambda is None: init_Explambda = 0.01
        self.prior_a     = prior_a
        self.prior_b     = prior_b
        self.x_vec_full  = x_vec
        
        ## Variables for mcmc
        self.Explambda        = init_Explambda

    ## Update precision of exponential
    def updateLambdaExp(self, new_gammas): 
        num_data = 0.
        x_sum    = 0.
        for i in range(len(new_gammas)):
            num_data += new_gammas[i]
            x_sum    += self.x_vec_full[i]*new_gammas[i]
        cond_shape = self.prior_a + num_data
        cond_rate  = self.prior_b + x_sum
        self.Explambda  = gamma(cond_shape, 1./cond_rate)

    ## Calculate loglik for each data point (diregarding cluster assignment ie gamma)
    def logLike_allData(self): 
        size_vec    = len(self.x_vec_full)
        log_lik_vec = np.zeros(size_vec, dtype = np.double)
        const_val   = log(self.Explambda)
        for i in range(size_vec):        
            log_lik_vec[i] = const_val - self.Explambda*self.x_vec_full[i]
        return log_lik_vec

    ## Update both gaussian parameters
    def update_params(self, new_gammas): 
        self.updateLambdaExp(new_gammas)

    def gammaDistr(self, a, b, x):
        # Plot gamma distribution
        return b**a*x**(a-1)*np.exp(-b*x)/gamma_fun(a)

#    def plotPriors(self):
#        # Plot gamma distribution
#        mean_gamma = self.prior_a/self.prior_b
#        var_gamma  = self.prior_a/self.prior_b**2
#        xmax       = mean_gamma + 3.*var_gamma
#        xmin       = mean_gamma - 3.*var_gamma
#        if xmin < 0: xmin = 0
#        x = np.arange(xmin, xmax, (xmax-xmin)/200.)
#        y = self.gammaDistr(self.prior_a, self.prior_b, x)
#        plt.plot(x, y, linestyle = '-', linewidth=2)
#        plt.show()
        
                

