# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:08:12 2015

#author: edward
"""
from numpy.random import normal, gamma
import numpy as np
from numba import jitclass, double
from scipy.special import gamma as gamma_fun
from math import log


spec = dict([
    ('prior_a', double),          
    ('prior_b', double),          
    ('prior_mu', double),          
    ('prior_lamb', double),          
    ('A_vec_full', double[:]),          
    ('x_vec_full', double[:]),          
    ('log_A_vec_full', double[:]),          
    ('mu', double),          
    ('lamb', double),          
    ('const_val', double),          
    ('Ax_vec_full', double[:])          
])


@jitclass(spec)
class Gaussian_Component_Gibbs(object):
    def __init__(self, x_vec, A_vec, prior_mu,  prior_lamb, prior_a, prior_b, init_mu): 
        init_lambda = 0.01
        self.prior_a        = prior_a
        self.prior_b        = prior_b
        self.prior_mu       = prior_mu
        self.prior_lamb     = prior_lamb
        self.A_vec_full     = A_vec
        self.x_vec_full     = x_vec
        self.log_A_vec_full = np.log(A_vec)
               
        ## Variables for mcmc
        self.mu          = init_mu
        self.lamb        = init_lambda

        self.const_val   = -0.5*np.log(2.*np.pi)
        self.Ax_vec_full = A_vec*x_vec
        
        
    ## Update mean of gaussian
    def updateMu(self, new_gammas): 
        A_sum     = 0.
        x_sum     = 0.
        for i in range(len(new_gammas)):
            A_sum  += self.A_vec_full[i]*new_gammas[i]
            x_sum  += self.Ax_vec_full[i]*new_gammas[i]
        cond_lamb = self.lamb*A_sum + self.prior_lamb
        cond_mu   = (self.prior_lamb*self.prior_mu+self.lamb*x_sum)/cond_lamb
        self.mu   = normal(cond_mu, 1./np.sqrt(cond_lamb))

    ## Update precision of gaussian
    def updateLambda(self, new_gammas): 
        len1  = 0.
        sumsq = 0.
        for i in range(len(new_gammas)):
            len1  += new_gammas[i]
            sumsq += new_gammas[i] * self.A_vec_full[i]*(self.x_vec_full[i]-self.mu)**2
        cond_shape = self.prior_a + 0.5*len1
        cond_scale = self.prior_b + 0.5*sumsq        
        self.lamb = gamma(cond_shape, 1./cond_scale)

    def logLike_allData(self): #mu, lamb, A_vec, x_vec        
        size_vec    = len(self.A_vec_full)
        log_lik_vec = np.zeros(size_vec, dtype = np.double)
        const_val2  = self.const_val + 0.5*log(self.lamb)
        for i in range(size_vec):        
            log_lik_vec[i] = const_val2 + 0.5*self.log_A_vec_full[i] - 0.5*self.lamb*self.A_vec_full[i]*(self.x_vec_full[i] - self.mu)**2
        return log_lik_vec
       
    ## Update both gaussian parameters
    def update_params(self, new_gammas): #mu, lamb, A_vec, x_vec
        self.updateMu(new_gammas)
        self.updateLambda(new_gammas)

    def gammaDistr(self, a, b, x):
        # Plot gamma distribution
        return b**a*x**(a-1)*np.exp(-b*x)/gamma_fun(a)





#kk = Gaussian_Component_Gibbs(np.ones(4), np.ones(4), 1, 2 ,3 ,4,5)


#    def plotPriors(self):
#        # Plot gamma distribution
#        mean_gamma = self.prior_a/self.prior_b
#        var_gamma  = self.prior_a/self.prior_b**2
#        print   mean_gamma ,   var_gamma    
#        xmax       = mean_gamma + 6.*np.std(var_gamma)
#        xmin       = mean_gamma - 3.*var_gamma
#        if xmin < 0: xmin = 0
#        x_gam = np.linspace(xmin, xmax, 200)
#        y_gam = self.gammaDistr(self.prior_a, self.prior_b, x_gam)
#        
#        sd_gam = 1./np.sqrt(self.prior_lamb)
#        x_norm = np.linspace(self.prior_mu - 3.*sd_gam, self.prior_mu + 3.*sd_gam, 200)
#        y_norm = norm.pdf(x_norm, self.prior_mu, sd_gam)
#
#        plt.subplot(121)
#        plt.plot(x_norm, y_norm, linestyle = '-', linewidth=2)
#        plt.title('Prior on mean of Gaussian', color = 'r')
#        plt.subplot(122)
#        plt.plot(x_gam, y_gam, linestyle = '-', linewidth=2)
#        plt.title('Prior on precision of Gaussian', color = 'r')
#        plt.show()
#
#
#
#@jit("double(double[:])")
#def Sum_nb(x_vec):
#    out_num = 0.
#    for i in x_vec:
#        out_num += i
#    return out_num
 


#lamb       = 10.
#A_vec_full = 1.*np.arange(1,1001)
#x_vec_full = 1.*np.arange(1,1001)
#mu         = 10.
#const_val  = 0.2
#log_A_vec_full = np.log(A_vec_full)
#
#%timeit uu = logLike_allData_i(lamb, A_vec_full, log_A_vec_full, const_val, x_vec_full, mu)
#%timeit uu1 = logLike_allData_i_old(lamb, A_vec_full, const_val, x_vec_full, mu)
#








