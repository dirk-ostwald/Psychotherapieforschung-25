"""

This script implements Hedges' bias control for standardized effect size 
estimation.

Author - Dirk Ostwald

"""

# initialization
# -----------------------------------------------------------------------------
# general utility import
import numpy as np
import scipy.stats as rv                                                        # random variable module
from scipy.special import gamma as gamma_fun                                    # the gamma function
import matplotlib.pyplot as plt                                                 # matplotlib
plt.close("all")                                                                # close all figures
plt.rc('text', usetex = True)                                                   # Latex annotations
import matplotlib.gridspec as gridspec                                          # subplot utilities
import os                                                                       # OS utilities
import sys                                                                      # system utilities

# directory management
wdir            = os.getcwd()                                                   # working directory
udir            = os.path.join(wdir,'Utilities')                                # project utitilities directory
rdir            = os.sep.join(wdir.split(os.sep)[:-1])                          # project root directory
fdir            = os.path.join(rdir, 'Figures', 'Statistics_for_meta_analysis') # figure directory

# statistics for meta-analysis utility import
sys.path.append(udir)                                                           # utilities directory addition to Python path
from sma_imagesc import sma_imagesc                                             # Python imagesc utility

# functions
# -----------------------------------------------------------------------------
def S_fun(S_e, S_c, n_e, n_c):
      
    """
    This function computes a pooled standard deviation based on two group 
    standard deviations and their respective group sizes
    
    Inputs
        S_e     : experimental group standard deviation
        S_c     : control group standard deviation
        n_e     : experimental group size
        n_c     : control group size
            
    Outputs
        S       : pooled standard deviation
    
    """
    S = np.sqrt( ((n_e - 1) * S_e ** 2 + (n_c - 1) * S_c ** 2)/(n_e + n_c - 2))
    
    return S 

def n_fun(n_e, n_c):
    
    """
    This function computes a pooled sample size based on two group sizes
    
    Inputs
        n_e     : experimental group size
        n_c     : control group size
            
    Outputs
        tilde_n : pooled group size
    
    """
    tilde_n = (n_e * n_c) / (n_e + n_c)

    return tilde_n    
    
# non-central t-distribution
# -----------------------------------------------------------------------------
t_min       = -4                                                                # minimum T value
t_max       = 12                                                               # maximum T value
t_res       = 1000                                                              # T space resolution
t           = np.linspace(t_min,t_max,t_res)                                    # T space
mu          = np.array([0,3,6])                                                 # non-centrality parameter values
nu          = np.array([30,10,2])                                               # degrees of freedom
colors      = np.array([[.1,.1,.3],[.2,.2,.6],[.3,.3,.9]])
linesty     = ['-', '--', ':']


# visualization
fig         = plt.figure(figsize = (9,5))                 
gs          = gridspec.GridSpec(1,1)                                       
ax          = {}            
ax[0]       = plt.subplot(gs[0,0]) 
for i, mu_i in np.ndenumerate(mu):
    for j, nu_j in np.ndenumerate(nu):
        ax[0].plot( t, 
                    rv.nct.pdf(t, nu_j, mu_i,), 
                    label = r'$\mu = {0:1.2f}, \nu = {1:1.2f}$'.format(mu_i,nu_j), 
                    color = colors[i[0],:],
                    linestyle = linesty[j[0]])
ax[0].legend(fontsize = 12)
ax[0].set_xlim(t_min,t_max)
ax[0].set_title(r'$t(\mu,\nu)$', fontsize = 24)
ax[0].set_xlabel(r'$T$', fontsize = 20)
ax[0].tick_params(labelsize = 18)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)



# figure printing 
fig.tight_layout()
fig.savefig(os.path.join(fdir, 'sma_figure_1_0.pdf'), dpi = 300, format = 'pdf')

# single study standardized effect size estimation
# -----------------------------------------------------------------------------
# simulation parameters
n_sim       = np.int32(1e4)                                                     # number of simulations

# model parameter specifications
Delta       = 2                                                                 # true, but unknown, standardized effect size
d_min       = 1                                                                 # minimum d value
d_max       = 11                                                                # maximum d value
d_res       = np.int32(1e2)                                                     # d space resolution
d           = np.linspace(d_min, d_max, d_res)                                  # d space

# study-specific parameters
n_e         = 10                                                                # number of experimental units, experimental group
n_c         = n_e                                                               # number of experimental units, control group
sigma       = .5                                                                # sigma value
gamma       = 1                                                                 # gamma value
mu_e        = sigma*Delta + sigma*gamma                                         # experimental group expectation
mu_c        = sigma*gamma                                                       # experimental group expectation
delta       = (mu_e - mu_c)/sigma                                               # study-specific standardized effect size
m           = n_e + n_c - 2                                                     # degrees of freedom
n           = n_fun(n_e,n_c)                                                    # pooled group size

# simulation
d_rvs       = np.full(n_sim, np.nan)                                            # d realizations array initialization
for s in range(n_sim):                                                          # simulation iterations
    X       = np.full((n_e,2), np.nan)                                          # study data set initialization
    for g,mu in enumerate((mu_e, mu_c)):
        for j in range(n_e):                                                    # experimental unit iterations
            X[j,g] = rv.norm.rvs(mu, sigma)                                     # experimental unit sampling
    
    # summary statistics evaluation
    X_e_bar, X_c_bar    = X.mean(axis = 0)                                      # group sample means
    S_e, S_c            = X.std(axis = 0, ddof = 1)                             # unbiased group standard deviations
    S                   = S_fun(S_e, S_c, n_e, n_c)                             # pooled standard deviation
    d_rvs[s]            = (X_e_bar - X_c_bar)/S
   

# visualization
print('Pooled group size n = {0:3.1f}'.format(n))
fig             = plt.figure(figsize = (18,6))                           
gs              = gridspec.GridSpec(1,4)                                       
ax              = {}            

# sqrt(n)d distribution                                                                                                     
ax[0] = plt.subplot(gs[0,0:2])  
ax[0].hist(np.sqrt(n)*d_rvs, density = True, bins = 'auto', facecolor = [.9, .9, .9], edgecolor = 'black', linewidth = .5)
ax[0].plot(d, rv.nct.pdf(d, m, np.sqrt(n)*delta), linewidth = 2)  
ax[0].set_xlim(d_min,d_max)
ax[0].set_title(r'$\sqrt{n}d \sim t(\sqrt{n}\Delta,m)$', fontsize = 35)
ax[0].set_xlabel(r'$\sqrt{n}d$', fontsize = 30)
ax[0].tick_params(labelsize = 25)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)


# expected value/variance
c_m     = (gamma_fun(m/2))/(np.sqrt(m/2)*gamma_fun((m-1)/2))
ax[1] = plt.subplot(gs[0,2])  
ax[1].errorbar( np.array([1,2]), 
                np.array([d_rvs.mean(), c_m*d_rvs.mean()]), 
                np.array([d_rvs.var(), (c_m**2)*d_rvs.var()]), linestyle = '', markersize = 10,
                capsize = 6, color = [0,0,0], marker = 'o', mfc = [0,0,0], lw = 1)
ax[1].plot(np.linspace(0,3,10), Delta*np.ones(10), color = [.7,.7,.7])
ax[1].set_xlim(.5, 2.5)
ax[1].set_ylim(1.6, 2.7)
ax[1].set_title(r'$\mbox{Bias and Variance}$', fontsize = 35)
ax[1].set_xticks([1,2])
ax[1].set_xticklabels([r'$\bar{d}$', r'$c_m\bar{d}$'])
ax[1].tick_params(labelsize = 25)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

# bias correction factor
m       = np.arange(2,40,1)
c_m     = (gamma_fun(m/2))/(np.sqrt(m/2)*gamma_fun((m-1)/2))
ax[2] = plt.subplot(gs[0,3])  
ax[2].plot(m, c_m, color = [0,0,0], lw  = 1)
ax[2].set_xlim(0,40)
ax[2].set_title(r'$c_m$', fontsize = 35)
ax[2].tick_params(labelsize = 25)
ax[2].set_xlabel(r'$m$', fontsize = 30)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)

# figure printing 
fig.tight_layout()
fig.savefig(os.path.join(fdir, 'sma_figure_1_1.pdf'), dpi = 300, format = 'pdf')

# multiple study standardized effect size estimation
# -----------------------------------------------------------------------------
# simulation parameters
n_sim   = 10000                                                                  # number of simulations/bias & variance estimate
Delta   = 2                                                                     # true, but unknown, effect size
k_min   = 5                                                                     # minimum number of studies
k_max   = 40                                                                    # maximum number of studies
gamma   = rv.norm.rvs(0, 1, size = k_max)                                       # study offsets
sigma   = rv.uniform.rvs(size = k_max)                                          # study scaling parameters (variances)
n_min   = 3                                                                     # minimum number of experimental group participants per study
n_max   = 30                                                                    # maximum number of experimental group participants per study

# simulation
d       = np.full([k_max-k_min, n_max-n_min,n_sim], np.nan)                     # uncorrected standardized effect sizes array initialization
d_u     = np.full([k_max-k_min, n_max-n_min,n_sim], np.nan)                     # corrected standardized effect sizes array initialization
for i,k in enumerate(range(k_min, k_max)):
    for j,n in enumerate(range(n_min, n_max)):
        
        # user information
        print('k = {0:2}, n = {1:2}'.format(k,n))
        
        # study parameters
        n_e     = n                                                             # experimental group size
        n_c     = n_e                                                           # control group size
        m       = n_e + n_c - 2                                                 # degrees of freedom
        nn      = n_fun(n_e,n_c)                                                # pooled group size
        mu_e    = sigma[i]*Delta + sigma[i]*gamma[i]                            # experimental group expectation
        mu_c    = sigma[i]*gamma[i]                                             # experimental group expectation
        delta   = (mu_e - mu_c)/sigma[i]                                        # study-specific standardized effect size
        c_m     = (gamma_fun(m/2))/(np.sqrt(m/2)*gamma_fun((m-1)/2))            # bias correction factor
        
        # simulation iterations
        for s in range(n_sim):
 
            # data set sampling
            X       = np.full((n_e,2), np.nan)                                  # study data set initialization
            for g, mu in enumerate((mu_e, mu_c)):
                for h in range(n_e):                                            # experimental unit iterations
                    X[h,g] = rv.norm.rvs(mu, sigma[i])                          # experimental unit sampling
           
            # summary statistics evaluation
            X_e_bar, X_c_bar    = X.mean(axis = 0)                              # group sample means
            S_e, S_c            = X.std(axis = 0, ddof = 1)                     # unbiased group standard deviations
            S                   = S_fun(S_e, S_c, n_e, n_c)                     # pooled standard deviation
            d[i,j,s]            = (X_e_bar - X_c_bar)/S                         # uncorrected standardized effect size estimate
            d_u[i,j,s]          = c_m*d[i,j,s]                                  # corrected standardized effect size estimate
            

# visualization
fig             = plt.figure(figsize = (10,7))                                  # figure initialization and figure size
gs              = gridspec.GridSpec(2,2)                                        # subplot layout
ax              = {}                                                            # axes dictionary initialization  
ax[0]           = plt.subplot(gs[0,0])
im, cbar        = sma_imagesc(ax[0], 
                              np.mean(d, axis = 2) - Delta, 
                              range(n_min,n_max), 
                              range(k_min,k_max),
                              -.8, .8, 
                              cmap = plt.get_cmap('bwr'))
ax[0].set_xticks(np.arange(n_min, n_max, 5))
ax[0].set_yticks(np.arange(k_min, k_max, 5))
ax[0].tick_params(labelsize = 12) 
ax[0].set_xlabel(r'Number of participants $n_e,n_c$', fontsize = 15)
ax[0].set_ylabel(r'Number of studies $k$', fontsize = 15)
ax[0].set_title(r'Bias $\hat{E}(d) - \Delta$', fontsize = 18)

ax[1]           = plt.subplot(gs[0,1])
im, cbar        = sma_imagesc(ax[1], 
                              np.var(d, axis = 2 ), 
                              range(n_min,n_max), 
                              range(k_min,k_max),
                              0, 2, 
                              cmap = plt.get_cmap('inferno'))
ax[1].set_xticks(np.arange(n_min, n_max,5))
ax[1].set_yticks(np.arange(k_min, k_max,5))
ax[1].tick_params(labelsize = 12) 
ax[1].set_xlabel(r'Number of participants $n_e,n_c$', fontsize = 15)
ax[1].set_ylabel(r'Number of studies $k$', fontsize = 15)
ax[1].set_title(r'Variance $\hat{V}(d)$', fontsize = 18)

ax[2]           = plt.subplot(gs[1,0])
im, cbar        = sma_imagesc(ax[2], 
                              np.mean(d_u, axis = 2) - Delta, 
                              range(n_min,n_max), 
                              range(k_min,k_max),
                              -.8, .8, 
                              cmap = plt.get_cmap('bwr'))
ax[2].set_xticks(np.arange(n_min, n_max,5))
ax[2].set_yticks(np.arange(k_min, k_max,5))
ax[2].tick_params(labelsize = 12) 
ax[2].set_xlabel(r'Number of participants $n_e,n_c$', fontsize = 15)
ax[2].set_ylabel(r'Number of studies $k$', fontsize = 15)
ax[2].set_title(r'Bias $\hat{E}(d_u) - \Delta$', fontsize = 18)

ax[3]           = plt.subplot(gs[1,1])
im, cbar        = sma_imagesc(ax[3], 
                              np.var(d_u, axis = 2), 
                              range(n_min,n_max), 
                              range(k_min,k_max),
                              0, 2, 
                              cmap = plt.get_cmap('inferno'))
ax[3].set_xticks(np.arange(n_min, n_max, 5))
ax[3].set_yticks(np.arange(k_min, k_max, 5))
ax[3].tick_params(labelsize = 12) 
ax[3].set_xlabel(r'Number of participants $n_e,n_c$', fontsize = 15)
ax[3].set_ylabel(r'Number of studies $k$', fontsize = 15)
ax[3].set_title(r'Variance $\hat{V}(d_u)$', fontsize = 18)

# figure printing 
fig.tight_layout()
fig.savefig(os.path.join(fdir, 'sma_figure_1_2.pdf'), dpi = 300, format = 'pdf')