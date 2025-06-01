"""

This script implements Hedges' publication selection effects model.

Author - Dirk Ostwald

"""

# initialization
# -----------------------------------------------------------------------------
# general utility import
import numpy as np                                                              # NumPy
import scipy.stats as rv                                                        # random variable module
from scipy.stats import rv_continuous                                           # new continuous rv class
import matplotlib.pyplot as plt                                                 # Matplotlib
plt.close("all")                                                                # close all figures
plt.rc('text', usetex = True)               # Latex annotations

import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import os
import sys

# directory management
wdir    = os.getcwd()                                                           # working directory
udir    = os.path.join(wdir,'Utilities')                                        # project utitilities directory
rdir    = os.sep.join(wdir.split(os.sep)[:-1])                                  # project root directory
fdir    = os.path.join(rdir,'Figures\Statistics_for_Meta_Analysis')             # figure directory

# project specific utilities
sys.path.append(udir)                                                           # utilities directory addition to Python path
from sma_imagesc import sma_imagesc                                             # Python imagesc utility
from sma_structure import sma_structure                                         # Python structure utility

def wp_fun(p,a,omega):

    """
    This function implements the p-value-based study publication probability
    function. It returns the weight w(p) associated with a given p-value
    depending on the weight function's step point and omega parameters.
    
    Inputs
            p       : p-value
            a       : l   x 1 array of step-points 0 = a_0, a_1, ..., a_l = 1
            omega   : l-1 x 1 array of function values omega_1, ..., omega_l
    
    Outputs
            w       : scalar weight function values
            
    
    """  
    # default w value
    w = np.nan
        
    # non-zero p-values
    for j in range(1,len(a)):
        if a[j-1] < p < a[j]:
            w = omega[j-1]
    
    return w
    

def w_fun(d, sigma, a, omega):
    
    """ 
    This function implements the p-value-based study publication probability
    function. It returns the weight w(d,\sigma) associated with a given effect
    size and standard deviation depending on the weight function's step point 
    and omega parameters.
    
    Inputs
            d       : scalar effect size value  
            sigma   : scalar standard deviation value
            a       : (l - 2) x 1 array of step-points
            omega   : l x 1 array of weight function value 
    
    Outputs
            w       : scalar weight function values
            p       : associated p-value
    """
    z   = np.abs(d) / sigma                                                     # z-value
    p   = 2*rv.norm.cdf(-z,0,1)                                                 # p-value
    w   = wp_fun(p,a,omega)                                                     # weight value
    
    return w, p


def c_fun(theta):
    
    """
    This function evaluates the normalization constant of the publication 
    selection effects model.
    
    Inputs
        theta   : parameter structure with fields
            .Delta      : effect size parameter
            .tau        : between-study effect size variance    
            .sigsqr     : within-study effect size variance
            .a          : weight function step points
            .omega      : weight function value parameter
    
    Outputs
        normalization constant
        
    """
    eta     = np.sqrt(theta.tau + theta.sigsqr)                                 # marginal standard deviation
    B_j     = np.full([len(theta.omega)], np.nan)                               # B_j array initialization
    b_j     = -theta.sigsqr * rv.norm.ppf(theta.a[1:-1:1]/2,0,1)
    
    # B_1
    B_j[0]  = 1  - rv.norm.cdf((  b_j[0]  - theta.Delta)/eta,0,1)     \
                 + rv.norm.cdf(( -b_j[0]  - theta.Delta)/eta,0,1)
    
    # B_1<j<k
    for j in range(1, len(theta.omega)-1):
        B_j[j] =   rv.norm.cdf(( b_j[j-1] - theta.Delta)/eta,0,1)   \
                 - rv.norm.cdf(( b_j[j]   - theta.Delta)/eta,0,1)   \
                 + rv.norm.cdf((-b_j[j]   - theta.Delta)/eta,0,1)   \
                 - rv.norm.cdf((-b_j[j-1] - theta.Delta)/eta,0,1)  
        
    # B_k
    B_j[-1] =     rv.norm.cdf((  b_j[-1]   - theta.Delta)/eta,0,1)    \
                - rv.norm.cdf(( -b_j[-1]   - theta.Delta)/eta,0,1)

    return np.dot(B_j, theta.omega)                                             # normalization constant



def pdf_d_fun(d, theta):
    
    """
    This function evaluates the PDF of the publication selection effects model.
    
    Inputs
        d       : scalar effect size
        theta   : parameter structure with fields
            .Delta      : effect size parameter
            .tau        : between-study effect size variance    
            .sigsqr     : within-study effect size variance
            .a          : weight function step points
            .omega      : weight function value parameter
    
    Outputs
        phi value of d_i
        weight value of d_i
        PDF value of d_i
        
    """
    eta = np.sqrt(theta.tau + theta.sigsqr)                                     # marginal standard deviation
    phi = rv.norm.pdf((d-theta.Delta) / eta, 0, 1)                              # standard normal density value
    w,p = w_fun(d, eta, theta.a, theta.omega)                                   # weight function value, p-value
    c   = c_fun(theta)                                                          # normalization constant

    
    return phi, w, c*w*phi
    
class hedges_gen(rv_continuous):

    """ 
    This function creates a new instance of class stats.rv conforming to a 
    Hedges' p-valued based selected effect size model
    
    """
    def _pdf(self, d, Delta, tau, sigsqr, a, omega):
        
        # parameter packing
        theta           = sma_structure()
        theta.Delta     = Delta
        theta.tau       = tau
        theta.sigsqr    = sigsqr
        theta.a         = a
        theta.omega     = omega
        
        # PDF evaluation
        phi, w, pdf_d   = pdf_d_fun(d, theta)
       
        return pdf_d  
    
# standard normal distribution functions and p-values
# -----------------------------------------------------------------------------
a       = np.array([0, 0.001, 0.05, 0.1, 0.5, 1])                               # step-points a \in [0,1]
omega   = np.array([1,.9, .5,.3,.1])                                            # weight function values
d_min   = -2                                                                    # minimum effect size
d_max   = 2                                                                     # maximum effect size
d_res   = 1000                                                                  # effect size space resolution
d       = np.linspace(d_min, d_max, d_res)                                      # effect size space
s_min   = 1e-1                                                                  # minimum standard deviation
s_max   = 1                                                                     # maximum standard deviation
s_res   = 1000                                                                  # standard deviation space resolution
s       = np.linspace(s_min, s_max,s_res)                                       # standard deviation space
w_ds    = np.full([s_res, d_res], np.nan)                                       # weight function value array initialization
p_ds    = np.full([s_res, d_res], np.nan)                                       # p-valeu array initialization
for i, s_i in np.ndenumerate(s):
    for j, d_j in np.ndenumerate(d):
        w_ds[i,j], p_ds[i,j] = w_fun(d_j,s_i,a,omega)
    

fig         = plt.figure(figsize = (12,4))                                      # figure initialization and figure size
gs          = gridspec.GridSpec(1,3)                                            # subplot layout
ax          = {}                                                                # axes dictionary initialization
prop_cycle  = plt.rcParams['axes.prop_cycle']                                   # default color cycle
colors      = prop_cycle.by_key()['color']                                      # default color cycle


# \phi(z)
z_min = -4                                                                      # minimum Z-value
z_max =  4                                                                       # maximum Z-value
z_res = 1000                                                                    # Z-space resolution
z     = np.linspace(z_min, z_max, z_res)                                        # Z-space
Z_i   = 1.7                                                                   # exemplary Z_i value
z_i   = np.linspace(Z_i, z_max, z_res)                                          # Z > Z_i     

ax[0] = plt.subplot(gs[0,0])
ax[0].plot(z,rv.norm.pdf(z,0,1))
ax[0].fill_between(z_i, 0, rv.norm.pdf(z_i,0,1), 
                   facecolor = '#1f77b4', edgecolor = '#1f77b4', alpha = 0.2)
ax[0].fill_between(-z_i[-1::-1], 0, rv.norm.pdf(-z_i[-1::-1],0,1), 
                   facecolor = '#1f77b4', alpha = 0.2)
ax[0].set_xlim(z_min,z_max)
ax[0].set_ylim(0,.45)
ax[0].set_xlabel(r'$z$', fontsize = 18)
ax[0].set_title(r'$\phi : \mathbf{R} \to \mathbf{R}, z \mapsto \phi(z)$', fontsize = 18)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].tick_params(labelsize = 16)

# \Phi(z)
ax[1] = plt.subplot(gs[0,1])
ax[1].plot(z,rv.norm.cdf(z,0,1))
ax[1].fill_between(-z_i[-1::-1], 0, rv.norm.cdf(-z_i[-1::-1],0,1), 
                   facecolor = '#1f77b4', alpha = 0.2)
ax[1].set_xlim(z_min,z_max)
ax[1].set_ylim(0,1.05)
ax[1].set_xlabel(r'$z$', fontsize = 18)
ax[1].set_title(r'$\Phi : \mathbf{R} \to [0,1], z \mapsto \Phi(z)$', fontsize = 18)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].tick_params(labelsize = 16)
ax[1].annotate(r'$2\Phi(-1.7) = 2\cdot {0:1.2f} = {1:1.2f}$'.format(rv.norm.cdf(-1.7,0,1), 2*rv.norm.cdf(-1.7,0,1)), (-.5,0.2), fontsize = 12)


# p-values
ax[2] = plt.subplot(gs[0,2])
im, cbar = sma_imagesc(ax       = ax[2], 
                       A        = p_ds.T, 
                       x        = s, 
                       y        = d, 
                       xticks   = np.linspace(s_min,s_max, 5), 
                       yticks   = np.linspace(d_min,d_max, 5) ,
                       zmin     = 0, 
                       zmax     = 1, 
                       cmap     = plt.get_cmap('RdYlGn'),
                       cfs      = 14)
ax[2].set_title(r'$p(d_i,\sigma_i)$', fontsize = 18)
ax[2].set_ylabel(r'$d_i$', fontsize = 18, rotation = 0,position = (-.1,.44))
ax[2].set_xlabel(r'$\sigma_i$', fontsize = 18)
ax[2].tick_params(labelsize = 14)
ax[2].xaxis.set_major_formatter(FormatStrFormatter('$%.2f$'))
ax[2].yaxis.set_major_formatter(FormatStrFormatter('$%.2f$'))

# figure printing 
fig.tight_layout()
fig.savefig(os.path.join(fdir, 'sma_figure_2_1.pdf'), dpi = 300, format = 'pdf')

# p-value-based weight function
# -----------------------------------------------------------------------------
fig     = plt.figure(figsize = (18,5))                                         # figure initialization and figure size
gs      = gridspec.GridSpec(1,3)                                                # subplot layout
ax      = {}                                                                    # axes dictionary initialization
p_min   = 1e-5                                                                  # minimal p-value
p_max   = 1 - 1e-5                                                              # maximal p-value
p_res   = 1000                                                                  # p-value space resolution
p       = np.linspace(p_min, p_max, p_res)                                      # p-value space 
wp      = np.full([p_res], np.nan)                                              # weight function
for i, p_i in np.ndenumerate(p):
    wp[i] = wp_fun(p_i,a,omega)                                                   # weight value
  
ax[0] = plt.subplot(gs[0,0])
ax[0].plot(p, wp, label = r'$w(p_i)$')
ax[0].plot(a, np.zeros(len(a)), 'o', linestyle = '', clip_on = False, label = r'$a$')
ax[0].set_xlim(0,1)
ax[0].set_ylim(0,1)
ax[0].legend(fontsize = 18)
ax[0].set_xlabel(r'$p_i$', fontsize = 20)
ax[0].set_title(r'$w(p_i)$', fontsize = 22)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].tick_params(labelsize = 16)

# weights and p-values for constant \sigma
soi   = np.int32(d_res/2)
ax[1] = plt.subplot(gs[0,1])
ax[1].plot(d,w_ds[soi,:], label = r'$w$')
ax[1].plot(-2*np.ones(len(a)),a, 'o', linestyle = '', clip_on = False, label = r'$a$')
ax[1].plot(d,p_ds[soi,:], label = r'$p$')
ax[1].set_title(r'$\sigma = {0:1.1f}$'.format(s[soi]), fontsize = 22)
ax[1].set_xlabel(r'$d$', fontsize = 18)
ax[1].set_xlim(d_min,d_max)
ax[1].set_ylim(0,1)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].tick_params(labelsize = 16)
ax[1].legend(fontsize = 18)
ax[1].xaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))
ax[1].yaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))

# weights
ax[3] = plt.subplot(gs[0,2])
im, cbar = sma_imagesc(ax       = ax[3], 
                       A        = w_ds.T, 
                       x        = s, 
                       y        = d, 
                       xticks   = np.linspace(s_min,s_max, 5), 
                       yticks   = np.linspace(d_min,d_max, 5) ,
                       zmin     = 0, 
                       zmax     = 1, 
                       cmap     = plt.get_cmap('bwr'),
                       cfs      = 14)

ax[3].set_title(r'$w(d_i,\sigma_i)$', fontsize = 22)
ax[3].set_ylabel(r'$d_i$', fontsize = 18, rotation = 0, position = (.1,.46))
ax[3].set_xlabel(r'$\sigma_i$', fontsize = 18)
ax[3].tick_params(labelsize = 16)
ax[3].xaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))
ax[3].yaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))


# figure printing 
fig.tight_layout()
fig.savefig(os.path.join(fdir, 'sma_figure_2_2.pdf'), dpi = 300, format = 'pdf')


# likelihood function
# -----------------------------------------------------------------------------
tau     = .1                                                                    # between-study variance parameter
a       = np.array([0, 0.001, 0.05, 0.1, 0.5, 1])                               # step-points a \in [0,1]
omega   = np.array([1,.9, .5,.3,.1])                                            # weight function values
Delta   = np.array([0, 1, 2])                                                   # true, but unknown, effect size
sigsqr  = np.array([.05,.5])                                                        # within-study variances
d_min   = -2                                                                    # minimum effect size
d_max   = 4                                                                     # maximum effect size
d_res   = 1000                                                                  # effect size space resolution
d       = np.linspace(d_min, d_max, d_res)                                      # effect size space
pdf_d   = np.full([len(Delta), len(sigsqr), d_res,3], np.nan)                   # PDF array initialization

# single study likelihood functions
for i, Delta_i in np.ndenumerate(Delta):
    for j, sigsqr_j in np.ndenumerate(sigsqr):
        
        # likelihood function evaluation
        theta           = sma_structure()                                           # structure initialization
        theta.Delta     = Delta_i
        theta.tau       = tau
        theta.sigsqr    = sigsqr_j
        theta.a         = a
        theta.omega     = omega
  
        # likelihood function evaluation
        for k, d_k in np.ndenumerate(d):
            phi, w, w_phi   = pdf_d_fun(d_k,theta)
            pdf_d[i,j,k,0]  = phi
            pdf_d[i,j,k,1]  = w
            pdf_d[i,j,k,2]  = w_phi
    

# visualization
for j, sigsqr_j in np.ndenumerate(sigsqr):
    fig         = plt.figure(figsize = (12,8))                                      # figure initialization and figure size
    gs          = gridspec.GridSpec(3,3)                                            # subplot layout
    ax          = {}                                                                # axes dictionary initialization
    idx         = 0 
    for i, delta_i in np.ndenumerate(Delta):
        title  = (r'$\phi(d_i), \delta = {0:1.0f}$'.format(delta_i), r'$w(d_i,\sigma_i)$', r'$p(d_i)$')
        for k in range(3):
                ax[idx] = plt.subplot(gs[i[0],k])
                ax[idx].plot(d, np.squeeze(pdf_d[i,j,:,k]), label = r'$\sigma_i = {0:1.1}$'.format(sigsqr_j))
                ax[idx].set_xlim(d_min,d_max)
                ax[idx].set_xlabel(r'$d_i$', fontsize = 18)
                ax[idx].set_title(title[k], fontsize = 18)
                ax[idx].spines['right'].set_visible(False)
                ax[idx].spines['top'].set_visible(False)
                ax[idx].legend(loc = 'lower right', fontsize = 10)
                ax[idx].tick_params(labelsize = 14)
                ax[idx].grid(True, linewidth = .5, color = [.9,.9,.9])
                idx = idx + 1

    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 'sma_figure_2_{0:1}.pdf'.format(j[0]+3)), dpi = 300, format = 'pdf')
 

