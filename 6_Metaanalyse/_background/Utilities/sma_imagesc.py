def extents(f):
    """ 
    This function converts axis coordinates into an mpl.imshow.extent 
    compatible representation
    
    Inputs
        f : axis coordinates
        
    Outputs
        extent compatible tuple
    
    Author - Dirk Ostwald
    
    """
    delta = f[1] - f[0]                                                         # step size
    return [f[0] - delta/2, f[-1] + delta/2]
    


def sma_imagesc(ax, A, x, y, zmin, zmax, xticks, yticks, cmap, cfs):
    
    """ 
    This function creates 2D array visualizations similar to Matlab's imagesc
    function.
    
    Inputs:
        ax      : mpl.axes.Axes instance to which the image is plotted
        A       : 2D array to be visualized
        x       : x-axis coordinates (array column labels)
        y       : y-axis cooridnates (array row labels)
        xticks  : x-axis ticks
        yticks  : y-axis ticks 
        zmin    : z-axis max
        zmax    : z-axis min
        cmap    : colormap
        cfs     : colorbar fontsize
        
    Outputs:
        im      : image object
        cbar    : colorbar object
    
    Author - Dirk Ostwald
    
    """
    
    # visualization
    im      = ax.imshow(A,                                                      # array visualization
                        aspect          = 'auto',                               # rectangular pixels  
                        interpolation   = 'none',                               # no interpolation
                        extent          = extents(x) + extents(y),              # appropriate centering  
                        origin          = 'lower',                              # appropriate orientation  
                        vmin            = zmin,                                 # imshow vmin
                        vmax            = zmax,                                 # imshow vmax
                        cmap            = cmap)                                 # colormap                                                                       
    
    # image annotations
    ax.set_xticks(xticks)                                                       # x ticks of interest only
    ax.set_yticks(yticks)                                                       # y ticks of interest only
    
    # colorbar
    cbar    = ax.figure.colorbar(im, ax = ax )                                  # colorbar visualization
    cbar.ax.tick_params(labelsize = cfs)                                        # colorbar annotations
    
    return im, cbar