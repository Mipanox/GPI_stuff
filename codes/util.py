"""
Utility functions not directly relevant for computations.
"""

def read_fits(filepath):
    with get_readable_fileobj(filepath, cache=True) as e:
        fitsfile = fits.open(e)
        data     = fitsfile[0].data
        header   = fitsfile[0].header
        
    return data, header

def grad_phase_x(bounds,N_pix):
    """
    Generate a uniform (tip/tilt) phase in the x-direction.
    
    Inputs:
    - bounds: tuple of floats
      The lower and upper bounds of the phase 'difference'
      at the edges of the aperture
      
    - N_pix: integer
      The desired size of the array
    """
    try:
        low, high = (bounds[0].to(u.rad)).value, \
                    (bounds[1].to(u.rad)).value
    except:
        raise TypeError("Please specify unit")
    
    gd = np.linspace(low,high,N_pix)
    return np.broadcast_to(gd,(N_pix,N_pix))

def pad_zeros(array,N_pix):
    """
    Pad an array to size of N_pix x N_pix
    """
    ### see numpy document: 
    #-- https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    def padwithtens(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector
    
    ## assumed square shape
    padded = np.lib.pad(pri_d, int((N_pix-array.shape[0])/2), padwithtens)
    return padded

############################
def expand_array(array):
    """
    Interpolate a even-sized array to the nearest odd size.
    Move the 'centroid' to (Nx-1)/2, (Ny-1)/2 in the new array
    """
    #from scipy.ndimage.interpolation import shift
    ## extend the array for one more row and column
    array = np.vstack([array,np.zeros(array.shape[1])])
    array = np.column_stack([array,np.zeros(array.shape[0])])
    
    ## centroid
    return shift(array,0.5)

