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

def pad_array(array,N_pix,pad=0):
    """
    Pad an array to size of N_pix x N_pix
    
    Parameters
    - pad: float
      The value to be padded
    """
    ### see numpy document: 
    #-- https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    def padwithtens(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = pad
        vector[-pad_width[1]:] = pad
        return vector
    
    ## assumed square shape
    padded = np.lib.pad(array, int((N_pix-array.shape[0])/2), padwithtens)
    return padded

def zoomArray(inArray, finalShape, sameSum=False, **zoomKwargs):
    """
    Reshape an array to a new size by upscaling first using interpolation
    and then block averaging to the desired size.
    
    Reference:
    - http://stackoverflow.com/questions/34122012/python-downsample-2d-numpy-array-by-a-non-integer-factor
    """
    from scipy.ndimage import zoom
    inArray = np.asarray(inArray, dtype = np.double)
    inShape = inArray.shape
    assert len(inShape) == len(finalShape)
    mults = []
    for i in range(len(inShape)):
        if finalShape[i] < inShape[i]:
            mults.append(int(np.ceil(inShape[i]/finalShape[i])))
        else:
            mults.append(1)
    tempShape = tuple([i * j for i,j in zip(finalShape, mults)])

    zoomMultipliers = np.array(tempShape) / np.array(inShape) + 0.0000001
    rescaled = zoom(inArray, zoomMultipliers, **zoomKwargs)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] / mult, mult] + sh[ind+1:]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis = ind+1)
    assert rescaled.shape == finalShape

    if sameSum:
        extraSize = np.prod(finalShape) / np.prod(inShape)
        rescaled /= extraSize
    return rescaled


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

