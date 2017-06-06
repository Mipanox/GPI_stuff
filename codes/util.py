"""
Utility functions not directly relevant for computations.
"""
from __future__ import division
import numpy as np

import sys
sys.path.append("../codes/")

from zernike import *
from skimage.restoration import unwrap_phase

def read_fits(filepath):
    from astropy.utils.data import get_readable_fileobj
    from astropy.io import fits
    with get_readable_fileobj(filepath, cache=True) as e:
        fitsfile = fits.open(e)
        data     = fitsfile[0].data
        header   = fitsfile[0].header
        
    return data, header

def grad_phase_x(bounds,Npix):
    import astropy.units as u
    """
    Generate a uniform (tip/tilt) phase in the x-direction.
    
    Inputs:
    - bounds: tuple of floats
      The lower and upper bounds of the phase 'difference'
      at the edges of the aperture
      
    - Npix: integer
      The desired size of the array.
      Should be identical to those of the masks
    """
    
    try:
        low, high = (bounds[0].to(u.rad)).value, \
                    (bounds[1].to(u.rad)).value
    except:
        raise TypeError("Please specify unit")
    
    gd = np.linspace(low,high,Npix)
    return np.broadcast_to(gd,(Npix,Npix))

def pad_array(array,N_pix,pad=0):
    """
    Pad an array to size of N_pix x N_pix
    
    Parameters
    - pad: float
      The value to be padded
    """
    if (N_pix-array.shape[0]) < 2: 
        raise ValueError('N_pix must be at least greater than the original array dimension by 2')
    ### see numpy document: 
    #-- https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    def padwith(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = pad
        vector[-pad_width[1]:] = pad
        return vector
    
    ## assumed square shape
    padded = np.lib.pad(array, int((N_pix-array.shape[0])/2), padwith)
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

def fullcmask(array,pad=0):
    """
    Variation of `cmask`, with radius equal to Npix.
    """
    nx,ny = array.shape
    if nx%2 or ny%2:
        raise ValueError('Array dimensions should be even')
    
    a , b = (nx-1)/2, (ny-1)/2 ## centroid
    y , x = np.ogrid[-a:nx-a,-b:ny-b]
    
    radius = a
    mask = x*x + y*y > radius**2
        
    arr = np.copy(array)
    arr[mask] = pad
    return arr

def Idxcmask(array=None,Npix=None,pad=0):
    """
    The mask indices for fullcmask
    """
    if array is not None:
        nx,ny = array.shape
    elif Npix is not None:
        nx,ny = Npix,Npix
    
    if nx%2 or ny%2:
        raise ValueError('Array dimensions should be even')
    
    a , b = (nx-1)/2, (ny-1)/2 ## centroid
    y , x = np.ogrid[-a:nx-a,-b:ny-b]
        
    radius = a
    mask = x*x + y*y > radius**2
    
    return mask

def ccmask(array,radius):
    """ Leaving only circular region within radius """
    nx,ny = array.shape
    if nx%2 or ny%2:
        raise ValueError('Array dimensions should be even')
    
    a , b = (nx-1)/2, (ny-1)/2 ## centroid
    y , x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y > radius**2
    
    arr = np.copy(array)
    arr[mask] = 0
    return arr

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

#############################
def ctr_mask(array,size,center,
             shape='circular'):
    """ 
    Masking images given coordinates of the center
    and the desired size (see below)
    
    Parameters:
    - size: float/integer
      The radius (circular) or length of side (square)
      Better be integer      
    - center: tuple of integers
      The pixel values of the center
      
    Options
    - shape: 'circular' or 'square'
      Shape of the mask. Default to be 'circular'
    """
    arr = np.copy(array)
    ##
    nx, ny = arr.shape
    ## center
    cty, ctx = center
    
    #-- circular
    if shape=='circular':
        x, y = np.ogrid[-ctx:nx-ctx,-cty:ny-cty]
        mask = x*x + y*y > (size/2)**2
    
    #-- square
    elif shape=='square':
        x, y = np.meshgrid(np.arange(-ctx,nx-ctx),np.arange(-cty,ny-cty))
        mask = np.logical_or(abs(x)>size,abs(y)>size)
        
    else: raise NameError('No such shape. Use "circular" or "square"')
        
    ##
    arr[mask] = 0
    return arr, mask

def clipping(array,Npix,size,center=None,allpos=True,**kwargs):
    """
    Clipping the array according to Npix 
    and specified center position
    Call the `ctr_mask` method to mask out unwanted region
    
    Parameters:
    - Npix: integer
      Should be even. The final desired dimension of the array
      
    Options:
    - allpos: boolean
      Force negatively-valued pixels to be zero
      In normal cases this should be True as negative intensity
      is not physical
    """
    if Npix%2: raise ValueError('Npix should be even')
    arr = np.copy(array)
        
    # center
    if center is None:
        cty, ctx = (arr.shape[0]-1)/2.,(arr.shape[1]-1)/2.
    else:
        cty, ctx = center
    
    center = cty,ctx
    ## masking
    masked,_ = ctr_mask(arr,size,center,**kwargs)
    
    #
    nx, ny = masked.shape
    
    ## clipping
    #-- obtain a small array first
    if (size%2 and ctx-int(ctx)!=0) or \
       (size%2==0 and ctx-int(ctx)==0):
        temp = masked[int(ctx-size/2):int(ctx+size/2),
                      int(cty-size/2):int(cty+size/2)]
    else:
        temp = masked[int(ctx-size/2):int(ctx+size/2)+1,
                      int(cty-size/2):int(cty+size/2)+1]
    #-- then pad it with zeros
    padded = pad_array(temp,Npix,pad=0)
    
    if allpos==True:
        padded[padded<0] = 0.
    return padded

##################################################
## Zernike decomposition                        ##
#   Ref: https://github.com/david-hoffman/pyOTF ##
##################################################
def zern_exp(Npix,m=15,oversamp=2):
    """
    Expansion in Zernike modes.
    Returns a 3-dimensional array in shape
     m x Npix x Npix
    
    Parameters:
    - m: integer
      How many modes. Defaults to 15
    
    Options
    - oversamp: integer
      The oversampling. Should be the same as 
      the data to be fitted.
    """
    def zern_padded(i,Npix=Npix,oversamp=oversamp):
        coef = np.zeros(35)
        coef[i-1] = 1
        z = Zernike(coeff=coef,Npix=int(Npix/2/oversamp))
        zz = z.crCartAber(plot=False)
        Zz = pad_array(zz,Npix,pad=0)
        return Zz
    
    return np.array([zern_padded(i) for i in range(1,m+1)])

def fit_to_zerns(data, zerns, mask, **kwargs):
    """
    Least-square fitting for Zernike coefficients.
    !!! Ignore the zeroth mode - piston !!!
    !!!  since DC offset is not physically meaningful
    
    Inputs
    - data, zerns: np.ndarrays
      The image to be fitted and the Zernike-expansion.
      See `zern_exp`
    - mask: np.2darray
      The ROI where the fitting is applied.
      Normally it is the support
    """
    data2fit = data[mask]
    zern2fit = zerns[:, mask].T
    
    ## least-square fitting. More robust than simple "dot-product"
    coefs, _, _, _ = np.linalg.lstsq(zern2fit, data2fit, **kwargs)
    return coefs

def wrap_up_zern_fit(obj,Recon_phasor,P_phasor=None,
                     oversamp=2,m=15,flip=False):
    """
    Wrap-up method for Zernike coefficient fitting
    
    Inputs
    - obj: class object
      The object storing the particular PR procedure
    - Recon_phasor:
      The phasor (complex) image of the reconstructed
      pupil image
      
    Options
    - P_phasor: 
      The phasor (complex) image of the true pupil image
      If provided, the resulting plot will juxtapose
      thes 'true' coefficients with the reconstructed ones
    - flip: boolean
      Flip and reverse the phase. Used only when comparing
      with true pupil image and in OSS
    """
    fit_corr=[]
    
    ## initialize
    mask = np.invert(obj.support) # support=True
    Npix = obj.N_pix
    
    fit_Z = zern_exp(Npix,m=m,oversamp=oversamp)
    
    ## data
    zer_reco = unwrap_phase(np.angle(Recon_phasor))
    if flip:
        zer_reco = np.fliplr(np.flipud(-zer_reco))
    
    fit_reco = fit_to_zerns(zer_reco,fit_Z,mask)
    
    ## plot
    plt.figure(figsize=(10,7))
    
    if P_phasor is not None:
        zer_corr = unwrap_phase(np.angle(P_phasor))
        fit_corr = fit_to_zerns(zer_corr,fit_Z,mask)
    
        plt.plot(range(1,15+1),[np.nan]+list(fit_corr[1:]), 'r-.+',label='True' ,ms=20,mew=3)
    ### Note: DC offset is neglected
    plt.plot(range(1,15+1),[np.nan]+list(fit_reco[1:]), 'b-.x',label='Best-fit of recon.',ms=20,mew=3)
    plt.xlim(0,15); plt.legend()
    plt.xlabel('Mode (Noll)'); plt.ylabel('Relative strength (a.u.)')
    
    return fit_reco,fit_corr

##################
## Data related ##
##################

class data_manage(object):
    """
    Class for data handling.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage.interpolation import rotate
    from astropy.io import fits
    
    def __init__(self,path,angle,cube=False):
        ##
        self.path = path
        if cube==False:
            self.data = fits.open(path)[0].data
            self.hdr  = fits.open(path)[0].header
        else:
            self.data = fits.open(path)[1].data
            self.hdr  = fits.open(path)[1].header
        
        ##
        self.angle = angle        
    
    def __call__(self,wavelength):
        wvl_ref = self.hdr['CRVAL3']
        wvl_stp = self.hdr['CD3_3']
        
        ## the closest slice
        idx = int((wavelength-wvl_ref)/wvl_stp)
        
        self.data_slc = self.data[idx]
        
    def rot_img(self,plot=True,pad=-100,
                xlim=None,ylim=None,**kwargs):
        """
        Rotating image plane to align the axes
        
        Options:
        - plot: boolean
          If `True`, plot the images (before and after)
          Can zoom in by specifying `xlim` and `ylim`
        - pad: number
          The value to pad the NaN values in the array
          for the sake of interpolation
        """
        ## padding
        if self.data.ndim==3:
            data_ori      = np.copy(self.data_slc)
            self.data_pad = np.copy(self.data_slc)
            self.data_pad[np.isnan(self.data_pad)] = pad
            
        else:        
            data_ori      = np.copy(self.data)
            self.data_pad = np.copy(self.data)
            self.data_pad[np.isnan(self.data_pad)] = pad
        
        ## 
        if self.angle != 0:
            self.data_rot = rotate(self.data_pad,
                                   angle=self.angle,
                                   **kwargs)
        else: 
            self.data_rot = np.copy(self.data_pad)
        
        if plot==True:
            plt.title('Original')
            plt.imshow(data_ori,origin='lower')
            plt.colorbar(); plt.show()
            
            plt.title('Rotated')
            plt.imshow(self.data_rot,origin='lower')
            if xlim is not None:
                plt.xlim(xlim); plt.ylim(ylim) 
            plt.show()
            
        return self.data_rot