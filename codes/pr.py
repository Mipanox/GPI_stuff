"""
Phase retrieval.
"""

import sys
sys.path.append("../codes/")

from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
from util import *
from zernike import *

def true_imgs(Npix,coeff1,coeff2,
              max_aberA=0.2,max_aberP=0.2):
    """
    Generate true images (both domains)
    for a given size in combined Zernike modes
    
    Inputs
    - Npix: integer
      Size of the images (assumed symmetric)
    - coeff1, coeff2: lists of floats
      Coefficients of the Zernike modes for the
      object and Fourier domain images respectively
      
    Parameters
    - max_aberA, max_aberP: positive floats
      Maximum variations of the aberration in
      amplitude and phase respectively
    """
    ## set up object domain (pupil plane) image
    zerP = Zernike(coeff=coeff1,Npix=Npix)
    zerF = Zernike(coeff=coeff2,Npix=Npix)
    
    Pamp = abs(zerP.crCartAber(plot=False))
    Ppha = zerF.crCartAber(plot=False)

    #-- maximum
    Pamp *= max_aberA
    Ppha *= max_aberP
    
    Pamp += fullcmask(np.ones((Npix,Npix)))
    Ppha += fullcmask(np.ones((Npix,Npix)))
    
    P_ = Pamp*np.exp(1j*Ppha)
    P  = abs(P_)**2

    ## Fourier domain image
    F_rec = np.fft.fftshift(np.fft.fft2(P_))
    F_ = F_rec
    Famp = abs(F_)
    Fpha = np.arctan2(F_.imag,F_.real)
    F = Famp**2
    
    return P,P_,F,F_

class PR(object):
    def __init__(self,foc,pup=None,oversamp=1,
                 support='circular',
                 true_foc=None,true_pup=None):
        """
        Default to single-image case
        
        Inputs
        - Foc: np.2darray
          The focal plane (Fourier domain) image
          Only intensity
          
        Parameters
        - oversamp: float (ideally integer)
          The oversampling (zero-padding) rate,
          defined by the total grid size divided
          by the image diameter
          
        Options
        - support: string
          Choice of support. The input image(s) should 
          have been masked by the same support.
          Defaults to 'circular'
          -- 'circular': circular aperture with diameter
                         the same size as the image
          -- 
        - pup: np.2darray
          The intensity of the pupil plane 
          (object domain) image. Only used when 
          running with the original Gerchberg-Saxton
          two-image algorithm
        - true_foc, true_pup: np.2darray
          The full (complex) TRUE images
          Displayed when calling `plot_recon`
          
        """
        
        self.npix  = foc.shape[0]
        self.N_pix = self.npix * oversamp
        
        ## images
        self.foc_ = foc
        self.pup  = pup
        self.true_foc = true_foc
        self.true_pup = true_pup
        
        ## support
        self.supp = support
        supp_temp = self._gen_supp()
        
        ## padding / oversampling
        if (self.N_pix-self.npix) > 2:
            self.foc     = pad_array(foc,self.N_pix)
            self.support = pad_array(supp_temp,self.N_pix)
            
        else:
            self.foc     = foc
            self.support = supp_temp
            print 'The image is not extended due to insufficient oversampling.'
        
    def GS(self):
        """
        Original two-image Gerchberg-Saxton algorithm
        """
        if self.pup is None:
            raise NameError('Please provide pupil plane (object domain) intensity')
        
        
        
    def ER(self):
        if self.pup is not None:
            print 'Caution: Pupil image is not used for constraints.'
            print '         This is one-image process.'
        
    def HIO(self):
        if self.pup is not None:
            print 'Caution: Pupil image is not used for constraints.'
            print '         This is one-image process.'
            
    #############################        
    def _gen_supp(self):
        if self.supp == 'circular':
            return Idxcmask(self.foc_)
        else:
            raise NameError('No such support type')
            
def projection(inarray,support,cons_type='support',pad=0):
    """
    Handling the support mask
    Will apply mask and/or constraints onto the input array
    according to the type of constraints demanded
    
    Inputs
    - inarray: np.2darray
      Input array
    - support: np.2darray
      The support mask. Should be valued 0 (or FALSE) in the support
      and 1 (or TRUE) outside
    
    Options
    - cons_type: string
      Type of constraints in addition to the defining region of support
      -- 'support': do nothing
      -- 'realpos': real positive. Used when e.g. the true pupil image
                    is a photograph which also has no phase
      -- 'comppos': complex positive. Both real and imaginary components
                    are positive. Rarely used.
    """
    ## apply support
    arr = np.copy(inarray)
    arr[support] = pad
    
    if cons_type=='support':
        return arr
    
    elif cons_type=='realpos':
        a_real = arr.real
        arr[a_real<0] = 0
        return arr
    
    elif cons_type=='comppos':
        a_real = arr.real
        a_imag = arr.imag
        arr[np.logical_and(a_real<0,a_imag<0)] = 0
        return arr