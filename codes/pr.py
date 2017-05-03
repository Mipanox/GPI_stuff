"""
Phase retrieval.
"""

import sys
sys.path.append("../codes/")

from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
from util import *
from zernike import *

def true_imgs(Npix,coeff1,coeff2,oversamp=1,
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
    - oversamp: see the `PR` class. Should be the same
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
    
    
    ## oversampling (zero-padding)
    npix = Pamp.shape[0]
    Npix = oversamp * npix
    if (Npix-npix) > 2:
        P_ = pad_array(P_,Npix,pad=0)  
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
          -- 'none'    : no support
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
        
        self.N_pix = foc.shape[0]
        self.npix  = self.N_pix / oversamp 
        #self.N_pix = self.npix * oversamp
        
        ## images
        self.foc_ = foc
        self.pup  = pup
        self.true_foc = true_foc
        self.true_pup = true_pup
        
        ## support
        self.supp = support
        supp_temp = self._gen_supp()
        
        self.foc = foc
        self.support = pad_array(supp_temp,self.N_pix,pad=1)
        
    def GS(self,init='random',threshold=None):
        """
        Original two-image Gerchberg-Saxton algorithm
        """
        if self.pup is None:
            raise NameError('Please provide pupil plane (object domain) intensity')
        
        pupil = self.pup
        focus = self.foc

        ## intensity to amplitude
        pup,foc = np.sqrt(pupil),np.sqrt(focus)
    
        ## initialize error and phase
        err = 1e10
        if init=='random':
            pha = np.random.random(pup.shape) * 2*np.pi
        elif init=='uniform':
            pha = np.ones(pup.shape)
        else:
            raise NameError('No such choice. Use "random" or "uniform"')
    
        ## initial states
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(pup,origin='lower')
        plt.title('Input Amplitude')
        plt.subplot(122); plt.imshow(pha,origin='lower')
        plt.title('Initial Phase'); plt.show()
    
        ##
        i = 1
        pup_sum = np.sum(pup)
        
        err_list = []
        
        if threshold is None: 
            ## iteration limit
            threshold = 1e-15
        while err > threshold:
            Fpup = fftshift(fft2(pup*np.exp(1j*pha)))
            pha  = np.arctan2(Fpup.imag,Fpup.real)
            Ifoc = projection(ifft2(ifftshift(foc*np.exp(1j*pha))), self.support)
            pha  = np.arctan2(Ifoc.imag,Ifoc.real)
            
            ## error (intensity) computed in pupil plane
            #-- defined as rms error / sum of true input image
            err =  np.sqrt(np.sum((abs(Ifoc)-pup)**2)) / pup_sum
            i += 1
            if i%100==0:
                print 'Current step : {0}'.format(i)
                print '        Error: {0:.2e}'.format(err)
        
            err_list.append(err)
        
            ## maximum iterations
            if i >= 500:
                break
            
        print 'Final step : {0}'.format(i)
        print 'Final Error: {0:.2e}'.format(err)
        
        return Ifoc, Fpup
        
        
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
        if self.supp == 'none':
            return np.zeros(Npix=self.npix)
        elif self.supp == 'circular':
            return Idxcmask(Npix=self.npix)
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