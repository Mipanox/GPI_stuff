"""
Functions for phase retrieval.
"""
import sys
sys.path.append("../codes/")

from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
from util import *
from zernike import *

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

def PR_ER(image,threshold=5e-3,init='random'):
    """
    Error reduction with full aperture. 
    First guess is determined by the amplitude of 
    IFTed image + `init` phase.
    
    Inputs
    - image: np.2darray
      The true "intensity" of the focal plane image.
    """
    ## intensity to amplitude
    img = np.sqrt(image)
    
    ## initialize error and phase
    err = 1e10
    if init=='random':
        pha = fullcmask(np.random.random(img.shape) * 2*np.pi)
    elif init=='uniform':
        pha = fullcmask(np.ones(img.shape))
    else:
        raise NameError('No such method. Use "random" or "uniform"')
    
    ## initial guess
    pup = fullcmask(ifft2(ifftshift(img*np.exp(1j*pha))))
    pup_ = abs(pup)
    
    ## initial states
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(pup_,origin='lower')
    plt.title('Initial guess Amplitude')
    plt.subplot(122); plt.imshow(pha,origin='lower')
    plt.title('Initial Phase'); plt.show()
    
    ##
    i = 1
    img_sum = np.sum(img)
    
    err_list = []
    while err > threshold:
        foc = fullcmask(fftshift(fft2(pup*np.exp(1j*pha))))
        pha = fullcmask(np.arctan2(foc.imag,foc.real))
        pup = fullcmask(ifft2(ifftshift(img*np.exp(1j*pha)))) ## Fourier constraint
        pha = fullcmask(np.arctan2(pup.imag,pup.real))
        
        ## error (intensity) computed in pupil plane
        #-- defined as rms error / sum of true input image
        err =  np.sqrt(np.sum((abs(foc)-img)**2)) / img_sum
        i += 1
        if i%100==0:
            print 'Current step : {0}'.format(i)
            print '        Error: {0:.2e}'.format(err)
        
        err_list.append(err)
        ## maximal iteration
        if i >= 2000:
            break
            
    print 'Final step : {0}'.format(i)
    print 'Final Error: {0:.2e}'.format(err)
        
    return pup, foc


##################################################################
def plot_true(pupil,focus):
    A = abs(pupil)
    Apha = np.arctan2(pupil.imag,pupil.real)
    B = abs(focus)
    Bpha = np.arctan2(focus.imag,focus.real)
    
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(A,origin='lower')
    plt.title('Amplitude - Pupil (linear)')
    plt.subplot(122); plt.imshow(B,origin='lower',norm=LogNorm())
    plt.title('Amplitude - Focal plane (log)')
    plt.colorbar();
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(Apha,origin='lower')
    plt.title('Phase - Pupil')
    plt.subplot(122); plt.imshow(Bpha,origin='lower')
    plt.title('Phase - Focal plane')
    plt.show()
    
def plot_recon(true_pup,true_foc,rec_pup_,rec_foc_):
    ## true
    A = abs(true_pup)
    Apha = np.arctan2(true_pup.imag,true_pup.real)
    B = abs(true_foc)
    Bpha = np.arctan2(true_foc.imag,true_foc.real)
    
    ## reconstructed
    rec_pup = abs(rec_pup_)
    rec_puppha = np.arctan2(rec_pup_.imag,rec_pup_.real)
    rec_foc = abs(rec_foc_)
    rec_focpha = np.arctan2(rec_foc_.imag,rec_foc_.real)
    
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(A,origin='lower')
    plt.title('Amplitude - True pupil image')
    plt.subplot(122); plt.imshow(rec_pup,origin='lower')
    plt.title('Amplitude - Reconstructed')
    plt.show()

    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(Apha,origin='lower')
    plt.title('Phase - True pupil image')
    plt.subplot(122); plt.imshow(rec_puppha,origin='lower')
    plt.title('Phase - Reconstructed')
    plt.show()
    
    ###
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(B,origin='lower',norm=LogNorm())
    plt.title('Amplitude - True focal image (log)'); plt.colorbar()
    plt.subplot(122); plt.imshow(rec_foc,origin='lower',norm=LogNorm())
    plt.title('Amplitude - Reconstructed (log)'); plt.colorbar()
    plt.show()

    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(Bpha,origin='lower')
    plt.title('Phase - True focal image'); plt.colorbar()
    plt.subplot(122); plt.imshow(rec_focpha,origin='lower')
    plt.title('Phase - Reconstructed'); plt.colorbar()
    plt.show()
    
def plot_phase_residual(tru_img,rec_img):
    """ Images in phasor form """
    tru_pha = np.arctan2(tru_img.imag,tru_img.real)
    rec_pha = np.arctan2(rec_img.imag,rec_img.real)
    
    residual = tru_pha - rec_pha
    
    plt.figure(figsize=(8,8))
    plt.imshow(abs(residual),origin='lower',cmap='gray')
    clb = plt.colorbar()
    clb.ax.set_title('rad')
    print 'rms residual error (in phase) = {0:.2f} %'.format(np.sqrt(np.sum(residual**2))/np.sum(tru_pha)*100)