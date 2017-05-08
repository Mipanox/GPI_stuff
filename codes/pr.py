"""
Phase retrieval.
"""
from __future__ import division
import sys
sys.path.append("../codes/")

from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
from util import *
from zernike import *
from skimage.restoration import unwrap_phase

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
    else:
        raise ValueError('Oversampling rate should be larger')
    P  = abs(P_)**2
    
    ## Fourier domain image
    F_ = fftshift(fft2(P_))
    Famp = abs(F_)
    Fpha = np.arctan2(F_.imag,F_.real)
    F = Famp**2
    
    return P,P_,F,F_

def true_imgs_defocus(Npix,coeff1,coeff2,oversamp=1,
                      max_aberA=0.5,max_aberP=0.5,
                      defocus=10):
    """
    Generate true images (both domains) defocused
    and at focus for a given size in combined Zernike modes
    
    Inputs & Parameters
     see descriptions in `true_imgs_defocus`
    
    Parameters
    - defocus: float
      Degree of defocusing. Defined as the effective 
      focal point deviation from the focal point. 
      Unit in fraction of aperture diameter.
      Default is 10
      
    Returns
    - focused: list of np.2darray
      The pupil plane intensity, complex (complete phasor) 
      form, and same for focal plane images
    - defocused: list of np.2darray
      The defocused images. As for `focused`
    """
    ## Zernike
    ### defocusing
    coeff3 = np.copy(coeff2)
    coeff3[3] += defocus
    zerP = Zernike(coeff=coeff1,Npix=Npix)
    zerF = Zernike(coeff=coeff2,Npix=Npix)
    zerD = Zernike(coeff=coeff3,Npix=Npix)
    
    Pamp = abs(zerP.crCartAber(plot=False))
    Ppha = zerF.crCartAber(plot=False)
    Dpha = zerD.crCartAber(plot=False)
    
    #-- maximum
    Pamp *= max_aberA
    Ppha *= max_aberP
    
    Pamp += fullcmask(np.ones((Npix,Npix)))
    Ppha += fullcmask(np.ones((Npix,Npix)))
    Dpha += fullcmask(np.ones((Npix,Npix)))
    
    P_ = Pamp*np.exp(1j*Ppha)
    D_ = Pamp*np.exp(1j*Dpha)
    
    #-- oversampling (zero-padding)
    npix = Pamp.shape[0]
    Npix = oversamp * npix
    if (Npix-npix) > 2:
        P_ = pad_array(P_,Npix,pad=0)
        D_ = pad_array(D_,Npix,pad=0)
    else:
        raise ValueError('Oversampling rate should be larger')
    P = abs(P_)**2
    D = abs(D_)**2
    
    ## Fourier domain image
    F_  = fftshift(fft2(P_))
    F_d = fftshift(fft2(D_))
    
    Famp = abs(F_)
    D_da = abs(F_d)
    Fpha = np.arctan2(F_.imag,F_.real)
    F_dp = np.arctan2(F_d.imag,F_d.real)
    F  = Famp**2
    Fd = D_da**2
    
    ## focused
    focused = [P,P_,F,F_]
    defocused = [D,D_,Fd,F_d]
    
    return focused, defocused

#####################################################
class PR(object):
    def __init__(self,foc,pup=None,oversamp=1,
                 support='circular',
                 true_foc=None,true_pup=None):
        """
        Default to single-image case
        
        Inputs
        - foc: np.2darray
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
        self.npix  = self.N_pix / oversamp # original size
        
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
    
    def __call__(self,foc_defoc):
        """
        Only call this before running phase diversity methods
        
        Inputs
        - foc_defoc: list of np.2darray
          The focused and defocused Fourier domain images
        """
        if len(foc_defoc)!=2:
            raise ValueError('I need two images. The focused one first')
        
        self.foc_foc = foc_defoc[0]
        self.foc_def = foc_defoc[1]
    
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
            pha  = np.angle(Fpup)
            Ifoc,_ = projection(ifft2(ifftshift(foc*np.exp(1j*pha))), self.support)
            pha    = np.angle(Ifoc)
            
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
        
        
    def ER(self,init='random',cons_type='support',
           threshold=None,iterlim=500):
        """
        Error reduction with full aperture. 
    
        First guess is determined by the amplitude of 
        IFTed image + `init` phase.
    
        Object-domain constraints:
        - Support: defined by the aperture
        - Positivity: replace negative pixels with zeros
    
        Options
        - init: 'random' or 'uniform'
          Initial phase setting
        - cons_type: see `projection` method
        - threshold: float
          Error threshold. If `None`, iterations will stop 
          when `iterlim` is reached
        - iterlim: integer
          Maximum number of iterations
        """
        if self.pup is not None:
            print 'Caution: Pupil image is not used for constraints.'
            print '         This is one-image process.'
            
        ## intensity to amplitude
        image = self.foc
        img = np.sqrt(image)
    
        ## initialize error and phase
        err = 1e10
        if init=='random':
            pha = np.random.random(img.shape) * 2*np.pi
        elif init=='uniform':
            pha = np.ones(img.shape)
        else:
            raise NameError('No such method. Use "random" or "uniform"')
    
        ## initial guess
        pup,_ = projection(ifft2(ifftshift(img*np.exp(1j*pha))), self.support)
        pup_  = abs(pup)
    
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
        if threshold is None: 
            ## iteration limit
            threshold = 1e-15
        while err > threshold:
            ## Object-domain constraints
            pup,_ = projection(pup,self.support,cons_type=cons_type)
            foc   = fftshift(fft2(pup))
            ## Fourier constraint
            fo2 = img * (foc/abs(foc))
            pup = ifft2(ifftshift(fo2)) 

            ## error (mag) computed in focal plane
            err =  np.sqrt(np.sum((abs(foc)-img)**2)) / img_sum
            i += 1
            if i%100==0:
                print 'Current step : {0}'.format(i)
                print '        Error: {0:.2e}'.format(err)
            
            err_list.append(err)
            ## maximal iteration
            if i >= iterlim:
                break
        print '-----------------------'
        print 'First iteration error: {0:.2e}'.format(err_list[0])
        print 'Final step : {0}'.format(i)
        print 'Final Error: {0:.2e}'.format(err)
        
        return pup, foc, err_list

    def HIO(self,beta,init='random',cons_type='support',
           threshold=None,iterlim=500):
        """
        Hybrid Input-Output algorithm
        First guess is determined by the amplitude of 
        IFTed image + `init` phase.
      
        Parameters
        - beta: float
          The scaling of HIO correction.
      
        Options
         see descriptions in `ER`
        """
        if self.pup is not None:
            print 'Caution: Pupil image is not used for constraints.'
            print '         This is one-image process.'
            
        ## intensity to amplitude
        image = self.foc
        img = np.sqrt(image)
    
        ## initialize error and phase
        err = 1e10
        if init=='random':
            pha = np.random.random(img.shape) * 2*np.pi
        elif init=='uniform':
            pha = np.ones(img.shape)
        else:
            raise NameError('No such method. Use "random" or "uniform"')
    
        ## initial guess
        pup,_ = projection(ifft2(ifftshift(img*np.exp(1j*pha))), self.support)
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
        if threshold is None: 
            ## iteration limit
            threshold = 1e-15
        while err > threshold:
            pup_old = pup
            
            foc = fftshift(fft2(pup))
            ## Fourier constraint, update 'inside support'
            fo2 = img * (foc/abs(foc))
            pup = ifft2(ifftshift(fo2)) 
            
            pu2,mask = projection(pup,self.support,cons_type=cons_type)

            ## HIO
            pup[mask] = pup_old[mask]-beta*pup[mask]
        
            ## error (mag) computed in focal plane
            err =  np.sqrt(np.sum((abs(foc)-img)**2)) / img_sum
            i += 1
            if i%100==0:
                print 'Current step : {0}'.format(i)
                print '        Error: {0:.2e}'.format(err)
        
            err_list.append(err)
            ## maximal iteration
            if i >= iterlim:
                break
        print '-----------------------'
        print 'First iteration error: {0:.2e}'.format(err_list[0])
        print 'Final step : {0}'.format(i)
        print 'Final Error: {0:.2e}'.format(err)
        
        return pup, foc, err_list   
            
    def PD_ER(self,defocus,init='random',
              cons_type='support',
              threshold=None,iterlim=500,true_phasorP=None,true_phasorF=None):
        """
        Phase diversity with error reduction implementation
        Two images. One on focus the other out of focus
    
        See `ER` documenation for details.
        
        Inputs
        - defocus: float
          Degree of defocusing. Defined as the effective 
          focal point deviation from the focal point. 
          Unit in fraction of aperture diameter.
          No default. Should be the same as the true inputs
        """
        if self.pup is not None:
            print 'Caution: Pupil image is not used for constraints.'
            
        try:
            foc_foc = self.foc_foc
            foc_def = self.foc_def
        except:
            raise NameError('Please provide the Fourier domain images \
                             by calling the object')
            
        ## intensity to amplitude
        img_foc = np.sqrt(foc_foc)
        img_def = np.sqrt(foc_def)
    
        ## initialize error and phase
        err = 1e10
        
        #-- defocusing
        coeff = [0]*15
        coeff[3] += defocus
        
        zerD = Zernike(coeff=coeff,Npix=self.npix)
        Dpha = zerD.crCartAber(plot=False)
        Dpha = pad_array(Dpha,self.N_pix,pad=0)
        
        ### we don't care about 'defocusing' here
        if init=='random':
            pha_f = np.random.random(img_foc.shape) * 2*np.pi
            pha_d = np.random.random(img_foc.shape) * 2*np.pi
        elif init=='uniform':
            pha_f = np.ones(img_foc.shape)
            pha_d = np.ones(img_foc.shape)
        elif init=='test':
            pha_f = unwrap_phase(np.angle(true_phasorP)) + \
                    np.random.random(img_foc.shape)*1e-4
            pha_d = unwrap_phase(np.angle(true_phasorF)) + \
                    np.random.random(img_foc.shape)*1e-4
            iterlim = 1
        else:
            raise NameError('No such method. Use "random" or "uniform"')
    
        ## initial guess
        pup_f,_ = projection(ifft2(ifftshift(img_foc*np.exp(1j*pha_f))), self.support)
        pup_d,_ = projection(ifft2(ifftshift(img_def*np.exp(1j*pha_d))), self.support)
        pup_f_ = abs(pup_f)
        pup_d_ = abs(pup_d)
        
        ## initial states
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(pup_f_,origin='lower'); plt.colorbar()
        plt.title('Initial guess Amplitude')
        plt.subplot(122); plt.imshow(pha_f,origin='lower'); plt.colorbar()
        plt.title('Initial guess Phase'); plt.show()
    
        ##
        i = 1
        img_sum = np.sum(img_foc)
    
        err_list = []
        if threshold is None: 
            ## iteration limit
            threshold = 1e-15
        while err > threshold:
            ## Object-domain constraints
            pup_f,_ = projection(pup_f,self.support,cons_type=cons_type)
            pup_d,_ = projection(pup_d,self.support,cons_type=cons_type)
            foc_f   = fftshift(fft2(pup_f))
            foc_d   = fftshift(fft2(pup_d))
            ## Fourier constraint
            fo_f2 = img_foc * (foc_f/abs(foc_f))
            fo_d2 = img_def * (foc_d/abs(foc_d))
            pup_f = ifft2(ifftshift(fo_f2)) 
            pup_d = ifft2(ifftshift(fo_d2))
            
            #--- refocusing
            pup_d_pha = np.angle(pup_d)
            pup_d_ref = abs(pup_d)*np.exp(1j*(pup_d_pha-Dpha))
            
            ## averaging
            pup_f = (abs(pup_f)+abs(pup_d_ref))/2 * \
                    np.exp(1j*((np.angle(pup_d_ref)+np.angle(pup_f))/2))
            
            ## error (mag) computed in focal plane
            err =  np.sqrt(np.sum((abs(foc_f)-img_foc)**2)) / img_sum
            
            if i%100==0:
                print 'Current step                    : {0}'.format(i)
                print 'Error (of focused Fourier plane): {0:.2e}'.format(err)
            
            err_list.append(err)
            
            #--- defocusing
            pup_f_pha = np.angle(pup_f)
            pup_d     = abs(pup_f)*np.exp(1j*(pup_f_pha+Dpha))
            
            ## maximal iteration
            if i >= iterlim:
                break
            
            i += 1
                
        print '-----------------------'
        print 'First iteration error: {0:.2e}'.format(err_list[0])
        print 'Final step : {0}'.format(i)
        print 'Final Error: {0:.2e}'.format(err)
        
        pup_f_proj,_ = projection(pup_f,self.support,cons_type=cons_type)
        return pup_f, foc_f, err_list, pup_f_proj
    
    
    
    #############################        
    def _gen_supp(self):
        if self.supp == 'none':
            return np.zeros(Npix=self.npix)
        elif self.supp == 'circular':
            return Idxcmask(Npix=self.npix)
        else:
            raise NameError('No such support type')

############################################################################
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
        return arr, support
    
    elif cons_type=='realpos':
        a_real = arr.real
        mask = a_real<0
        arr[mask] = 0
        return arr, mask
    
    elif cons_type=='comppos':
        a_real = arr.real
        a_imag = arr.imag
        mask = np.logical_and(a_real<0,a_imag<0)
        arr[mask] = 0
        return arr, mask
    
def plot_recon(true_pup,true_foc,rec_pup_,rec_foc_,mod2pi=False):
    """
    Juxtaposing true/reconstructed amplitude/phase images
    
    Inputs
    - true_pup, true_foc: np.2darrays
      True pupil (object-domain) and focal plane (Fourier-domain) images.
      In complex form
    - rec_pup_, rec_foc_: np.2darrays
      Reconstructed pupil (object-domain) and focal plane (Fourier-domain) images.
      In complex form. Obtained via the `PR` class
    """
    ## true
    A = abs(true_pup)
    Apha = unwrap_phase(np.angle(true_pup))
    B = abs(true_foc)
    Bpha = unwrap_phase(np.angle(true_foc))
    
    ## reconstructed
    rec_pup = abs(rec_pup_)
    rec_puppha = unwrap_phase(np.angle(rec_pup_))
    rec_foc = abs(rec_foc_)
    rec_focpha = unwrap_phase(np.angle(rec_foc_))
    
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(A,origin='lower')
    plt.title('Amplitude - True pupil image'); plt.colorbar()
    plt.subplot(122); plt.imshow(rec_pup,origin='lower')
    plt.title('Amplitude - Reconstructed'); plt.colorbar()
    plt.show()
    
    if mod2pi==False:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(Apha,origin='lower')
        plt.title('Phase - True pupil image'); plt.colorbar()
        plt.subplot(122); plt.imshow(rec_puppha,origin='lower')
        plt.title('Phase - Reconstructed'); plt.colorbar()
        plt.show()
    else:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(np.mod(Apha,2*np.pi),origin='lower')
        plt.title('Phase - True pupil image'); plt.colorbar(); plt.clim(0,2*np.pi)
        plt.subplot(122); plt.imshow(np.mod(rec_puppha,2*np.pi),origin='lower')
        plt.title('Phase - Reconstructed'); plt.colorbar(); plt.clim(0,2*np.pi)
        plt.show()
    
    ###
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(B**2,origin='lower')#,norm=LogNorm())
    plt.xlim(220,292); plt.ylim(220,292)
    plt.title('Intensity - True focal image'); plt.colorbar()
    plt.subplot(122); plt.imshow(rec_foc**2,origin='lower')#,norm=LogNorm())
    plt.xlim(220,292); plt.ylim(220,292)
    plt.title('Intensity - Reconstructed'); plt.colorbar()
    plt.show()

    if mod2pi==False:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(Bpha,origin='lower')
        plt.title('Phase - True focal image'); plt.colorbar()
        plt.subplot(122); plt.imshow(rec_focpha,origin='lower')
        plt.title('Phase - Reconstructed'); plt.colorbar()
        plt.show()
    else:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(np.mod(Bpha,2*np.pi),origin='lower')
        plt.title('Phase - True focal image'); plt.colorbar()
        plt.subplot(122); plt.imshow(np.mod(rec_focpha,2*np.pi),origin='lower')
        plt.title('Phase - Reconstructed'); plt.colorbar()
        plt.show()

def plot_phase_residual(true_pup,true_foc,rec_pup_,rec_foc_):
    ## true
    Apha = np.angle(true_pup)
    Bpha = np.angle(true_foc)
    
    ## reconstructed
    rec_puppha = np.angle(rec_pup_)
    rec_focpha = np.angle(rec_foc_)
    
    pup_diff = unwrap_phase(Apha-rec_puppha)
    foc_diff = unwrap_phase(Bpha-rec_focpha)
    ### "Difference" of phases
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(pup_diff,origin='lower')
    plt.title('Pupil plane phase diff.')
    clb = plt.colorbar(); clb.ax.set_title('rad')
    plt.subplot(122); plt.imshow(foc_diff,origin='lower')
    plt.title('Focal plane phase diff.')
    clb = plt.colorbar(); clb.ax.set_title('rad')
    plt.show()
    
def plot_errlist(errlist,logy=False,loglog=True):
    """ 
    Plot the evolution of error (convergence) 
    
    Inputs
    - errlist: list of floats
      The recorded error from one of the PR algorithms
      
    Options
    - logy: boolean
      Plot in linear-log scale. Default False
    - loglog: boolean
      Plot in log-log scale. If `True`, ignore `logy`
      Defaults to True
    """
    
    plt.figure(figsize=(12,8))
    plt.plot(errlist,'b',lw=10)
    if logy==True:
        plt.yscale('log')
    if loglog==True:
        plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Iteration (#)'); plt.ylabel('Rms error (fraction)')
    plt.show()