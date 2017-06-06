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
              max_aberA=0.2,max_aberP=0.2,
              noise_level=0.):
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
      amplitude and phase respectively.
      For phase, it's the fraction of a full wave
    """
    ## set up object domain (pupil plane) image
    zerP = Zernike(coeff=coeff1,Npix=Npix)
    zerF = Zernike(coeff=coeff2,Npix=Npix)
    
    Pam_ = zerP.crCartAber(plot=False)
    Pamp = abs(Pam_)
    Ppha = zerF.crCartAber(plot=False)

    #-- maximum
    Pam_ *= max_aberA/(np.max(Pamp)+1e-10)
    Ppha *= max_aberP/(np.max(Ppha)+1e-10) * 2*np.pi
    
    Pam_ += fullcmask(np.ones((Npix,Npix)))
    Ppha += fullcmask(np.ones((Npix,Npix)))
    
    P_ = Pam_*np.exp(1j*Ppha)
    
    
    ## oversampling (zero-padding)
    npix = Pamp.shape[0]
    Npix = 2 * oversamp * npix
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
    
    ## noise only in measured PSF
    #-- fraction of the peak
    Fpeak = np.max(F)
    gau_n = abs(Fpeak*np.random.randn(Npix,Npix)*noise_level)
    
    F += gau_n
    Famp = np.sqrt(F)
    F_ = Famp* F_/abs(F_)
    
    return P,P_,F,F_

def true_imgs_defocus(Npix,coeff1,coeff2,oversamp=1,
                      max_aberA=0.2,max_aberP=0.2,
                      defocus=0.1,
                      noise_level=0.):
    """
    Generate true images (both domains) defocused
    and at focus for a given size in combined Zernike modes
    !!! CAUTION: never let the coeff to be exactly zeros
    
    Inputs & Parameters
     see descriptions in `true_imgs_defocus`
    
    Parameters
    - defocus: float
      Degree of defocusing. Defined as the rms aberration
      in unit of wavelength. For example, defouces=1
      defines an rms WFE of 2pi rad. ~ P-V of 14 rad.
      
    Returns
    - focused: list of np.2darray
      The pupil plane intensity, complex (complete phasor) 
      form, and same for focal plane images
    - defocused: list of np.2darray
      The defocused images. As for `focused`
    """
    ## Zernike
    ### defocusing
    coeff3 = [0.]*35
    #--- rms defocusing in rad
    coeff3[3] += (defocus *np.sqrt(15)/3.*np.pi)
    zerP = Zernike(coeff=coeff1,Npix=Npix)
    zerF = Zernike(coeff=coeff2,Npix=Npix)
    zerD = Zernike(coeff=coeff3,Npix=Npix)
    
    Pam_ = zerP.crCartAber(plot=False)
    Pamp = abs(Pam_)
    Ppha = zerF.crCartAber(plot=False)
    Dpha = zerD.crCartAber(plot=False)
    
    #-- maximum
    Pam_ *= max_aberA/(np.max(Pamp)+1e-10)
    Ppha *= max_aberP/(np.max(Ppha)+1e-10) * 2*np.pi     
    
    Pam_ += fullcmask(np.ones((Npix,Npix)))
    Ppha += fullcmask(np.ones((Npix,Npix)))
    #-- "defocusing"
    Dpha += Ppha
    
    P_ = Pam_*np.exp(1j*Ppha)
    D_ = Pam_*np.exp(1j*Dpha)
    
    #-- oversampling (zero-padding)
    npix = Pamp.shape[0]
    Npix = 2 * oversamp * npix
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
    
    ## noise only in measured PSF
    #-- fraction of the peak
    Fpeak  = np.max(F)
    Fdpeak = np.max(Fd)
    gau_nF = abs(Fpeak *np.random.randn(Npix,Npix)*noise_level)
    gau_nD = abs(Fdpeak*np.random.randn(Npix,Npix)*noise_level)
    
    F  += gau_nF
    Fd += gau_nD
    
    Famp  = np.sqrt(F)
    Fdamp = np.sqrt(Fd)
    
    F_  = Famp * F_/abs(F_)
    F_d = Fdamp* F_d/abs(F_d)
    
    ## 
    focused = [P,P_,F,F_]
    defocused = [D,D_,Fd,F_d]
    
    return focused, defocused

#####################################################
class PR(object):
    def __init__(self,foc,pup=None,oversamp=1,
                 support='circular',
                 true_foc=None,true_pup=None):
        """
        Default to single-image case. In general,
        error is assessed in 'amplitude' not 'intensity'
        in the focal plane.
        
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
        self.npix  = self.N_pix / oversamp / 2 # original size
        
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
    
    def GS(self,init='random',threshold=None,iterlim=1000):
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
        i = 0
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
            if i >= iterlim:
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
            
            true_pup = np.sqrt(self.pup)
            tpup_sum = np.sum(true_pup)
            
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
    
        err_list,err_pup = [],[]
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
            
            if true_pup is not None:
                err_p = np.sqrt(np.sum((abs(pup)-true_pup)**2)) / tpup_sum
                err_pup.append(err_p)
                
            ## maximal iteration
            if i >= iterlim:
                break
        print '-----------------------'
        print 'First iteration error: {0:.2e}'.format(err_list[0])
        print 'Final step : {0}'.format(i)
        print 'Final Error: {0:.2e}'.format(err)
        
        return pup, foc, err_list, err_pup

    def HIO(self,beta,init='random',cons_type='support',
           threshold=None,iterlim=500,force_only_phase=False):
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
            
            true_pup = np.sqrt(self.pup)
            tpup_sum = np.sum(true_pup)
            
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
        if force_only_phase==True:
            pup = true_pup*np.exp(1j*pha)
        else:
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
    
        err_list,err_pup = [],[]
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
            
            ##
            if force_only_phase==True:                
                pup_amp = true_pup
                pup_pha = np.angle(pup)
                pup = true_pup*np.exp(1j*pup_pha)
        
            ## error (mag) computed in focal plane
            err =  np.sqrt(np.sum((abs(foc)-img)**2)) / img_sum
            
            if true_pup is not None:
                err_p = np.sqrt(np.sum((abs(pup)-true_pup)**2)) / tpup_sum
                err_pup.append(err_p)
            
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
        
        pup_proj,_ = projection(pup,self.support,cons_type=cons_type)
        return pup, foc, err_list, pup_proj, err_pup  
            
    def PD_ER(self,defocus,init='random',
              cons_type='support',
              threshold=None,iterlim=500,
              true_phasorP=None,true_phasorF=None,
              force_only_phase=False):
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
          
        Options
        - true_phasorP, true_phasorF: np.2darrays
          True complex arrays for pupil and focal plane images.
          Used when testing: `init='random'`
        - force_only_phase: boolean
          An additional constraint on pupil plane amplitude
          If True, the amplitude is set to be the true one.
          [Not sure how to do this (forcing to be uniform)]
          [when true image is unknown yet]
        """
        if self.pup is not None:
            print 'Caution: Pupil image is not used for constraints.'
            true_pup = np.sqrt(self.pup)
            tpup_sum = np.sum(true_pup)
            
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
        defocus *= 4*np.sqrt(15)/3*np.pi ## conversion to Z-coeff
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
        if force_only_phase==True:
            pup_f = true_pup*np.exp(1j*pha_f)
            pup_d = true_pup*np.exp(1j*pha_d)
        else:
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
    
        err_list,err_pup = [],[]
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
            pup_f =           ( abs(pup_f)     +abs(pup_d_ref)     )/2 * \
                        np.exp(1j*((unwrap_phase(np.angle(pup_f)) + \
                                    unwrap_phase(np.angle(pup_d_ref))/2)))
            
            ## forcing only phase aberration
            #-- i.e. another constraint on amplitude
            if force_only_phase==True:
                ### now (temporarily) set to the true amplitude
                pup_f_amp = true_pup
                pup_f_pha = np.angle(pup_f)
                pup_f = true_pup*np.exp(1j*pup_f_pha)
            
            ## error (mag) computed in focal plane
            err =  np.sqrt(np.sum((abs(foc_f)-img_foc)**2)) / img_sum
            
            if i%100==0:
                print 'Current step                    : {0}'.format(i)
                print 'Error (of focused Fourier plane): {0:.2e}'.format(err)
            
            err_list.append(err)
            
            if true_pup is not None:
                err_p = np.sqrt(np.sum((abs(pup_f)-true_pup)**2)) / tpup_sum
                err_pup.append(err_p)
            
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
        return pup_f, foc_f, err_list, pup_f_proj, err_pup
    
    def OSS(self,beta,alpha_par=None,
            init='random',cons_type='support',
            threshold=None,iterlim=2000,
            force_only_phase=False,
            true_phasorP=None):
        """
        Oversampling smoothness incorporating HIO.
        First guess is determined by the amplitude of 
        IFTed image + `init` phase.
      
        Parameters
        - beta: float
          The scaling of HIO correction. Recommended range is 0.5-1.0
        - alpha_par: list of numbers (length of 3)
          Parameters for Gaussian smoothing. They are respectively:
          The starting (largest) scale, the final (smallest) scale,
          and the number of steps at each scale.
          If `None` (default), taken to be [N_pix,1/N_pix,10]
      
        Options
         see descriptions in `ER` and `PD_ER`
         Note: due to the nature of scaling (alpha), 
         one cannot set threshold (keep it `None`)
        """
        if self.pup is not None:
            print 'Caution: Pupil image is not used for constraints.'
            print '         This is one-image process.'
            
            true_pup = np.sqrt(self.pup)
            tpup_sum = np.sum(true_pup)
        
        if threshold is not None: 
            print '-'*30
            print 'Cannot set error threshold. Changed to `None`'
            threshold = None
        
        ## smoothing functions
        chunk, filter_gauss = self._gen_alpha(alpha_par,iterlim)
        
        ## intensity to amplitude
        image = self.foc
        img = np.sqrt(image)
    
        ## initialize error and phase
        err = 1e10
        #--- uniform pupil intensity
        unf_pup = np.invert(self.support)*1
        if init=='random':
            coeff = np.random.random(35)
            zerI = Zernike(coeff=coeff,Npix=self.npix)
            Ipha = zerI.crCartAber(plot=False)
            Ipha = pad_array(Ipha,self.N_pix,pad=0)
            pup = unf_pup*np.exp(1j*Ipha)
        elif init=='uniform':
            pup = unf_pup*np.exp(1j*np.ones(img.shape))
        elif init=='test':
            Ipha = np.angle(true_phasorP) + np.random.random(img.shape)*1e-10
            pup = unf_pup*np.exp(1j*Ipha)
        else:
            raise NameError('No such method. Use "random" or "uniform"')
            
        pup_ = abs(pup)
    
        ## initial states
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(pup_,origin='lower')
        plt.title('Init. Amp. guess (pupil)')
        plt.subplot(122); plt.imshow(Ipha,origin='lower')
        plt.title('Init. Pha. guess (pupil)'); plt.show()
    
        #------------------------------
        i,itr = 0,0
        img_sum = np.sum(img)
    
        err_list,err_pup = [],[]
        
        if threshold is None: 
            ## iteration limit
            threshold = 1e-15
        while err > threshold:
            print 'Current filter:'
            plt.figure(figsize=(6,6))
            plt.imshow(filter_gauss[itr],cmap='gray',origin='lower'); 
            plt.clim(0,1); plt.colorbar(); plt.show()
            
            ## steps
            for j in range(chunk):
                pup_old = pup
                
                foc = fftshift(fft2(pup))
                ## Fourier constraint, update 'inside support'
                fo2 = img * (foc/abs(foc))
                pup = ifft2(ifftshift(fo2))
                
                pu2,mask = projection(pup,self.support,cons_type=cons_type)
                
                ## HIO
                pup[mask] = pup_old[mask]-beta*pup[mask]
                #--- filtering
                pup[mask] = ifft2(ifftshift(fftshift(fft2(pup))*filter_gauss[itr]))[mask]
                
                ## forcing only phase aberration
                #-- i.e. another constraint on amplitude
                if force_only_phase==True:
                    pup_pha = np.angle(pup)
                    pup = unf_pup*np.exp(1j*pup_pha)
                
                ## error (mag) computed in focal plane
                err_ =  np.sqrt(np.sum((abs(foc)-img)**2)) / img_sum
                if err_ < err:
                    ## updating best-fit
                    temp_best_pup = pup
                err = err_
                
                i += 1
                if i%100==0:
                    print 'Current iter. : {0}'.format(i)
                    print '        Error : {0:.2e}'.format(err)
        
                err_list.append(err)
                
                if true_pup is not None:
                    err_p = np.sqrt(np.sum((abs(pup)-true_pup)**2)) / tpup_sum
                    err_pup.append(err_p)
            
            ## new initial input for next step
            pup  = temp_best_pup
            itr += 1
            
            ## maximal iteration
            if i >= iterlim:
                break
        print '-----------------------'
        print 'First iteration error: {0:.2e}'.format(err_list[0])
        print 'Final iteration : {0}'.format(i)
        print 'Final Error: {0:.2e}'.format(err)
        
        pup_proj,_ = projection(pup,self.support,cons_type=cons_type)
        return pup, foc, err_list, pup_proj, err_pup

    
    def PD_ER_smoothing(self,defocus,alpha_par=None,
                        init='random',cons_type='support',
                        threshold=None,iterlim=2000,
                        level=1.,
                        true_phasorPf=None,true_phasorPd=None,
                        force_only_phase=False,
                        smoo_in=False):
        """
        Phase diversity with error reduction implementation
        and gradual smoothing in the Fourier domain
        Two images. One on focus the other out of focus
    
        At this point it keeps ER and the smoothing is done in
        the averaged pupil image
        
        See `PD_ER` and `OSS` documenations for other details.
        
        Inputs
        - defocus: float
          Degree of defocusing. See `true_imgs_defocus`
          
        Parameters
        - alpha_par: list of numbers (length of 3)
          Parameters for Gaussian smoothing. They are respectively:
          The starting (largest) scale, the final (smallest) scale,
          and the number of steps at each scale.
          If `None` (default), taken to be [N_pix,1/N_pix,10]
          
        - level: float
          Level of aberration for the initial guess.
          Can be taken as the same as the max_aber level
          
        Options
        - smoo_in: boolean
          Do the gradual smoothing in all the image plane,
          including within the support or not.
        """
        if self.pup is not None:
            print 'Caution: Pupil image is not used for constraints.'
            
            true_pup = np.sqrt(self.pup)
            tpup_sum = np.sum(true_pup)
            
        if threshold is not None: 
            print '-'*30
            print 'Cannot set error threshold. Changed to `None`'
            threshold = None
        
        print '=-'*20
        if smoo_in==False: print 'Gradual filtering done outside the support'
        else: print 'Gradual filtering done in the whole plane'
            
        ##     
        try:
            foc_foc = self.foc_foc
            foc_def = self.foc_def
        except:
            raise NameError('Please provide the Fourier domain images \
                             by calling the object')
        ## smoothing functions
        if alpha_par is None:
            chunk, filter_gauss = self._gen_alpha([self.N_pix,self.npix/4,10],iterlim)
        else:
            chunk, filter_gauss = self._gen_alpha(alpha_par,iterlim)
        
        #--------------------------
        ## intensity to amplitude
        img_foc = np.sqrt(foc_foc)
        img_def = np.sqrt(foc_def)
    
        ## initialize error and stuff
        err = 1e10
        #--- uniform pupil intensity
        unf_pup = np.invert(self.support)*1
        #-- defocusing
        defocus *= np.sqrt(15)/3.*np.pi ## conversion to Z-coeff
        coeff = [0.]*35
        coeff[3] += defocus
        
        zerD = Zernike(coeff=coeff,Npix=self.npix)
        Dpha = zerD.crCartAber(plot=False)
        Dpha = pad_array(Dpha,self.N_pix,pad=0)
        
        if init=='random' or init=='uniform':
            if init=='random': coeff = (np.random.random(35)*2-1) * level            
            else:              coeff = [0.]*35
            ## phases
            zerI = Zernike(coeff=coeff,Npix=self.npix)
            Ipha = zerI.crCartAber(plot=False)
            Ipha = pad_array(Ipha,self.N_pix,pad=0)
            
            coefd = np.copy(coeff)
            coefd[3] += defocus 
            zerd = Zernike(coeff=coefd,Npix=self.npix)
            dpha = zerd.crCartAber(plot=False)
            dpha = pad_array(dpha,self.N_pix,pad=0)
        
        elif init=='test':
            Ipha = unwrap_phase(np.angle(true_phasorPf)) + np.random.random(img_foc.shape)*1e-10
            dpha = unwrap_phase(np.angle(true_phasorPd)) + np.random.random(img_foc.shape)*1e-10
        else:
            raise NameError('No such method. Use "random" or "uniform"')
        
        ## initial guesses
        pup_f = unf_pup*np.exp(1j*Ipha)
        pup_d = unf_pup*np.exp(1j*dpha) ## also guessing uniform
        
        pup_f_ = abs(pup_f)
        pup_d_ = abs(pup_d)
        
        ## initial states
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(pup_f_,origin='lower'); plt.colorbar()
        plt.title('Init. Amp. guess (pupil)')
        plt.subplot(122); plt.imshow(Ipha,origin='lower'); plt.colorbar()
        plt.title('Init. Pha. guess (pupil)'); plt.show()
    
        ##
        i,itr = 0,0
        img_sum = np.sum(img_foc)
        
        err_list,err_pup = [],[]
        if threshold is None: 
            ## iteration limit
            threshold = 1e-15
        while err > threshold:
            print 'Current filter:'
            plt.figure(figsize=(6,6))
            plt.imshow(filter_gauss[itr],cmap='gray',origin='lower'); 
            plt.clim(0,1); plt.colorbar(); plt.show()
            
            err_step = 1e10
            for j in range(chunk):
                foc_f = fftshift(fft2(pup_f))
                foc_d = fftshift(fft2(pup_d))
                ## Fourier constraint
                fo_f2 = img_foc * np.exp(1j*(np.angle(foc_f)))#(foc_f/abs(foc_f))
                fo_d2 = img_def * np.exp(1j*(np.angle(foc_d)))#(foc_d/abs(foc_d))
                pup_f = ifft2(ifftshift(fo_f2)) 
                pup_d = ifft2(ifftshift(fo_d2))
            
                #--- refocusing
                pup_d_pha = np.angle(pup_d)
                pup_d_ref = abs(pup_d)*np.exp(1j*(pup_d_pha-Dpha))
                
                ## averaging
                pup_f =           ( abs(pup_f)     +abs(pup_d_ref)     )/2 * \
                        np.exp(1j*((unwrap_phase(np.angle(pup_f)) + \
                                    unwrap_phase(np.angle(pup_d_ref))/2)))
            
                ## ER
                pup_f,mask = projection(pup_f,self.support,cons_type=cons_type)
                
                #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
                #--- smoothing in "averaged" image 
                ## 
                if smoo_in==False:
                    pup_f[mask] = ifft2(ifftshift(fftshift(fft2(pup_f))*filter_gauss[itr]))[mask]
                else:
                    pup_f = ifft2(ifftshift(fftshift(fft2(pup_f))*filter_gauss[itr]))                    
                
                #--- defocusing
                pup_f_pha = np.angle(pup_f)
                pup_d     = abs(pup_f)*np.exp(1j*(pup_f_pha+Dpha))
                
                ## forcing only phase aberration
                #-- i.e. another constraint on amplitude
                if force_only_phase==True:
                    pup_f_pha = np.angle(pup_f)
                    pup_f = unf_pup*np.exp(1j*pup_f_pha)
                
                ## error (mag) computed in focal plane
                err_ =  np.sqrt(np.sum((abs(foc_f)-img_foc)**2)) / img_sum
                if err_ < err_step:
                    ## updating best-fit
                    temp_best_pup = pup_f
                    temp_best_pud = pup_d
                    err_step = err_
                err = err_
            
                if i%100==0:
                    print 'Current step                    : {0}'.format(i)
                    print 'Error (of focused Fourier plane): {0:.2e}'.format(err)
            
                err_list.append(err)
                
                if true_pup is not None:
                    err_p = np.sqrt(np.sum((abs(pup_f)-true_pup)**2)) / tpup_sum
                    err_pup.append(err_p)
                
                i += 1
               
            ## new initial input for next step
            pup_f = temp_best_pup
            pup_d = temp_best_pud
            itr += 1
            
            ## maximal iteration
            #if smoo_in==True:
            #    if i >= iterlim-200: break
            if i >= iterlim:
                break
                
        print '-----------------------'
        print 'First iteration error: {0:.2e}'.format(err_list[0])
        print 'Final step : {0}'.format(i)
        print 'Final Error: {0:.2e}'.format(err)
        
        pup_f_proj,_ = projection(pup_f,self.support,cons_type=cons_type)
        return pup_f, foc_f, err_list, pup_f_proj, err_pup
    
    #############################        
    def _gen_supp(self):
        if self.supp == 'none':
            return np.zeros(Npix=self.npix)
        elif self.supp == 'circular':
            return Idxcmask(Npix=self.npix)
        elif isinstance(self.supp, np.ndarray) and self.supp.ndim==2:
            print 'using user-provided support array'
            return self.supp
        else:
            raise NameError('No such support type')
            
    def _gen_alpha(self,alpha_par,iterlim):
        """ 
        Generator for gradual (linearly changing) smoothing
        Called by `OSS` and `PD_ER_smoothing`
        """
        if alpha_par==None:
            sca_lar = self.N_pix
            sca_sml = 1/self.N_pix
            alpha_steps = 10
        else:
            sca_lar = alpha_par[0]
            sca_sml = alpha_par[1]
            alpha_steps = alpha_par[2]
            
        alpha = np.linspace(sca_lar,sca_sml,alpha_steps)
        if iterlim%alpha_steps:
            raise ValueError('Steps for alpha must be a divisor of total number of iterations')
        chunk = int(iterlim/alpha_steps)
        
        ## filter
        grid_x,grid_y = np.ogrid[-(self.N_pix-1)/2:(self.N_pix+1)/2,
                                 -(self.N_pix-1)/2:(self.N_pix+1)/2]
        grid_r = np.sqrt(grid_x**2+grid_y**2)
        
        filter_gauss = []
        for alp in alpha:
            filter_gauss.append(np.exp(-0.5*(grid_r/alp)**2))
        
        return chunk,filter_gauss

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
    
def plot_recon(true_pup,true_foc,rec_pup_,rec_foc_,
               mod2pi=False,fint_vmin=36,fint_vmax=36,
               max_abrAmp=0.2,max_abrPha=0.2,
               recons_clim=False):
    """
    Juxtaposing true/reconstructed amplitude/phase images
    
    Inputs
    - true_pup, true_foc: np.2darrays
      True pupil (object-domain) and focal plane (Fourier-domain) images.
      In complex form
    - rec_pup_, rec_foc_: np.2darrays
      Reconstructed pupil (object-domain) and focal plane (Fourier-domain) images.
      In complex form. Obtained via the `PR` class
      
    Options
    - mod2pi: boolean
      Display the phases in 0--2pi or not
    - fint_vmin,fint_vmax: integers
      Where to zoom-in in the Fourier plane for intensity
      Given in pixels away from the center
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
    plt.clim(1-max_abrAmp,1+max_abrAmp)
    plt.title('Amplitude - True pupil image'); plt.colorbar()
    plt.subplot(122); plt.imshow(rec_pup,origin='lower')
    plt.title('Amplitude - Reconstructed'); plt.colorbar()
    if recons_clim==True:
        plt.clim(1-max_abrAmp,1+max_abrAmp)
    plt.show()
    
    max_abrPha *= 2*np.pi
    if mod2pi==False:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(Apha,origin='lower')
        plt.title('Phase - True pupil image'); plt.colorbar()
        plt.clim(1-max_abrPha,1+max_abrPha)
        plt.subplot(122); plt.imshow(rec_puppha,origin='lower')
        plt.title('Phase - Reconstructed'); plt.colorbar()
        if recons_clim==True:
            cr = int(rec_puppha.shape[0]/2)
            med = np.median(rec_puppha[cr-50:cr+50,cr-50:cr+50])
            plt.clim(med-max_abrPha,med+max_abrPha)
        plt.show()
    else:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(np.mod(Apha,2*np.pi),origin='lower')
        plt.title('Phase - True pupil image'); plt.colorbar(); plt.clim(0,2*np.pi)
        plt.subplot(122); plt.imshow(np.mod(rec_puppha,2*np.pi),origin='lower')
        plt.title('Phase - Reconstructed'); plt.colorbar(); plt.clim(0,2*np.pi)
        plt.show()
    
    ###
    crx,cry = (B.shape[0]-1)/2, (B.shape[1]-1)/2
    #----
    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(B**2,origin='lower')#,norm=LogNorm())
    plt.xlim(crx-fint_vmax,crx+fint_vmax); plt.ylim(cry-fint_vmax,cry+fint_vmax)
    plt.title('Intensity - True focal image'); plt.colorbar()
    plt.subplot(122); plt.imshow(rec_foc**2,origin='lower')#,norm=LogNorm())
    plt.xlim(crx-fint_vmax,crx+fint_vmax); plt.ylim(cry-fint_vmax,cry+fint_vmax)
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

def plot_phase_residual(true_pup,true_foc,
                        rec_pup_,rec_foc_,
                        clim=True):
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
    if clim==True:
        ## some range for clim
        cr = int(pup_diff.shape[0]/2)
        #plt.clim(pup_diff[cr,cr]-0.2,pup_diff[cr,cr]+0.2)
    clb = plt.colorbar(); clb.ax.set_title('rad')
    plt.subplot(122); plt.imshow(foc_diff,origin='lower')
    plt.title('Focal plane phase diff.')
    clb = plt.colorbar(); clb.ax.set_title('rad')
    plt.show()
    
def plot_errlist(errlist,errpuplist=None,logy=False,loglog=True):
    """ 
    Plot the evolution of error (convergence) 
    
    Inputs
    - errlist: list of floats
      The recorded error from one of the PR algorithms
    - errpuplist: list of floats
      The recorded error in the pupil plane. 
      This would be `None` if run in real situations
      
    Options
    - logy: boolean
      Plot in linear-log scale. Default False
    - loglog: boolean
      Plot in log-log scale. If `True`, ignore `logy`
      Defaults to True
    """
    if errpuplist is not None:
        plt.figure(figsize=(24,8))
        plt.subplot(121)
        plt.plot(errlist,'b',lw=10)
        if logy==True:
            plt.yscale('log')
        if loglog==True:
            plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Iteration (#)'); plt.ylabel('Rms error (fraction)')
        plt.title('Amp. error in focal plane')
        
        plt.subplot(122)
        plt.plot(errpuplist,'b',lw=10)
        if logy==True:
            plt.yscale('log')
        if loglog==True:
            plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Iteration (#)'); plt.ylabel('Rms error (fraction)')
        plt.title('Amp. error in pupil plane')
        plt.show()
    
    else:
        plt.figure(figsize=(12,8))
        plt.plot(errlist,'b',lw=10)
        if logy==True:
            plt.yscale('log')
        if loglog==True:
            plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Iteration (#)'); plt.ylabel('Rms error (fraction)')
        plt.show()