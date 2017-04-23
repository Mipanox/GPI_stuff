"""
Codes for simulator of optics. 
Fourier optics for APLC optical path in 4 bands:
 - Primary/secondary mirror
 - Apodizer and gratings
 - Focal plane mask
 - Lyot stop
Bands: 
 - Y band : 1.02 um
 - J band : 1.22 um
 - H band : 1.65 um
 - K band : 2.19 um
 
// Note: APLC stands for "Apodized Pupil Lyot Coronagraph"
 
(Last updated: 04/22/2017)
"""
from __future__ import division
import sys
sys.path.append("../codes/")

import time
import numpy as np
import astropy.units as u
from util import *
from func import *

import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

class APLC_sim(object):
    
    def __init__(self,wavelength,pnt_contrast=1e-5,pnt_sep=0.6*u.arcsec):
        self.wavelength = wavelength
        
        ## contrast for planet
        self.pnt_contrast = pnt_contrast
        
        ## x tilt. Angular separation of planet
        try:
            self.pnt_sep = pnt_sep.to(u.arcsec)
        except:
            raise ValueError('Please specify unit')

        
                 
    def setPath(self,pri_path,apo_path,fpm_path,lyo_path):
        """
        Read in fits files and prepare for various masks
        and set up parameters for subsequent computation.
        """
        pri_d_, pri_h = read_fits(pri_path)
        apo_d_, apo_h = read_fits(apo_path)
        fpm_d_, fpm_h = read_fits(fpm_path)
        lyo_d_, lyo_h = read_fits(lyo_path)
        
        ## resize the masks
        self.pri_d = pri_d_[::4,::4]
        self.apo_d = apo_d_[::4,::4]
        self.fpm_d = fpm_d_[::2,::2]
        self.lyo_d = lyo_d_[::4,::4]

        pNaxis, aNaxis, fNaxis, lNaxis = pri_h['NAXIS1']/4 * u.m, \
                                         apo_h['NAXIS1']/4 * u.m, \
                                         fpm_h['NAXIS1']/2 * u.arcsec, \
                                         lyo_h['NAXIS1']/4 * u.m
        p_pixs, aNaxis, f_pixs, l_pixs = pri_h['pixscale']*4 * u.m, \
                                         apo_h['pixscale']/4 * u.arcsec, \
                                         fpm_h['pixscale']*2 * u.arcsec, \
                                         lyo_h['pixscale']*4 * u.m
        
        self.pNaxis = pNaxis.value
        
        ## parameters
        self._setParameters(f_pixs,p_pixs)
        
        ## pad 
        self.pri_p = pad_array(self.pri_d,self.N_pix)
        self.apo_p = pad_array(self.apo_d,self.N_pix)
        self.fpm_p = pad_array(self.fpm_d,self.N_pix)
        self.lyo_p = pad_array(self.lyo_d,self.N_pix)
        
    def run(self):
        One = np.ones((self.N_pix,self.N_pix))
        Zer = np.zeros((self.N_pix,self.N_pix))
        
        ## Initial
        #-- star
        Str_amp = One * self.pri_p
        Str_pha = Zer * self.pri_p * u.rad

        #-- planet
        Pnt_amp = One * self.pri_p * self.pnt_contrast**0.5
        
        p_shift = (self.D_tel*self.pnt_sep).to(u.nm*u.rad).value
        tilt    =  (p_shift*u.nm/self.wavelength).to(u.dimensionless_unscaled) \
                  * 2*np.pi * u.rad
        
        Pnt_pha = self.pri_p * pad_array(grad_phase_x((0*u.rad,tilt),self.pNaxis),self.N_pix) * u.rad
        
        ###
        self.Str_inc_ef = Str_amp*np.exp(1j*Str_pha.value)
        self.Pnt_inc_ef = Pnt_amp*np.exp(1j*Pnt_pha.value)
        
        ## Apodizer
        self.Str_apo_ef = self.Str_inc_ef * self.apo_p
        self.Pnt_apo_ef = self.Pnt_inc_ef * self.apo_p
        
        ## First image plane
        tt_fi_0 = time.time()
        self.Str_ffoc_ef = Ef_aft_from_Ef(self.Str_apo_ef)
        self.Pnt_ffoc_ef = Ef_aft_from_Ef(self.Pnt_apo_ef)
        tt_fi_1 = time.time()
        print 'Done first image plane. It took me {0:.3f} s'.format(tt_fi_1-tt_fi_0)
        
        ## FPM
        self.Str_coro_ef = self.Str_ffoc_ef * self.fpm_p
        self.Pnt_coro_ef = self.Pnt_ffoc_ef * self.fpm_p
        
        ## Second pupil plane
        tt_sp_0 = time.time()
        self.Str_2pup_ef = Ef_ift_from_Ef(self.Str_coro_ef)
        self.Pnt_2pup_ef = Ef_ift_from_Ef(self.Pnt_coro_ef)
        tt_sp_1 = time.time()
        print 'Done second pupil plane. It took me {0:.3f} s'.format(tt_sp_1-tt_sp_0)
        
        ## Lyot stop
        self.Str_lyo_ef = self.Str_2pup_ef * self.lyo_p
        self.Pnt_lyo_ef = self.Pnt_2pup_ef * self.lyo_p
        
        ## Final image plane
        tt_ni_0 = time.time()
        self.Str_final_ef = Ef_aft_from_Ef(self.Str_lyo_ef)
        self.Pnt_final_ef = Ef_aft_from_Ef(self.Pnt_lyo_ef)
        tt_ni_1 = time.time()
        print 'Done final image plane. It took me {0:.3f} s'.format(tt_ni_1-tt_ni_0)
      
    #############
    def _setParameters(self,f_pixs,p_pixs):
        """
        setup the parameters. Called by `setPath`
        """
        ## parameters
        #-- pixel scales : conform to masks
        self.k_pix_arc = f_pixs
        self.p_pix_met = p_pixs

        #-- extension of the primary mask
        allowedx = np.where(self.pri_d==1)[0]
        self.D_tel = (allowedx.max()-allowedx.min())*self.p_pix_met

        #-- sampling frequency
        self.f_s = ((self.wavelength/self.D_tel/self.k_pix_arc).to(u.rad**-1)).value

        #-- required number of pixels
        npix_ = int((allowedx.max()-allowedx.min()) * self.f_s)
        if npix_%2:
            self.N_pix = npix_ + 1
        else: self.N_pix = npix_
       
        print 'N_pix is {0}'.format(self.N_pix)
        
        
        ## more parameters
        ## centroid
        self.ctx,self.cty = (self.N_pix-1)/2, (self.N_pix-1)/2

        ##
        self.kx_min,self.kx_max = ((0-self.ctx)*self.k_pix_arc).value, \
                                  ((self.N_pix-self.ctx)*self.k_pix_arc).value
        self.ky_min,self.ky_max = ((0-self.cty)*self.k_pix_arc).value, \
                                  ((self.N_pix-self.cty)*self.k_pix_arc).value
        self.klin = np.linspace(self.kx_min,self.kx_max,self.N_pix)

        self.px_min,self.px_max = ((0-self.ctx)*self.p_pix_met).value, \
                                  ((self.N_pix-self.ctx)*self.p_pix_met).value
        self.py_min,self.py_max = ((0-self.cty)*self.p_pix_met).value, \
                                  ((self.N_pix-self.cty)*self.p_pix_met).value
        self.plin = np.linspace(self.px_min,self.px_max,self.N_pix)
        
    #############
    def plot_mask(self):
        plt.figure(figsize=(24,24))
        plt.subplot(221); plt.title('Spiders'); plt.xlabel('m')
        plt.imshow(self.pri_d,origin='lower',
                   extent=(self.px_min,self.px_max,self.py_min,self.py_max))
        plt.subplot(222); plt.title('Apodizer'); plt.xlabel('m')
        plt.imshow(self.apo_d,origin='lower',
                   extent=(self.px_min,self.px_max,self.py_min,self.py_max),
                   cmap=plt.get_cmap('Oranges'))
        plt.subplot(223); plt.title('Focal plane mask'); plt.xlabel('arcsec')
        plt.imshow(self.fpm_d,origin='lower',
                   extent=(self.kx_min,self.kx_max,self.ky_min,self.ky_max),
                   cmap=plt.get_cmap('viridis'))
        plt.subplot(224); plt.title('Lyot stop'); plt.xlabel('m')
        plt.imshow(self.lyo_d,origin='lower',
                   extent=(self.px_min,self.px_max,self.py_min,self.py_max),
                   cmap=plt.get_cmap('winter'))
        
def plot_stage(star,planet,extent,limit,log=False,clim=None):
    S_int = abs(star)**2
    P_int = abs(planet)**2
    
    low, high = extent
    low_, high_ = limit
    
    if log==False:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.title('Star Intensity')
        plt.imshow(S_int,origin='lower',extent=(low,high,low,high)); 
        plt.xlim(low_,high_); plt.ylim(low_,high_); plt.colorbar(); plt.xlabel('m/arcsec') 
    
        plt.subplot(122); plt.title('Planet Phase (rad)')
        plt.imshow(P_int,origin='lower',extent=(low,high,low,high)); 
        plt.xlim(low_,high_); plt.ylim(low_,high_); plt.colorbar(); plt.xlabel('m/arcsec')
        
    else:
        if clim:
            plt.figure(figsize=(16,8))
            plt.subplot(121); plt.title('Star Intensity'); plt.xlabel('m/arcsec')
            plt.imshow(S_int,origin='lower',extent=(low,high,low,high),norm=LogNorm()); 
            plt.xlim(low_,high_); plt.ylim(low_,high_); plt.colorbar(); plt.clim(clim) 
    
            plt.subplot(122); plt.title('Planet Phase (rad)'); plt.xlabel('m/arcsec')
            plt.imshow(P_int,origin='lower',extent=(low,high,low,high),norm=LogNorm()); 
            plt.xlim(low_,high_); plt.ylim(low_,high_); plt.colorbar(); plt.clim(clim)
        else:
            plt.figure(figsize=(16,8))
            plt.subplot(121); plt.title('Star Intensity'); plt.xlabel('m/arcsec')
            plt.imshow(S_int,origin='lower',extent=(low,high,low,high),norm=LogNorm()); 
            plt.xlim(low_,high_); plt.ylim(low_,high_); plt.colorbar() 
    
            plt.subplot(122); plt.title('Planet Phase (rad)'); plt.xlabel('m/arcsec')
            plt.imshow(P_int,origin='lower',extent=(low,high,low,high),norm=LogNorm()); 
            plt.xlim(low_,high_); plt.ylim(low_,high_); plt.colorbar()

def plot_profile(what,extent,limit,clim=None):
    Amp = abs(what)
    Int = Amp**2
    
    Npix = Amp.shape[0]
    low, high = extent
    low_, high_ = limit
    
    lin = np.linspace(low,high,Npix)
    
    if clim:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(Int,origin='lower', \
                                     extent=(low,high,low,high),norm=LogNorm(), \
                                     cmap='Blues')
        plt.xlim(low_,high_); plt.ylim(low_,high_); plt.title('FPM (log)')
        plt.xlabel('m/arcsec'); plt.colorbar(); plt.clim(clim)

        plt.subplot(122); plt.plot(lin,Amp[int(Npix/2),:])
        plt.xlim(low_,high_); plt.title('Amp profile (log)')
        plt.xlabel('m/arcsec'); plt.yscale('log')
        
    else:
        plt.figure(figsize=(16,8))
        plt.subplot(121); plt.imshow(Int,origin='lower', \
                                     extent=(low,high,low,high),norm=LogNorm(), \
                                     cmap='Blues')
        plt.xlim(low_,high_); plt.ylim(low_,high_); plt.title('FPM (log)')
        plt.xlabel('m/arcsec'); plt.colorbar()

        plt.subplot(122); plt.plot(lin,Amp[int(Npix/2),:])
        plt.xlim(low_,high_); plt.title('Amp profile (log)')
        plt.xlabel('m/arcsec'); plt.yscale('log')