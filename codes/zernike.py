"""
Generator for Zernike polynomials (first 15)
Can either call from polar or Cartesian coordinates.

References
- Wikipedia: https://en.wikipedia.org/wiki/Zernike_polynomials
- `opticspy` package: https://github.com/Sterncat/opticspy
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

class Zernike(object):
    """
    First 15 Zernike polynomials
    The index convention follows the noll index
    """
    
    def __init__(self,Z1=0,Z2=0,Z3=0,Z4=0,Z5=0,
                      Z6=0,Z7=0,Z8=0,Z9=0,Z10=0,
                      Z11=0,Z12=0,Z13=0,Z14=0,Z15=0,Npix=1024,
                 coeff=None):
        ## coefficients
        if coeff is not None:
            if len(coeff) != 15:
                raise IndexError('Length of coefficient list should be 15')
            self.coeff = coeff
        else:
            self.coeff = [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, 
                          Z9, Z10, Z11, Z12, Z13, Z14, Z15]
        ## grid size
        self.Npix = Npix
        
    def crPolarAber(self,plot=True):
        """
        Combined aberration in polar coordinates
        
        Options
        - plot: boolean
          Plot the result. Defaults to `True`
        """
        coeff = self.coeff
        Npix  = self.Npix
        
        ##
        the = np.linspace(0, 2*np.pi, Npix)
        rho = np.linspace(0, 1, Npix)
        [u,r] = np.meshgrid(the,rho)
        
        Z_cb = self.ZernikePolar(r,u)
        
        if plot==True:
            X = r*np.cos(u)
            Y = r*np.sin(u)
            
            plt.figure(figsize=(10,10))
            plt.pcolormesh(X, Y, Z_cb)
            
        return Z_cb
    
    def crCartAber(self,plot=True):
        """
        Combined aberration in Cartesian coordinates
        
        Options
        - plot: boolean
          Plot the result. Defaults to `True`
        """
        coeff = self.coeff
        Npix  = self.Npix
        
        ##
        X, Y = np.meshgrid(np.linspace(-1, 1, Npix),
                           np.linspace(-1, 1, Npix))
        
        Z_cb = self.ZernikeCartesian(X,Y)
        Z_ap = self.cmask(Z_cb)
        
        if plot==True:
            plt.figure(figsize=(10,10))
            plt.pcolormesh(X, Y, Z_ap)
            
        return Z_ap
    
    
    #################    
    def ZernikePolar(self,r,theta):
        """
        Normalized Zernike polynomials in a circular aperture.
          Normalization is defined such that the 
          square integral (i.e. Z_j^2) over the aperture is pi

        Inputs:
        - coeff: list
          The coefficient of the polynomials (noll)
        - r, u: numpy 2d arrays
          Radial and angular coordinate in polar coordinates.

        Returns:
        - Z_comb: numpy 2d array
          Combined aberration
        """
        coeff = list(self.coeff)
        
        Z = [0] + coeff
        Z1  =  Z[1]  * 1*(np.cos(theta)**2 + np.sin(theta)**2)
        Z2  =  Z[2]  * 2*r*np.cos(theta)
        Z3  =  Z[3]  * 2*r*np.sin(theta)
        Z4  =  Z[4]  * np.sqrt(3)*(2*r**2 - 1)
        Z5  =  Z[5]  * np.sqrt(6)*r**2 * np.sin(2*theta)
        Z6  =  Z[6]  * np.sqrt(6)*r**2*np.cos(2*theta)
        Z7  =  Z[7]  * np.sqrt(8)*(3*r**2 - 2) * r*np.sin(theta)
        Z8  =  Z[8]  * np.sqrt(8)*(3*r**2 - 2) * r*np.cos(theta)
        Z9  =  Z[9]  * np.sqrt(8)*r**3*np.sin(3*theta)
        Z10 =  Z[10] * np.sqrt(8)*r**3*np.cos(3*theta)
        Z11 =  Z[11] * np.sqrt(5)*(1-6*r**2 + 6*r**4)
        Z12 =  Z[12] * np.sqrt(10)*(4*r**2-3)*r**2 * np.cos(2*theta)
        Z13 =  Z[13] * np.sqrt(10)*(4*r**2-3)*r**2 * np.sin(2*theta)
        Z14 =  Z[14] * np.sqrt(10)*r**4 * np.cos(4*theta)
        Z15 =  Z[15] * np.sqrt(10)*r**4 * np.sin(4*theta)

        Z_comb = Z1 + Z2  +  Z3 +  Z4 +  Z5 +  Z6 +  Z7 +  Z8 + \
                 Z9 + Z10 + Z11 + Z12 + Z13 + Z14 + Z15
    
        return Z_comb

    def ZernikeCartesian(self,x,y):
        """
        Zernike polynomials caculation in Cartesian coordinates
        Normalized in a circular aperture only

        Inputs:
        - coeff: list
          The coefficient of the polynomials (noll)
        - x, y: numpy 2d arrays
          X and Y coordinate (meshgrid)

        Returns:
        - Z_comb: numpy 2d array
          Combined aberration
        """
        coeff = list(self.coeff)
        
        Z = [0] + coeff
        r2 = x**2 + y**2
        Z1  =  Z[1]  * 1
        Z2  =  Z[2]  * 2*x
        Z3  =  Z[3]  * 2*y
        Z4  =  Z[4]  * np.sqrt(3)*(2*r2 - 1)
        Z5  =  Z[5]  * 2*np.sqrt(6)*x*y
        Z6  =  Z[6]  * np.sqrt(6)*(x**2 - y**2)
        Z7  =  Z[7]  * np.sqrt(8)*y*(3*r2 - 2)
        Z8  =  Z[8]  * np.sqrt(8)*x*(3*r2 - 2)
        Z9  =  Z[9]  * np.sqrt(8)*y*(3*x**2 - y**2)
        Z10 =  Z[10] * np.sqrt(8)*x*(x**2 - 3*y**2)
        Z11 =  Z[11] * np.sqrt(5)*(6*r2**2 - 6*r2 + 1)
        Z12 =  Z[12] * np.sqrt(10)*(x**2 - y**2)*(4*r2 - 3)
        Z13 =  Z[13] * 2*np.sqrt(10)*x*y*(4*r2 - 3)
        Z14 =  Z[14] * np.sqrt(10)*(r2**2 - 8*x**2*y**2)
        Z15 =  Z[15] * 4*np.sqrt(10)*x*y*(x**2 - y**2)

        Z_comb = Z1 + Z2  +  Z3 +  Z4 +  Z5 +  Z6 +  Z7 +  Z8 + \
                 Z9 + Z10 + Z11 + Z12 + Z13 + Z14 + Z15
    
        return Z_comb
    
    ###############
    def cmask(self,array,pad=0):
        """
        Create a circular mask as the aperture.
        Extended over the whole X, Y grid.
        Namely, the range of each axis defines the diameter.
    
        Inputs:
        - array: numpy 2d array. 
          The input distribution of amplitude, phase, etc.
          
        Options:
        - pad: float
          The value to be padded outside the aperture.
          Defaults to 0
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