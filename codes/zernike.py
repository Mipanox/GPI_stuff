"""
Generator for Zernike polynomials (first 35)
Can either call from polar or Cartesian coordinates.

The normalization choice (see below in `ZernikePolar`) actually
corresponds to the equivalence between rms levels and the coefficients
(see the notebook: 
( https://nbviewer.jupyter.org/github/Mipanox/GPI_stuff/blob/master/notebooks/zerniketest2.ipynb)

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
                      Z11=0,Z12=0,Z13=0,Z14=0,Z15=0,
                      Z16=0,Z17=0,Z18=0,Z19=0,Z20=0,
                      Z21=0,Z22=0,Z23=0,Z24=0,Z25=0,
                      Z26=0,Z27=0,Z28=0,Z29=0,Z30=0, 
                      Z31=0,Z32=0,Z33=0,Z34=0,Z35=0, Npix=1024,
                 coeff=None):
        ## coefficients
        if coeff is not None:
            if len(coeff) != 35:
                raise IndexError('Length of coefficient list should be 35')
            self.coeff = coeff
        else:
            self.coeff = [Z1,   Z2,  Z3,  Z4,  Z5,  Z6,  Z7, 
                          Z8,   Z9, Z10, Z11, Z12, Z13, Z14, 
                          Z15, Z16, Z17, Z18, Z19, Z20, Z21,
                          Z22, Z23, Z24, Z25, Z26, Z27, Z28,
                          Z29, Z30, Z31, Z32, Z33, Z34, Z35]
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
        Z16 =  Z[16] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.cos(u)
        Z17 =  Z[17] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.sin(u)
        Z18 =  Z[18] * np.sqrt(12)*(5*r**2-4)*r**3*np.cos(3*u)
        Z19 =  Z[19] * np.sqrt(12)*(5*r**2-4)*r**3*np.sin(3*u)
        Z20 =  Z[20] * np.sqrt(12)*r**5*np.cos(5*u)
        Z21 =  Z[21] * np.sqrt(12)*r**5*np.sin(5*u)
        Z22 =  Z[22] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1)
        Z23 =  Z[23] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.sin(2*u)
        Z24 =  Z[24] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.cos(2*u)
        Z25 =  Z[25] * np.sqrt(14)*(6*r**2-5)*r**4*np.sin(4*u)
        Z26 =  Z[26] * np.sqrt(14)*(6*r**2-5)*r**4*np.cos(4*u)
        Z27 =  Z[27] * np.sqrt(14)*r**6*np.sin(6*u)
        Z28 =  Z[28] * np.sqrt(14)*r**6*np.cos(6*u)
        Z29 =  Z[29] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.sin(u)
        Z30 =  Z[30] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.cos(u)
        Z31 =  Z[31] * 4*(21*r**4-30*r**2+10)*r**3*np.sin(3*u)
        Z32 =  Z[32] * 4*(21*r**4-30*r**2+10)*r**3*np.cos(3*u)
        Z33 =  Z[33] * 4*(7*r**2-6)*r**5*np.sin(5*u)
        Z34 =  Z[34] * 4*(7*r**2-6)*r**5*np.cos(5*u)
        Z35 =  Z[35] * 4*r**7*np.sin(7*u)

        Z_comb = Z1 + Z2  +  Z3 +  Z4 +  Z5 +  Z6 +  Z7 +  Z8 + \
                 Z9 + Z10 + Z11 + Z12 + Z13 + Z14 + Z15 + Z16 + \
                Z17 + Z18 + Z19 + Z20 + Z21 + Z22 + Z23 + Z24 + \
                Z25 + Z26 + Z27 + Z28 + Z29 + Z30 + Z31 + Z32 + \
                Z33 + Z34 + Z35
    
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
        Z16 =  Z[16] * np.sqrt(12)*x*(10*r2**2-12*r2+3)
        Z17 =  Z[17] * np.sqrt(12)*y*(10*r2**2-12*r2+3)
        Z18 =  Z[18] * np.sqrt(12)*x*(x**2-3*y**2)*(5*r2-4)
        Z19 =  Z[19] * np.sqrt(12)*y*(3*x**2-y**2)*(5*r2-4)
        Z20 =  Z[20] * np.sqrt(12)*x*(16*x**4-20*x**2*r2+5*r2**2)
        Z21 =  Z[21] * np.sqrt(12)*y*(16*y**4-20*y**2*r2+5*r2**2)
        Z22 =  Z[22] * np.sqrt(7)*(20*r2**3-30*r2**2+12*r2-1)
        Z23 =  Z[23] * 2*np.sqrt(14)*x*y*(15*r2**2-20*r2+6)
        Z24 =  Z[24] * np.sqrt(14)*(x**2-y**2)*(15*r2**2-20*r2+6)
        Z25 =  Z[25] * 4*np.sqrt(14)*x*y*(x**2-y**2)*(6*r2-5)
        Z26 =  Z[26] * np.sqrt(14)*(8*x**4-8*x**2*r2+r2**2)*(6*r2-5)
        Z27 =  Z[27] * np.sqrt(14)*x*y*(32*x**4-32*x**2*r2+6*r2**2)
        Z28 =  Z[28] * np.sqrt(14)*(32*x**6-48*x**4*r2+18*x**2*r2**2-r2**3)
        Z29 =  Z[29] * 4*y*(35*r2**3-60*r2**2+30*r2-4)
        Z30 =  Z[30] * 4*x*(35*r2**3-60*r2**2+30*r2-4)
        Z31 =  Z[31] * 4*y*(3*x**2-y**2)*(21*r2**2-30*r2+10)
        Z32 =  Z[32] * 4*x*(x**2-3*y**2)*(21*r2**2-30*r2+10)
        Z33 =  Z[33] * 4*(7*r2-6)*(4*x**2*y*(x**2-y**2)+y*(r2**2-8*x**2*y**2))
        Z34 =  Z[34] * (4*(7*r2-6)*(x*(r2**2-8*x**2*y**2)-4*x*y**2*(x**2-y**2)))
        Z35 =  Z[35] * (8*x**2*y*(3*r2**2-16*x**2*y**2)+4*y*(x**2-y**2)*(r2**2-16*x**2*y**2))

        Z_comb = Z1 + Z2  +  Z3 +  Z4 +  Z5 +  Z6 +  Z7 +  Z8 + \
                 Z9 + Z10 + Z11 + Z12 + Z13 + Z14 + Z15 + Z16 + \
                Z17 + Z18 + Z19 + Z20 + Z21 + Z22 + Z23 + Z24 + \
                Z25 + Z26 + Z27 + Z28 + Z29 + Z30 + Z31 + Z32 + \
                Z33 + Z34 + Z35
    
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