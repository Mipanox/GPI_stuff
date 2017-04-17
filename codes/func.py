"""
Functions dealing with Fourier optics
"""

def cmask(array,radius,fill=1):
    """
    Create a circular mask as the aperture.
    
    Inputs:
    - array: numpy 2d array. 
      The input distribution of amplitude, phase, etc.
      
    Parameters:
    - radius: float
      Radius of the circular mask in pixels.
    
    Options:
    - fill: integer. 1 or 0
      Fill in ones or zeros
    """
    nx,ny = array.shape
    if nx%2 or ny%2:
        raise ValueError('Array dimensions have to be odd')
    
    a , b = (nx-1)/2, (ny-1)/2 ## centroid
    y , x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius**2
    
    arr = np.copy(array)
    arr[mask] = fill
    return arr

def Ef_after(Amp,Pha):
    """ 
    Complex Fourier Transformed wave (phasor).
    
    Inputs:
    - Amp, Pha: numpy 2d arrays.
      The amplitude and phase distributions
    """
    try:
        if Pha.unit != 'rad':
            Pha = Pha.to(u.rad)
    except:
        raise TypeError("Please specify unit")
        
    E_ori = Amp * np.exp(1j*Pha.value)
    
    E_aft = np.fft.fft2(E_ori)
    E_fin = np.fft.fftshift(E_aft)
    
    return E_fin
    
def Int_after(Amp,Pha):
    """ 
    Intensity after FT given amplitude and phase.
    Namely, the absolute square of `Ef_after`.
    """
    E_after = Ef_after(Amp,Pha)
    Int_fi_ = abs(E_after)**2
    
    Int_fin = Int_fi_/np.max(Int_fi_)
    return Int_fin

def Ef_to_AP(Ef):
    """ 
    Convert a complex phasor to amplitude and phase 
    """
    return abs(Ef), np.arctan2(Ef.imag,Ef.real) * u.rad