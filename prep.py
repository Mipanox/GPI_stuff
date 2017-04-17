from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits
from scipy.ndimage.interpolation import shift

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
#plt.rcParams['legend.fontsize'] = 14
#plt.rc('text', usetex=True)