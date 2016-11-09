import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import scipy.integrate as spi
import scipy.special as sps
import copy as cp
import astropy.units as u
import spectra as sa
import griddedspectra as gs
import randspectra as rs
import sys

def sort_3D_to_1D(array_3D, args_1D):
    return array_3D.flatten()[args_1D]

def arrange_data_in_2D(array_1D, n_bins):
    shortened_length = int(mh.floor(array_1D.size / n_bins) * n_bins)
    return array_1D[:shortened_length].reshape((n_bins, -1))

def bin_data(array_1D, n_bins):
    return np.mean(arrange_data_in_2D(array_1D,n_bins), axis=-1) #, [np.mean(array_1D[shortened_length:])]))

def is_astropy_quantity(var):
    return hasattr(var,'value')

def evaluate_legendre_polynomial(array,multipole):
    legendre_polynomial = sps.legendre(multipole)  # l-th order Legendre polynomial
    return legendre_polynomial(array.value) * u.dimensionless_unscaled