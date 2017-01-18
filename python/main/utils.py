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

def arrange_data_in_3D(array_2D, n_bins):
    shortened_length = int(mh.floor(array_2D.shape[-1] / n_bins) * n_bins)
    return array_2D[:,:shortened_length].reshape((array_2D.shape[0],n_bins,-1))

def bin_1D_data(array_1D, n_bins):
    return np.mean(arrange_data_in_2D(array_1D,n_bins), axis=-1)

def bin_2D_data(array_2D, n_bins):
    return np.mean(arrange_data_in_3D(array_2D,n_bins), axis=-1)

def bin_f_x_y_histogram(x,y,f,n_bins_x,n_bins_y):
    samples_histogram = np.histogram2d(x, y, bins=[n_bins_x, n_bins_y])[0]
    f_summed = np.histogram2d(x, y, bins=[n_bins_x, n_bins_y], weights=f)[0] #Needs units of f
    return f_summed / samples_histogram

def get_end_index(bin_size):
    if bin_size == 1:
        return None
    else:
        return -1 * (bin_size - 1)

def calculate_local_average_of_array(array_nD, bin_size): #MAKE WORK WITH UNITS!!!
    array_nD_local_average = np.zeros(array_nD.shape + (bin_size,))
    for i in range(bin_size):
        array_nD_local_average[..., i] = np.roll(array_nD, -1 * i, axis=-1)
    return np.mean(array_nD_local_average, axis=-1)[..., :get_end_index(bin_size)]

def is_astropy_quantity(var):
    return hasattr(var,'value')

def evaluate_legendre_polynomial(array,multipole):
    legendre_polynomial = sps.legendre(multipole)  #l-th order Legendre polynomial
    return legendre_polynomial(array.value) * u.dimensionless_unscaled

def spherical_to_cylindrical_coordinates(k,mu):
    k_para = k * mu
    k_perp = k * np.sqrt(1. - mu**2)
    return k_para, k_perp