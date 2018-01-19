import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import scipy.stats as spt
import scipy.integrate as spi
import scipy.special as sps
import copy as cp
import astropy.units as u
import astropy.constants as c

from fake_spectra import spectra as sa
from fake_spectra import griddedspectra as gs
from fake_spectra import randspectra as rs

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
    '''samples_histogram = np.histogram2d(x, y, bins=[n_bins_x, n_bins_y])[0]
    f_summed = np.histogram2d(x, y, bins=[n_bins_x, n_bins_y], weights=f)[0] #Needs units of f
    return f_summed / samples_histogram'''
    return spt.binned_statistic_2d(x,y,f,statistic='mean',bins=[n_bins_x,n_bins_y])[0]

def standard_error(array_1D):
    return np.std(array_1D, ddof=1) / mh.sqrt(array_1D.size)

def bin_f_x_y_histogram_standard_error(x, y, f, n_bins_x, n_bins_y):
    return spt.binned_statistic_2d(x,y,f,statistic=standard_error,bins=[n_bins_x,n_bins_y])[0]

def bin_f_x_y_histogram_count(x,y,f,n_bins_x,n_bins_y):
    return spt.binned_statistic_2d(x,y,f,statistic='count',bins=[n_bins_x,n_bins_y])[0]

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

#http://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
def voigt(x, sigma, gamma, x0):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component std dev sigma.

    """

    return np.real(sps.wofz((x.value - x0.value + 1j*gamma.value)/sigma.value/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)

def voigt_amplified(x, sigma, gamma, amp, x0):
    return (amp * voigt(x, sigma, gamma, x0)) / voigt(x0, sigma, gamma, x0)

def full_voigt_optical_depth(velocity_samples, column_density, central_velocity, central_wavelength = 1215.67 * u.Angstrom): #lambda_0 for Ly-a forest
    gas_temperature = 1.e+4 * u.K
    transition_wavelength = 1215.67 * u.Angstrom #Ly-a
    ion_mass = c.m_p #Ionised hydrogen - Ly-a
    oscillator_strength = 0.4164 #Ly-a
    damping_constant = 6.265 * (10 ** 8) * u.Hz

    #wavelength_samples = central_wavelength * (1 + ((velocity_samples - central_velocity) / c.c))
    wavelength_samples = central_wavelength * np.exp((velocity_samples - central_velocity) / c.c)

    del_lambda_D = transition_wavelength * mh.sqrt(2. * c.k_B * gas_temperature / ion_mass / (c.c ** 2))
    x = (wavelength_samples - central_wavelength) / del_lambda_D
    y = (transition_wavelength ** 2) * damping_constant / 4. / mh.pi / c.c / del_lambda_D
    z = x + (y * 1.j)
    u_x_y = np.real(sps.wofz(z.value))
    atomic_absorption_coeff_numer = mh.sqrt(mh.pi) * (c.e.gauss ** 2) * oscillator_strength * (transition_wavelength ** 2) * u_x_y
    atomic_absorption_coeff_denom = c.m_e * (c.c ** 2) * del_lambda_D #* (1. * u.F / u.m) #c.eps0
    atomic_absorption_coeff = atomic_absorption_coeff_numer / atomic_absorption_coeff_denom
    optical_depth = (column_density * atomic_absorption_coeff).to(u.dimensionless_unscaled)

    return optical_depth, del_lambda_D, z, wavelength_samples

def voigt_power_spectrum(spectrum_length, velocity_bin_width, mean_flux, column_density=None, sigma=None, gamma=None, amp=None):
    n_velocity_samples = spectrum_length / velocity_bin_width
    velocity_samples = np.arange(-1 * n_velocity_samples / 2, n_velocity_samples / 2 + 1) * velocity_bin_width
    if column_density is not None:
        optical_depth, del_lambda_D, z, wavelength_samples = full_voigt_optical_depth(velocity_samples, column_density, 0. * u.km / u.s)
    else:
        optical_depth = voigt_amplified(velocity_samples, sigma, gamma, amp, 0. * u.km / u.s)
    flux = np.exp(-1. * optical_depth.value)
    delta_flux = flux / mean_flux - 1.
    delta_flux_FT = np.fft.rfft(delta_flux) / delta_flux.shape[0]
    k_samples = np.fft.rfftfreq(delta_flux.shape[0], d = velocity_bin_width) * 2. * mh.pi
    return (np.real(delta_flux_FT)**2 + np.imag(delta_flux_FT)**2) * spectrum_length, k_samples, velocity_samples, optical_depth, del_lambda_D, z, wavelength_samples, delta_flux_FT, delta_flux

def _set_real_values_in_hermitian_box(box,x,y,z):
    box[0, 0, 0] = np.real(box[0, 0, 0]) * mh.sqrt(2.)  # Force mean mode to be real - PROBS NEED TO * SQRT(2)
    if x % 2 == 0:
        box[int(x / 2), :, :] = np.real(box[int(x / 2), :, :]) * mh.sqrt(2.)  # Force Nyquist frequencies to be real
    if y % 2 == 0:
        box[:, int(y / 2), :] = np.real(box[:, int(y / 2), :]) * mh.sqrt(2.)
    if z % 2 == 0:
        box[:, :, int(z / 2)] = np.real(box[:, :, int(z / 2)]) * mh.sqrt(2.)
    return box

def make_box_hermitian(box):
    if is_astropy_quantity(box):
        box = box.value
    x,y,z = box.shape
    box = _set_real_values_in_hermitian_box(box,x,y,z) #SLOW if dimensionless quantity
    for i in range(int(x//2 + 1)): #Only need to loop over half the samples
        for j in range(y):
            for k in range(z):
                box[i,j,k] = np.conj(box[-i,-j,-k])
    return box

def gen_log_space(limit, n): #Courtesy of http://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.round(result).astype('int') - 1 #np.array(map(lambda x: round(x)-1, result)) #, dtype=np.uint64)