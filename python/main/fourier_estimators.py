import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import scipy.integrate as spi
import scipy.special as sps
import copy as cp
import astropy.units as u

from fake_spectra import spectra as sa
from fake_spectra import griddedspectra as gs
from fake_spectra import randspectra as rs

import sys

from utils import *

def get_matter_power_spectrum_two_coords_binned(redshift, k_box, coord_box1, coord_box2, n_bins1, n_bins2,
                                                hubble_constant, cosmology_name='base_plikHM_TTTEEE_lowTEB_2015'):
    from CAMB_bDM_tests import main as camb_wrap

    k_h = k_box.flatten()[1:] / hubble_constant
    camb_cosmology_instance = camb_wrap.CAMB_bDM_cosmology(cosmology_name=cosmology_name, k_h_range=[np.min(k_h), np.max(k_h)], z=np.linspace(start=redshift, stop=0., num=4))
    matter_power_spectrum_unbinned = camb_cosmology_instance.get_P_k_z(k_h=k_h)[0][-1]

    x = coord_box1.flatten()[1:]
    y = coord_box2.flatten()[1:]

    return bin_f_x_y_histogram(x, y, matter_power_spectrum_unbinned, n_bins1, n_bins2)


class FourierEstimator(object):
    """Class to estimate power spectra from a box of fluctuations"""
    def __init__(self, first_box, second_box):
        self._first_box = first_box
        self._second_box = second_box


class FourierEstimator1D(FourierEstimator):
    """Sub-class to calculate 1D power spectra"""
    def __init__(self, first_box, second_box = None, n_skewers = None):
        super(FourierEstimator1D, self).__init__(first_box, second_box)
        if n_skewers == None:
            self._n_skewers = self._first_box.shape[0] * self._first_box.shape[1]
        else:
            self._n_skewers = n_skewers

    def samples_1D(self):
        return rd.sample(np.arange(self._first_box.shape[0] * self._first_box.shape[1]), self._n_skewers)

    def skewers_1D(self):
        if self._first_box.ndim == 3:
            return self._first_box.reshape((self._first_box.shape[0] * self._first_box.shape[1], -1))
        elif self._first_box.ndim == 2:
            return self._first_box

    def get_power_1D(self, norm = True):
        real_space_modes = self.skewers_1D()
        if norm == False:
            norm_fac = 1.
        elif norm == True:
            norm_fac = 1. / real_space_modes.shape[-1]
        fourier_modes = np.fft.rfft(real_space_modes, axis = 1) * norm_fac
        power = np.real(fourier_modes) ** 2 + np.imag(fourier_modes) ** 2
        average_power = np.mean(power, axis=0)
        return average_power


class FourierEstimator3D(FourierEstimator):
    """Sub-class to calculate 3D power spectra"""
    def __init__(self, first_box, second_box = None, grid = True, x_step = 1, y_step = 1, n_skewers = 0):
        super(FourierEstimator3D, self).__init__(first_box, second_box)
        self._grid = grid
        self._x_step = x_step
        self._y_step = y_step
        self._n_skewers = n_skewers

    def samples_3D(self):
        if self._grid == True:
            return np.arange(0, self._first_box.shape[0], self._x_step), np.arange(0, self._first_box.shape[1], self._y_step)
        elif self._grid == False:
            n_zeros = (self._first_box.shape[0] * self._first_box.shape[1]) - self._n_skewers
            return rd.sample(np.arange(self._first_box.shape[0] * self._first_box.shape[1]), n_zeros)

    def skewers_3D(self):
        if self._grid == True:
            xy_samps = self.samples_3D()
            return self._first_box[xy_samps[0], :, :][:, xy_samps[1], :]
        elif self._grid == False:
            skewers = cp.deepcopy(self._first_box)
            skewers = skewers.reshape((self._first_box.shape[0] * self._first_box.shape[1], -1))
            skewers[self.samples_3D(), :] = 0. + 0.j
            skewers = skewers.reshape(self._first_box.shape[0], self._first_box.shape[1], -1)
            return skewers

    def get_power_3D(self, norm = True):
        real_space_modes = self.skewers_3D()
        if norm == False:
            norm_fac = 1.
        elif norm == True:
            norm_fac = 1. / real_space_modes.size
        fourier_modes = np.fft.fftn(real_space_modes) * norm_fac
        if self._second_box is None:
            power = np.real(fourier_modes) ** 2 + np.imag(fourier_modes) ** 2
        else:
            fourier_modes_2 = np.fft.fftn(self._second_box) * norm_fac
            power = (fourier_modes.real * fourier_modes_2.real) + (fourier_modes.imag * fourier_modes_2.imag)
        return power, fourier_modes

    def get_power_3D_cylindrical_coords(self, k_z_mod_box, k_perp_box, n_bins_z, n_bins_perp, norm = True):
        power_sorted,k_z_sorted,k_perp_sorted = self.get_power_legendre_integrand(k_z_mod_box, k_perp_box, n_bins_z, norm)
        return bin_2D_data(power_sorted,n_bins_perp),bin_2D_data(k_z_sorted,n_bins_perp),bin_2D_data(k_perp_sorted,n_bins_perp)

    def get_power_3D_unique(self, k_box, norm = True):
        power = self.get_power_3D(norm)[0]
        k_unique = np.unique(k_box)
        power_unique = np.zeros_like(k_unique.value)
        for i in range(k_unique.shape[0]):
            print("Binning 3D power according to unique value of |k| #%i/%i" %(i+1, k_unique.shape[0]))
            power_unique[i] = np.mean(power[k_box == k_unique[i]])
        return power_unique, k_unique

    def get_flux_power_3D_sorted(self, k_box, norm = True, mu_box = None):
        k_argsort = np.argsort(k_box, axis = None)
        power = self.get_power_3D(norm)[0]
        if mu_box == None:
            return sort_3D_to_1D(power, k_argsort)[1:], sort_3D_to_1D(k_box, k_argsort)[1:]
        else:
            return sort_3D_to_1D(power, k_argsort)[1:],sort_3D_to_1D(k_box, k_argsort)[1:],sort_3D_to_1D(mu_box, k_argsort)[1:]

    def get_flux_power_3D_binned(self, k_box, n_bins, norm = True):
        power_sorted, k_sorted = self.get_flux_power_3D_sorted(k_box, norm)
        return bin_1D_data(power_sorted, n_bins), bin_1D_data(k_sorted, n_bins)

    def _form_return_list(self, x, y, n_bins_x, n_bins_y, norm, bin_coord_x, bin_coord_y, count, std_err):
        ensemble_statistic = self.get_power_3D(norm)[0].flatten()[1:]

        return_list = [None] * (1 + bin_coord_x + bin_coord_y + count + std_err) #Number of calculations
        return_list[0] = bin_f_x_y_histogram(x, y, ensemble_statistic, n_bins_x, n_bins_y) #Always bin power
        i = 1
        if bin_coord_x == True:
            return_list[i] = bin_f_x_y_histogram(x, y, x, n_bins_x, n_bins_y)
            i+=1
        if bin_coord_y == True:
            return_list[i] = bin_f_x_y_histogram(x, y, y, n_bins_x, n_bins_y)
            i+=1
        if count == True:
            return_list[i] = bin_f_x_y_histogram_count(x, y, ensemble_statistic, n_bins_x, n_bins_y)
            i+=1
        if std_err == True:
            return_list[i] = bin_f_x_y_histogram_standard_error(x, y, ensemble_statistic, n_bins_x, n_bins_y)
        return return_list

    def get_power_3D_two_coords_binned(self, coord_box1, coord_box2, n_bins1, n_bins2, norm=True, bin_coord1=True, bin_coord2=True, count=False, std_err=False):
        x = coord_box1.flatten()[1:]
        y = coord_box2.flatten()[1:]
        return self._form_return_list(x, y, n_bins1, n_bins2, norm, bin_coord1, bin_coord2, count, std_err)

    def get_power_legendre_integrand(self, k_box, mu_box, n_bins, norm = True): #NEEDS TIDYING-UP!!!
        power_sorted, k_sorted, mu_sorted = self.get_flux_power_3D_sorted(k_box, norm, mu_box)
        mu_2D_k_sorted = arrange_data_in_2D(mu_sorted, n_bins)
        k_2D_k_sorted = arrange_data_in_2D(k_sorted, n_bins)
        power_2D_k_sorted = arrange_data_in_2D(power_sorted, n_bins)
        mu_2D_mu_argsort = np.argsort(mu_2D_k_sorted)
        mu_2D_mu_sorted = np.zeros_like(mu_2D_k_sorted)
        k_2D_mu_sorted = np.zeros_like(k_2D_k_sorted)
        power_2D_mu_sorted = np.zeros_like(power_2D_k_sorted)
        for i in range(mu_2D_mu_sorted.shape[0]):
            mu_2D_mu_sorted[i,:] = mu_2D_k_sorted[i,mu_2D_mu_argsort[i]]
            k_2D_mu_sorted[i,:] = k_2D_k_sorted[i,mu_2D_mu_argsort[i]]
            power_2D_mu_sorted[i,:] = power_2D_k_sorted[i,mu_2D_mu_argsort[i]]
        return power_2D_mu_sorted, k_2D_mu_sorted, mu_2D_mu_sorted

    def get_flux_power_3D_multipole(self, multipole, k_box, mu_box, n_bins, norm = True):
        power_mu_sorted, k_mu_sorted, mu_mu_sorted = self.get_power_legendre_integrand(k_box, mu_box, n_bins, norm)
        total_integrand = power_mu_sorted * evaluate_legendre_polynomial(mu_mu_sorted, multipole)
        power_integrated = np.trapz(total_integrand, x = mu_mu_sorted) * ((2. * multipole + 1.) / 2.)
        return power_integrated, np.mean(k_mu_sorted, axis = -1), power_mu_sorted