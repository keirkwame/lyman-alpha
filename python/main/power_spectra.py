import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import scipy.integrate as spi
import scipy.interpolate as spp
import copy as cp
import astropy.units as u
import spectra as sa
import griddedspectra as gs
import randspectra as rs
import sys

from utils import *

class PowerSpectrum():
    """Class to evaluate 1D and 3D versions of a power spectrum"""
    def _integrand(self,k_perp,k_z):
        k3D = np.sqrt(k_perp ** 2 + k_z ** 2)
        Pk3D = self.evaluate3d_isotropic(k3D)
        return (Pk3D * k_perp) / (2. * mh.pi)

    def evaluate1d(self, k_x_vec, k_y_vec, k_z_vec):
        k_perp_min = np.sqrt(k_x_vec[0] ** 2 + k_y_vec[0] ** 2)
        k_perp_max = np.sqrt(np.max(k_x_vec) ** 2 + np.max(k_y_vec) ** 2)
        pow_kz = np.zeros(k_z_vec.shape[0])
        for i in range(k_z_vec.shape[0]):
            pow_kz[i] = spi.quad(self._integrand, k_perp_min.value, k_perp_max.value, (k_z_vec[i].value,))[0]
        return pow_kz

    def set_anisotropic_functional_form(self,mu_coefficients):
        self._mu_coefficients = mu_coefficients

    def evaluate3d_anisotropic(self,k,mu):
        k_para, k_perp = spherical_to_cylindrical_coordinates(k, mu)
        return self.evaluate3d_isotropic(k) * np.polyval(self._mu_coefficients(k_para,k_perp),mu)

    def evaluate_multipole(self,multipole,k):
        mu_samples = np.linspace(-1.,1.,2000) * u.dimensionless_unscaled
        anisotropic_power = self.evaluate3d_anisotropic(k[:, np.newaxis], mu_samples[np.newaxis, :])
        anisotropic_integrand=anisotropic_power * evaluate_legendre_polynomial(mu_samples[np.newaxis,:],multipole)
        return np.trapz(anisotropic_integrand,mu_samples[np.newaxis,:]) * ((2.*multipole + 1.) / 2.)


class PowerLawPowerSpectrum(PowerSpectrum):
    """Sub-class to evaluate a power law power spectrum"""
    def __init__(self,pow_index,pow_pivot,pow_amp):
        assert is_astropy_quantity(pow_pivot)
        self._pow_index = pow_index
        self._pow_pivot = pow_pivot
        self._pow_amp = pow_amp

    def evaluate3d_isotropic(self, k):
        Pk = self._pow_amp * ((k / self._pow_pivot) ** self._pow_index)
        if is_astropy_quantity(k):
            return Pk
        else:
            return Pk.value


class PreComputedPowerSpectrum(PowerSpectrum):
    """Sub-class to interpolate as necessary a pre-computed (isotropic) power spectrum"""
    def __init__(self,fname,n_interpolation_samples='default'):
        self._fname = fname
        if n_interpolation_samples == 'default':
            self._n_interpolation_samples = 2500
        else:
            self._n_interpolation_samples = n_interpolation_samples

        self.k_raw, self.power_raw = np.loadtxt(self._fname,unpack=True)
        self.k_raw = self.k_raw / u.Mpc #Convert to Astropy quantity

    def _form_reduced_arrays(self, k_raw_reduced, power_raw_reduced):
        slice_array = gen_log_space(k_raw_reduced.size,self._n_interpolation_samples)
        k_raw_reduced = k_raw_reduced[slice_array]
        power_raw_reduced = power_raw_reduced[slice_array]
        return k_raw_reduced, power_raw_reduced

    def _correction_for_interpolation(self,k_array):
        assert is_astropy_quantity(k_array)
        return np.log10(k_array.value) + 100.

    def _set_interpolating_function(self,k_min,k_max):
        k_raw_reduced_bool_arr = (self.k_raw >= (k_min-(1.e-3/u.Mpc))) * (self.k_raw <= (k_max+(1.e-3/u.Mpc)))
        k_reduced,power_reduced = self._form_reduced_arrays(self.k_raw[k_raw_reduced_bool_arr],self.power_raw[k_raw_reduced_bool_arr])
        self._interpolating_func=spp.interp1d(self._correction_for_interpolation(k_reduced),power_reduced,kind='cubic')

    def evaluate3d_isotropic(self, k):
        k_modified = k.to(1. / u.Mpc)
        k_modified[k == 0.] = np.min(k_modified[k_modified > 0. / u.Mpc])
        self._set_interpolating_function(np.min(k_modified),np.max(k_modified))
        power_interpolated = self._interpolating_func(self._correction_for_interpolation(k_modified))
        power_interpolated[k == 0.] = 0.
        return power_interpolated


class CAMBPowerSpectrum(PowerSpectrum):
    """Sub-class to evaluate a power spectrum using CAMB"""
    def __init__(self):
        pass

    def evaluate3d_isotropic(self, k):
        return 0.