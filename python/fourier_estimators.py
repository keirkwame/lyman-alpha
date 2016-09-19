import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import scipy.integrate as spi
import copy as cp
import astropy.units as u
import spectra as sa
import griddedspectra as gs
import randspectra as rs
import sys

from utils import *

class FourierEstimator(object): #Need object dependence so sub-classes can inherit __init__
    """Class to estimate power spectra from a box of fluctuations"""
    def __init__(self,gauss_box):
        self._gauss_box = gauss_box


class FourierEstimator1D(FourierEstimator):
    """Sub-class to calculate 1D power spectra"""
    def __init__(self,gauss_box,n_skewers):
        super(FourierEstimator1D, self).__init__(gauss_box)
        self._nskewers = n_skewers

    def samples_1D(self):
        return rd.sample(np.arange(self._gauss_box.shape[0] * self._gauss_box.shape[1]), self._nskewers)

    def skewers_1D(self):
        return self._gauss_box.reshape((self._gauss_box.shape[0] * self._gauss_box.shape[1], -1))[self.samples_1D(), :]

    #COURTESY OF SIMEON BIRD
    def get_flux_power_1D(self):
        delta_flux = self.skewers_1D()

        df_hat = np.fft.fft(delta_flux, axis=1)
        flux_power = np.real(df_hat) ** 2 + np.imag(df_hat) ** 2
        #Average over all sightlines
        avg_flux_power = np.mean(flux_power, axis=0)

        return avg_flux_power


class FourierEstimator3D(FourierEstimator):
    """Sub-class to calculate 3D power spectra"""
    def __init__(self,gauss_box,grid=True,x_step=1,y_step=1,n_skewers=0):
        super(FourierEstimator3D, self).__init__(gauss_box)
        self._grid = grid
        self._x_step = x_step
        self._y_step = y_step
        self._n_skewers = n_skewers

    def samples_3D(self):
        if self._grid == True:
            return np.arange(0,self._gauss_box.shape[0],self._x_step),np.arange(0,self._gauss_box.shape[1],self._y_step)
        elif self._grid == False:
            n_zeros = (self._gauss_box.shape[0] * self._gauss_box.shape[1]) - self._n_skewers
            return rd.sample(np.arange(self._gauss_box.shape[0] * self._gauss_box.shape[1]), n_zeros) #Sampling zeros

    def skewers_3D(self):
        if self._grid == True:
            xy_samps = self.samples_3D()
            return self._gauss_box[xy_samps[0],:,:][:,xy_samps[1],:]
        elif self._grid == False:
            skewers = cp.deepcopy(self._gauss_box)
            skewers = skewers.reshape((self._gauss_box.shape[0] * self._gauss_box.shape[1], -1))
            skewers[self.samples_3D(), :] = 0. + 0.j
            skewers = skewers.reshape(self._gauss_box.shape[0], self._gauss_box.shape[1], -1)
            return skewers

    def get_flux_power_3D(self,norm=True):
        flux_real = self.skewers_3D()
        if norm == False:
            norm_fac = 1.
        elif norm == True:
            norm_fac = flux_real.size
        df_hat = np.fft.fftn(flux_real) / norm_fac
        flux_power = np.real(df_hat) ** 2 + np.imag(df_hat) ** 2
        return flux_power, df_hat

    def get_flux_power_3D_mod_k(self,k_box,norm=True):
        flux_power = self.get_flux_power_3D(norm)[0]
        k_unique = np.unique(k_box)
        power_unique = np.zeros_like(k_unique.value)
        for i in range(k_unique.shape[0]):
            print("Binning 3D power according to unique value of |k| #%i/%i" %(i+1,k_unique.shape[0]))
            power_unique[i] = np.mean(flux_power[k_box == k_unique[i]])
        return power_unique, k_unique

    def get_flux_power_3D_binned(self,k_box,n_bins,norm=True):
        k_argsort = np.argsort(k_box,axis=None)
        k_sorted = sort_3D_to_1D(k_box, k_argsort)
        flux_power = self.get_flux_power_3D(norm)[0]
        power_sorted = sort_3D_to_1D(flux_power, k_argsort)
        return bin_data(power_sorted,n_bins), bin_data(k_sorted,n_bins)