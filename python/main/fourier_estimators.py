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

from utils import *

class FourierEstimator(object): #Need object dependence so sub-classes can inherit __init__
    """Class to estimate power spectra from a box of fluctuations"""
    def __init__(self,gauss_box,second_box):
        self._gauss_box = gauss_box
        self._second_box = second_box


class FourierEstimator1D(FourierEstimator):
    """Sub-class to calculate 1D power spectra"""
    def __init__(self,gauss_box,second_box=None,n_skewers=None):
        super(FourierEstimator1D, self).__init__(gauss_box,second_box)
        if n_skewers == None:
            self._n_skewers = self._gauss_box.shape[0] * self._gauss_box.shape[1]
        else:
            self._n_skewers = n_skewers

    def samples_1D(self): #NOT USED AT THE MOMENT!!!
        return rd.sample(np.arange(self._gauss_box.shape[0] * self._gauss_box.shape[1]), self._nskewers)

    def skewers_1D(self):
        if self._gauss_box.ndim == 3:
            return self._gauss_box.reshape((self._gauss_box.shape[0] * self._gauss_box.shape[1], -1)) #[self.samples_1D(), :]
        elif self._gauss_box.ndim == 2:
            return self._gauss_box

    #COURTESY OF SIMEON BIRD
    def get_flux_power_1D(self,norm=True): #MODIFY FOR SECOND BOX
        delta_flux = self.skewers_1D()
        if norm == False:
            norm_fac = 1.
        elif norm == True:
            norm_fac = 1. / delta_flux.shape[-1]
        df_hat = np.fft.rfft(delta_flux, axis=1) * norm_fac
        flux_power = np.real(df_hat) ** 2 + np.imag(df_hat) ** 2
        #Average over all sightlines
        avg_flux_power = np.mean(flux_power, axis=0)
        return avg_flux_power


class FourierEstimator3D(FourierEstimator):
    """Sub-class to calculate 3D power spectra"""
    def __init__(self,gauss_box,second_box=None,grid=True,x_step=1,y_step=1,n_skewers=0): #Probably want to init with k_box, etc.
        super(FourierEstimator3D, self).__init__(gauss_box,second_box)
        self._grid = grid
        self._x_step = x_step
        self._y_step = y_step
        self._n_skewers = n_skewers

    def samples_3D(self): #Needs test module!
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
            norm_fac = 1. / flux_real.size
        df_hat = np.fft.fftn(flux_real) * norm_fac
        if self._second_box is None:
            flux_power = np.real(df_hat) ** 2 + np.imag(df_hat) ** 2
        else:
            df_hat_2 = np.fft.fftn(self._second_box) * norm_fac
            flux_power = (df_hat.real * df_hat_2.real) + (df_hat.imag * df_hat_2.imag)
        return flux_power, df_hat

    def get_flux_power_3D_cylindrical_coords(self,k_z_mod_box,k_perp_box,n_bins_z,n_bins_perp,norm=True):
        #Need to generalise sorting of power by two coordinates!!!
        power_sorted,k_z_sorted,k_perp_sorted = self.get_flux_power_legendre_integrand(k_z_mod_box,k_perp_box,n_bins_z,norm)
        return bin_2D_data(power_sorted,n_bins_perp),bin_2D_data(k_z_sorted,n_bins_perp),bin_2D_data(k_perp_sorted,n_bins_perp)

    def get_flux_power_3D_unique(self, k_box, norm=True):
        flux_power = self.get_flux_power_3D(norm)[0]
        k_unique = np.unique(k_box)
        power_unique = np.zeros_like(k_unique.value)
        for i in range(k_unique.shape[0]):
            print("Binning 3D power according to unique value of |k| #%i/%i" %(i+1,k_unique.shape[0]))
            power_unique[i] = np.mean(flux_power[k_box == k_unique[i]])
        return power_unique, k_unique

    def get_flux_power_3D_sorted(self,k_box,norm=True,mu_box=None):
        k_argsort = np.argsort(k_box, axis=None)
        flux_power = self.get_flux_power_3D(norm)[0]
        if mu_box == None:
            return sort_3D_to_1D(flux_power, k_argsort)[1:], sort_3D_to_1D(k_box, k_argsort)[1:]
        else:
            return sort_3D_to_1D(flux_power, k_argsort)[1:],sort_3D_to_1D(k_box, k_argsort)[1:],sort_3D_to_1D(mu_box, k_argsort)[1:]

    def get_flux_power_3D_binned(self,k_box,n_bins,norm=True):
        power_sorted, k_sorted = self.get_flux_power_3D_sorted(k_box,norm)
        return bin_1D_data(power_sorted, n_bins), bin_1D_data(k_sorted, n_bins)

    def _form_return_list(self,x,y,n_bins_x,n_bins_y,norm,bin_coord_x,bin_coord_y,count,std_err):
        flux_power = self.get_flux_power_3D(norm)[0].flatten()[1:]
        return_list = [None]*(1+bin_coord_x+bin_coord_y+count+std_err) #Number of calculations
        return_list[0] = bin_f_x_y_histogram(x,y,flux_power,n_bins_x,n_bins_y) #Always bin power
        i=1
        if bin_coord_x == True:
            return_list[i] = bin_f_x_y_histogram(x,y,x,n_bins_x,n_bins_y)
            i+=1
        if bin_coord_y == True:
            return_list[i] = bin_f_x_y_histogram(x,y,y,n_bins_x,n_bins_y)
            i+=1
        if count == True:
            return_list[i] = bin_f_x_y_histogram_count(x,y,flux_power,n_bins_x,n_bins_y)
            i+=1
        if std_err == True:
            return_list[i] = bin_f_x_y_histogram_standard_error(x,y,flux_power,n_bins_x,n_bins_y)
        return return_list

    def get_flux_power_3D_two_coords_hist_binned(self,coord_box1,coord_box2,n_bins1,n_bins2,norm=True,bin_coord1=True,bin_coord2=True,count=True,std_err=True):
        x = coord_box1.flatten()[1:] #USE BINNUMBER!!!
        y = coord_box2.flatten()[1:]
        return self._form_return_list(x,y,n_bins1,n_bins2,norm,bin_coord1,bin_coord2,count,std_err)

    def get_flux_power_legendre_integrand(self,k_box,mu_box,n_bins,norm=True): #NEEDS TIDYING-UP!!!
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

    def get_flux_power_3D_multipole(self,multipole,k_box,mu_box,n_bins,norm=True):
        power_mu_sorted, k_mu_sorted, mu_mu_sorted = self.get_flux_power_legendre_integrand(k_box,mu_box,n_bins,norm)
        total_integrand = power_mu_sorted * evaluate_legendre_polynomial(mu_mu_sorted,multipole)
        power_integrated = np.trapz(total_integrand,x=mu_mu_sorted) * ((2.*multipole + 1.) / 2.)
        return power_integrated, np.mean(k_mu_sorted,axis=-1), power_mu_sorted #P_l(mod(k)), mod(k)
