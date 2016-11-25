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

from power_spectra import *
from utils import *

class Box(object):
    """Class to generate a box of fluctuations"""
    def __init__(self,redshift,H0,omega_m):
        self._redshift = redshift
        self._H0 = H0
        self._omega_m = omega_m
        self.convert_fourier_units_to_distance = False

    def k_i(self,i):
        if self.convert_fourier_units_to_distance == False:
            box_units = self.voxel_velocities[i]
        else:
            box_units = self.voxel_lens[i]
        return np.fft.fftfreq(self._n_samp[i], d=box_units)

    def k_z_mod_box(self): #Generalise to any k_i
        x = np.zeros_like(self.k_i('x'))[:, np.newaxis, np.newaxis]
        y = np.zeros_like(self.k_i('y'))[np.newaxis, :, np.newaxis]
        z = self.k_i('z')[np.newaxis, np.newaxis, :]
        return x + y + np.absolute(z)

    def k_perp_box(self): #Generalise to any pair of k_i
        x = self.k_i('x')[:, np.newaxis, np.newaxis]
        y = self.k_i('y')[np.newaxis, :, np.newaxis]
        z = np.zeros_like(self.k_i('z'))[np.newaxis, np.newaxis, :]
        return np.sqrt(x**2 + y**2) + z

    def k_box(self):
        x = self.k_i('x')[:,np.newaxis,np.newaxis]
        y = self.k_i('y')[np.newaxis,:,np.newaxis]
        z = self.k_i('z')[np.newaxis,np.newaxis,:]
        return np.sqrt(x**2 + y**2 + z**2)

    def mu_box(self):
        x = self.k_i('x')[:, np.newaxis, np.newaxis]
        y = self.k_i('y')[np.newaxis, :, np.newaxis]
        z = self.k_i('z')[np.newaxis, np.newaxis, :]
        return z / np.sqrt(x**2 + y**2 + z**2)

    def hubble_z(self):
        return self._H0 * np.sqrt(self._omega_m * (1 + self._redshift) ** 3 + 1. - self._omega_m)


class GaussianBox(Box):
    """Sub-class to generate a box of fluctuations from a Gaussian random field"""
    def __init__(self,x_max,n_samp,redshift,H0,omega_m):
        self._x_max = x_max #Tuples for 3 dimensions
        self._n_samp = n_samp
        super(GaussianBox, self).__init__(redshift,H0,omega_m)

        self.voxel_lens = {}
        self.voxel_velocities = {}
        for i in ['x','y','z']:
            self.voxel_lens[i] = self._x_max[i] / (self._n_samp[i] - 1)
            self.voxel_velocities[i] = self.voxel_lens[i] * self.hubble_z()

    def _gauss_realisation(self, power_spectrum_instance):
        k_box = self.k_box()
        mu_box = self.mu_box()
        gauss_k=np.sqrt(0.5*power_spectrum_instance.evaluate3d(k_box,mu_box))*(npr.standard_normal(size=k_box.shape)+npr.standard_normal(size=k_box.shape)*1.j)
        gauss_k[k_box == 0.] = 0.  # Zeroing the mean
        return np.fft.ifftn(gauss_k, s=(self._n_samp['x'], self._n_samp['y'], self._n_samp['z']), axes=(0, 1, 2))

    def isotropic_power_law_gauss_realisation(self,pow_index,pow_pivot,pow_amp):
        box_spectra = IsotropicPowerLawPowerSpectrum(pow_index, pow_pivot, pow_amp)
        return self._gauss_realisation(box_spectra)

    def anisotropic_power_law_gauss_realisation(self, pow_index, pow_pivot, pow_amp, mu_coefficients):
        box_spectra = AnisotropicPowerLawPowerSpectrum(pow_index, pow_pivot, pow_amp, mu_coefficients)
        return self._gauss_realisation(box_spectra)

    def isotropic_CAMB_gauss_realisation(self):
        return 0

    def anisotropic_CAMB_gauss_realisation(self):
        return 0


class SimulationBox(Box):
    """Sub-class to generate a box of Lyman-alpha spectra drawn from Simeon's simulations"""
    def __init__(self,snap_num,snap_dir,grid_samps,spectrum_resolution,reload_snapshot=True):
        self._n_samp = {}
        self._n_samp['x'] = grid_samps
        self._n_samp['y'] = grid_samps

        self.voxel_lens = {}
        self.voxel_velocities = {}

        self._snap_num = snap_num
        self._snap_dir = snap_dir
        self._grid_samps = grid_samps
        self._spectrum_resolution = spectrum_resolution
        self._reload_snapshot = reload_snapshot

        self.spectra_savefile = 'gridded_spectra_%i_%i.hdf5' %(self._grid_samps,self._spectrum_resolution.value)

        self.element = 'H'
        self.ion = 1
        self.line_wavelength = 1215 * u.angstrom

        gr = gs.GriddedSpectra(self._snap_num, self._snap_dir, nspec=self._grid_samps, res=self._spectrum_resolution.value,savefile=self.spectra_savefile,reload_file=self._reload_snapshot)
        self._n_samp['z'] = int(gr.vmax / gr.dvbin)
        #Planck 2015 parameters - how to read from simulations???
        H0 = (67.31 * u.km) / (u.s * u.Mpc)
        omega_m = 0.3149
        super(SimulationBox, self).__init__(gr.red, H0, omega_m)
        self.voxel_velocities['x'] = (gr.vmax / self._n_samp['x']) * (u.km / u.s)
        self.voxel_velocities['y'] = (gr.vmax / self._n_samp['y']) * (u.km / u.s)
        self.voxel_velocities['z'] = gr.dvbin * (u.km / u.s)
        print("Size of voxels in velocity units =", self.voxel_velocities) #, "\n")
        for i in ['x','y','z']:
            self.voxel_lens[i] = self.voxel_velocities[i] / self.hubble_z()

    def skewers_realisation(self):
        gr = gs.GriddedSpectra(self._snap_num,self._snap_dir,nspec=self._grid_samps,res=self._spectrum_resolution.value,savefile=self.spectra_savefile,reload_file=self._reload_snapshot)
        tau = gr.get_tau(self.element,self.ion,int(self.line_wavelength.value)) #SLOW if not reloading
        gr.save_file() #Save spectra to file
        delta_flux = np.exp(-1.*tau) / np.mean(np.exp(-1.*tau)) - 1.
        return delta_flux.reshape((self._grid_samps,self._grid_samps,-1))