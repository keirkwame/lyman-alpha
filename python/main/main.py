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
import os
import sys

from utils import *
from power_spectra import *
from boxes import *
from fourier_estimators import *

def snapshot_to_boxes(snap_num,snap_dir,grid_samps,spectrum_resolution,reload_snapshot=True,spec_root='gridded_spectra',spectra_savedir=None,mean_flux_desired=None):
    box_instance = SimulationBox(snap_num,snap_dir,grid_samps,spectrum_resolution,reload_snapshot=reload_snapshot,spectra_savefile_root=spec_root,spectra_savedir=spectra_savedir)
    box_instance.convert_fourier_units_to_distance = True
    print(box_instance._n_samp)
    print(box_instance.k_i('z')[1], np.max(box_instance.k_i('z')))
    return box_instance.skewers_realisation(mean_flux_desired=mean_flux_desired), box_instance.k_box(), box_instance.mu_box(), box_instance


#Get random Gaussian realisations
def _get_gaussian_box_instance(box_size, n_samp, redshift, H0, omega_m):
    box_instance = GaussianBox(box_size, n_samp, redshift, H0, omega_m)
    box_instance.convert_fourier_units_to_distance = True
    return box_instance

def isotropic_power_law_power_spectrum_to_boxes(pow_index, pow_pivot, pow_amp, box_size, n_samp, redshift, H0, omega_m):
    box_instance = _get_gaussian_box_instance(box_size, n_samp, redshift, H0, omega_m)
    print(box_instance.k_i('z'))
    return box_instance.isotropic_power_law_gauss_realisation(pow_index,pow_pivot,pow_amp),box_instance.k_box(),box_instance.mu_box(),box_instance

def anisotropic_power_law_power_spectrum_to_boxes(pow_index, pow_pivot, pow_amp, mu_coefficients, box_size, n_samp, redshift, H0, omega_m):
    box_instance = _get_gaussian_box_instance(box_size, n_samp, redshift, H0, omega_m)
    print(np.max(box_instance.k_i('z')))
    return box_instance.anisotropic_power_law_gauss_realisation(pow_index,pow_pivot,pow_amp,mu_coefficients),box_instance.k_box(),box_instance.mu_box(),box_instance

def anisotropic_pre_computed_power_spectrum_to_boxes(fname,mu_coeffs,box_size,n_samp,redshift,H0,omega_m):
    box_instance = _get_gaussian_box_instance(box_size, n_samp, redshift, H0, omega_m)
    k_box = box_instance.k_box()
    mu_box = box_instance.mu_box()
    print(box_instance.k_i('z')[1], np.max(box_instance.k_i('z')))
    return box_instance.anisotropic_pre_computed_gauss_realisation(fname,mu_coeffs),k_box,mu_box, box_instance


#Get Fourier estimates of power spectra
def boxes_to_power_3D_binned(simu_box,k_box,n_bins,norm=True):
    power_instance = FourierEstimator3D(simu_box)
    return power_instance.get_flux_power_3D_binned(k_box,n_bins,norm=norm)

def boxes_to_power_3D_mod_k_unique(simu_box,k_box,norm=True):
    power_instance = FourierEstimator3D(simu_box)
    return power_instance.get_flux_power_3D_unique(k_box, norm=norm)

def boxes_to_power_3D_cylindrical_binned(simu_box,k_z_box,k_perp_box,n_bins_z,n_bins_perp,norm=True):
    power_instance = FourierEstimator3D(simu_box)
    return power_instance.get_flux_power_3D_cylindrical_coords(k_z_box,k_perp_box,n_bins_z,n_bins_perp,norm=norm)

def boxes_to_power_3D_multipole(multipole,simu_box,k_box,mu_box,n_bins,norm=True):
    power_instance = FourierEstimator3D(simu_box)
    return power_instance.get_flux_power_3D_multipole(multipole, k_box, mu_box, n_bins, norm=norm)


if __name__ == "__main__":
    """Input arguments: Snapshot directory path; Snapshot number; grid_samps; Resolution of spectrum in km s^{-1};
    Spectra directory path (with '/snapdir_XXX' if necessary); Full path to fiducial cosmology datafile"""
    snap_dir = sys.argv[1]
    snap_num = int(sys.argv[2])
    grid_samps = int(sys.argv[3])
    spectrum_resolution = float(sys.argv[4])*(u.km / u.s)
    spectra_savedir = sys.argv[5]
    fiducial_cosmology_fname = sys.argv[6]
    reload_snapshot = False
    spec_root = 'gridded_spectra_DLAs_dodged'
    mean_flux = 0.66932662196737913 #0.75232943916324291 #0.36000591326127357 #None
    n_bins_k = 10 ** (np.linspace(-1.23,1.52,16) - 0) #3) #1/Mpc #TEST ADDING UNITS!
    n_bins_mu = np.linspace(0.,1.,8)
    norm = True

    #Test Gaussian realisations input
    '''pow_index = 0.
    pow_pivot = 1. / u.Mpc
    pow_amp = 1.
    box_size = {'x': 35.5 * u.Mpc, 'y': 35.5 * u.Mpc, 'z': 35.5 * u.Mpc}
    n_samp = {'x': 250, 'y': 250, 'z': 117}
    redshift = 2.499
    H0 = (70.4 * u.km) / (u.s * u.Mpc) #From default CLASS - 67.11
    omega_m = 0.2726 #omega_cdm + omega_b from default CLASS - 0.12029 + 0.022068'''

    #Anisotropic corrections
    def test_mu_coefficients(k_para, k_perp):
        amp = 1.
        mean = 0. / u.Mpc
        stddev = 1. / u.Mpc
        scale_dependence = amp * np.exp((k_para - mean)**2 / (-2. * stddev**2))
        return np.array([1.*scale_dependence,0.*scale_dependence,1.*scale_dependence,0.*scale_dependence,1.*scale_dependence])

    def BOSS_DLA_mu_coefficients(k_para,k_perp):
        b_forest = -0.157 #-0.522 #-0.157 #arxiv:1504.06656 - Blomqvist et al. 2015 data - z=2.3
        beta_forest = 1.39  #arxiv:1504.06656 - Blomqvist et al. 2015 data - z=2.3 (1.4 in sims)
        b_DLA = 0. #-0.03 #BOSS 2017 #2.17*(beta_forest**0.22) #(2.33) arxiv:1209.4596 - Font-Ribera et al. 2012 data - z=?
        beta_DLA = 0. #0.43 #1. / b_DLA #(0.43) arxiv:1209.4596 - Font-Ribera et al. 2012 data - z=?

        stddev = 10. / u.Mpc #10. / u.Mpc
        gamma = 0.1 * u.Mpc #0.1 * u.Mpc

        mean = 0. / u.Mpc
        gaussian_FT = np.exp((k_para - mean)**2 / (-2. * stddev**2))
        lorentzian_FT = np.exp(-1. * mh.pi * gamma * np.absolute(k_para))
        scale_dependence = gaussian_FT * lorentzian_FT #FT[Voigt]

        b_eff = b_forest + (b_DLA*scale_dependence)
        beta_eff = ((b_forest*beta_forest) + (b_DLA*beta_DLA*scale_dependence)) / (b_forest + (b_DLA*scale_dependence))

        mu_coeffs = np.array([b_eff**2]*5)
        mu_coeffs[0] *= beta_eff**2 #mu^4
        mu_coeffs[1] *= 0.
        mu_coeffs[2] *= 2. * beta_eff
        mu_coeffs[3] *= 0.
        mu_coeffs[4] *= 1. #mu^0

        return mu_coeffs

    #Generate boxes
    #(simu_box,input_k), k_box, mu_box, box_instance = anisotropic_pre_computed_power_spectrum_to_boxes(fiducial_cosmology_fname, BOSS_DLA_mu_coefficients, box_size, n_samp, redshift, H0, omega_m)
    #(simu_box,input_k), k_box, mu_box, box_instance = anisotropic_power_law_power_spectrum_to_boxes(pow_index,pow_pivot,pow_amp,BOSS_DLA_mu_coefficients,box_size, n_samp, redshift, H0, omega_m)
    simu_box2,k_box,mu_box,box_instance = snapshot_to_boxes(snap_num, snap_dir, grid_samps, spectrum_resolution, reload_snapshot,spec_root,spectra_savedir,mean_flux_desired=mean_flux)

    #Add Voigt profiles
    '''n_voigt = 6250 #1250
    sigma = 10.*(u.km/u.s)
    gamma = 10.*(u.km/u.s)
    amp_voigt = 6.e-4
    wrap_around = 10
    voigt_box = box_instance.add_voigt_profiles(simu_box,n_voigt,sigma,gamma,amp_voigt,wrap_around=wrap_around)[0]
    voigt_only = voigt_box - simu_box'''

    #Estimate power spectra
    fourier_instance = FourierEstimator3D(simu_box2)
    power_binned_k_mu2, k_binned_2D, errorbars2 = fourier_instance.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),n_bins_k,n_bins_mu,bin_coord2=False)

    '''voigt_instance = FourierEstimator3D(voigt_box)
    voigt_power, k, mu = voigt_instance.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),n_bins_k,n_bins_mu)

    voigt_only_instance = FourierEstimator3D(voigt_only)
    voigt_only_power, k2, mu2 = voigt_only_instance.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),n_bins_k,n_bins_mu)

    cross_instance = FourierEstimator3D(simu_box,second_box=voigt_only)
    cross_power, k3, mu3 = cross_instance.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),n_bins_k,n_bins_mu)'''

    #Calculate model power spectra
    #power_instance = PreComputedPowerSpectrum(fiducial_cosmology_fname)
    '''power_instance.set_anisotropic_functional_form(BOSS_DLA_mu_coefficients)
    raw_model_power = power_instance.evaluate3d_anisotropic(k_box,np.absolute(mu_box))'''
    #raw_model_isotropic_power = power_instance.evaluate3d_isotropic(k_box)
    #power_binned_model = bin_f_x_y_histogram(k_box.flatten()[1:],np.absolute(mu_box).flatten()[1:],raw_model_power.flatten()[1:],n_bins_k,n_bins_mu)
    #power_binned_isotropic_model = bin_f_x_y_histogram(k_box.flatten()[1:],np.absolute(mu_box).flatten()[1:],raw_model_isotropic_power.flatten()[1:],n_bins_k,n_bins_mu)

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'brown'] * 2
