import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import scipy.integrate as spi
import scipy.optimize as spo
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
    box_instance.convert_fourier_units_to_distance = False
    '''print(box_instance._n_samp)
    print(box_instance.k_i('z')[1], np.max(box_instance.k_i('z')))'''
    return box_instance.skewers_realisation(mean_flux_specified=mean_flux_desired) #, box_instance.k_box(), box_instance.mu_box(), box_instance


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
    return box_instance.anisotropic_pre_computed_gauss_realisation(fname,mu_coeffs),k_box,mu_box,box_instance


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
    spec_root = 'gridded_spectra_DLAs_LLS_dodged'
    mean_flux =  #None #0.68299813251533592 #0.66932662196737913 #0.75232943916324291 #0.36000591326127357 #None
    norm = True

    #Test Gaussian realisations input
    '''pow_index = 0.
    pow_pivot = 1. / u.Mpc
    pow_amp = 1.'''
    '''box_size = {'x': 106.5 * u.Mpc, 'y': 106.5 * u.Mpc, 'z': 106.5 * u.Mpc}
    n_samp = {'x': 751, 'y': 751, 'z': 301}
    redshift = 2.444
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
        b_forest = -0.178 #-0.522 #-0.157 #arxiv:1504.06656 - Blomqvist et al. 2015 data - z=2.3
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
    #simu_box, k_box, mu_box, box_instance = anisotropic_pre_computed_power_spectrum_to_boxes(fiducial_cosmology_fname, BOSS_DLA_mu_coefficients, box_size, n_samp, redshift, H0, omega_m)
    #(simu_box,input_k), k_box, mu_box, box_instance = anisotropic_power_law_power_spectrum_to_boxes(pow_index,pow_pivot,pow_amp,BOSS_DLA_mu_coefficients,box_size, n_samp, redshift, H0, omega_m)
    simu_box = snapshot_to_boxes(snap_num, snap_dir, grid_samps, spectrum_resolution, reload_snapshot,spec_root,spectra_savedir,mean_flux_desired=mean_flux)
    #simu_box,k_box,mu_box,box_instance_with_DLA

    #Column density distribution
    '''max_col_dens_100kms = box_instance_with_DLA.max_local_sum_of_column_density_in_each_skewer()

    #Masks
    mask_large_dla = max_col_dens_100kms > 1.e+21 / (u.cm * u.cm)
    mask_small_dla = (max_col_dens_100kms > 2.e+20 / (u.cm * u.cm)) * (max_col_dens_100kms <= 1.e+21 / (u.cm * u.cm))
    mask_sub_dla = (max_col_dens_100kms > 1.e+19 / (u.cm * u.cm)) * (max_col_dens_100kms <= 2.e+20 / (u.cm * u.cm))
    mask_lls = (max_col_dens_100kms > 1.6e+17 / (u.cm * u.cm)) * (max_col_dens_100kms <= 1.e+19 / (u.cm * u.cm))
    mask_forest = (max_col_dens_100kms > 0. / (u.cm * u.cm)) * (max_col_dens_100kms <= 1.6e+17 / (u.cm * u.cm))

    #Delta fluxes
    mean_flux_whole_box = np.mean(np.exp(-1.*box_instance_with_DLA.get_optical_depth()))'''

    '''simu_box_large_dla = box_instance_with_DLA.skewers_realisation_subset(mask_large_dla,mean_flux_specified=mean_flux_whole_box)
    simu_box_small_dla = box_instance_with_DLA.skewers_realisation_subset(mask_small_dla,mean_flux_specified=mean_flux_whole_box)
    simu_box_sub_dla = box_instance_with_DLA.skewers_realisation_subset(mask_sub_dla,mean_flux_specified=mean_flux_whole_box)
    simu_box_lls = box_instance_with_DLA.skewers_realisation_subset(mask_lls,mean_flux_specified=mean_flux_whole_box)'''
    #simu_box_forest = box_instance_with_DLA.skewers_realisation_subset(mask_forest,mean_flux_specified=mean_flux_whole_box)

    #1D powers
    '''power_total_instance = FourierEstimator1D(simu_box)
    power_total = power_total_instance.get_flux_power_1D()

    power_large_dla_instance = FourierEstimator1D(simu_box_large_dla)
    power_large_dla = power_large_dla_instance.get_flux_power_1D()
    power_small_dla_instance = FourierEstimator1D(simu_box_small_dla)
    power_small_dla = power_small_dla_instance.get_flux_power_1D()
    power_sub_dla_instance = FourierEstimator1D(simu_box_sub_dla)
    power_sub_dla = power_sub_dla_instance.get_flux_power_1D()
    power_lls_instance = FourierEstimator1D(simu_box_lls)
    power_lls = power_lls_instance.get_flux_power_1D()'''
    power_forest_instance = FourierEstimator1D(simu_box) #_forest)
    power_forest = power_forest_instance.get_flux_power_1D()

    '''power_hcd = ((power_lls * np.sum(mask_lls)) + (power_sub_dla * np.sum(mask_sub_dla)) + (power_small_dla * np.sum(mask_small_dla)) + (power_large_dla * np.sum(mask_large_dla))) / (mask_forest.size - np.sum(mask_forest))
    power_recon = ((power_hcd * (mask_forest.size - np.sum(mask_forest))) + (power_forest * np.sum(mask_forest))) / mask_forest.size'''

    k_z_mod = box_instance_with_DLA.k_z_mod()

    #Saving power spectra
    power_fname = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/contaminant_power_1D_z_2_44_750_10_DLAs_LLS_dodged.npy'
    np.save(power_fname,np.vstack((k_z_mod.value,power_forest))) #power_total,power_forest,power_lls,power_sub_dla,power_small_dla,power_large_dla)))'''

    #Template fit
    '''def hcd_model(k_z_mod, a, b, c):
        return (1. / (((a * np.exp(b * k_z_mod)) - 1.)**2)) + c
        #return (1. / ((k_z_mod + b)**a)) + c

    best_fit_params_large_dla, covar_large_dla = spo.curve_fit(hcd_model, k_z_mod[1:], power_large_dla[1:] / power_forest[1:])
    best_fit_params_small_dla, covar_small_dla = spo.curve_fit(hcd_model, k_z_mod[1:], power_small_dla[1:] / power_forest[1:])
    best_fit_params_sub_dla, covar_sub_dla = spo.curve_fit(hcd_model, k_z_mod[1:], power_sub_dla[1:] / power_forest[1:])
    best_fit_params_lls, covar_lls = spo.curve_fit(hcd_model, k_z_mod[1:], power_lls[1:] / power_forest[1:])

    #Model comparison
    def mcdonald(k):
        return 0.2 * ((1. / ((15000 * k) - 8.9)) + 0.018)'''

    #Binning
    '''n_bins_mu = 8
    n_bins_k = 15

    k_min = np.min(k_box[k_box > 0. / u.Mpc]) #2. * mh.pi * 0.704 / 75.
    k_max = np.max(k_box) #2. * mh.pi * 0.704 * 150. * mh.sqrt(3.) / 75.
    k_bin_max = mh.exp(mh.log(k_max.value) + ((mh.log(k_max.value) - mh.log(k_min.value)) / (n_bins_k - 1))) / u.Mpc

    k_bin_edges = np.exp(np.linspace(mh.log(k_min.value), mh.log(k_bin_max.value), n_bins_k + 1)) / u.Mpc
    k_bin_edges[-2] = k_max #HACK TO FIX BINNING OF NYQUIST FREQUENCY
    #-1.229308735891473,1.185343150524040,301)) #-1.23,1.19,16)) #-0.75,1.52,16 #-1.23,1.52,16) - 0) #3) #1/Mpc #TEST ADDING UNITS!
    mu_bin_edges = np.linspace(0., 1., n_bins_mu + 1)'''
    '''n_bins_mu = 1
    n_bins_k = 15
    k_min = np.min(k_box[k_box > 0. / u.Mpc])
    k_max = np.max(k_box)
    k_bin_max = mh.exp(mh.log(k_max.value) + ((mh.log(k_max.value) - mh.log(k_min.value)) / (n_bins_k - 1))) / u.Mpc
    k_bin_edges = np.exp(np.linspace(mh.log(k_min.value), mh.log(k_bin_max.value), n_bins_k + 1)) / u.Mpc
    k_bin_edges[-2] = k_max
    mu_bin_edges = np.linspace(0., 1., n_bins_mu + 1)'''

    #Add Voigt profiles
    '''n_voigt = 6250 #1250
    sigma = 10.*(u.km/u.s)
    gamma = 10.*(u.km/u.s)
    amp_voigt = 6.e-4
    wrap_around = 10
    voigt_box = box_instance.add_voigt_profiles(simu_box,n_voigt,sigma,gamma,amp_voigt,wrap_around=wrap_around)[0]
    voigt_only = voigt_box - simu_box'''

    #Estimate power spectra
    '''fourier_instance = FourierEstimator3D(simu_box)
    power_binned_k_mu, k_binned_2D = fourier_instance.get_flux_power_3D_two_coords_hist_binned(k_box, np.absolute(mu_box), k_bin_edges, mu_bin_edges, bin_coord2=False, count=False, std_err=False, norm=norm)
    power_3D_fname = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/flux_power_3D_k_Mpc_pow_fourier_75_Mpc_h_750_10.npy'
    np.save(power_3D_fname,np.vstack((k_binned_2D[:,0],power_binned_k_mu[:,0])))'''

    #Estimate 1D power spectra
    '''k_z_mod = box_instance_with_DLA.k_z_mod()
    fourier_instance_1D = FourierEstimator1D(simu_box)
    power_1D_with_DLA = fourier_instance_1D.get_flux_power_1D()'''

    '''simu_box_without_DLA = box_instance_with_DLA.skewers_realisation_without_DLAs(mean_flux_desired=mean_flux)
    fourier_instance_1D_without_DLA = FourierEstimator1D(simu_box_without_DLA)
    power_1D_without_DLA = fourier_instance_1D_without_DLA.get_flux_power_1D()

    simu_box_with_DLA_only = box_instance_with_DLA.skewers_realisation_with_DLAs_only(mean_flux_desired=mean_flux)
    fourier_instance_1D_with_DLA_only = FourierEstimator1D(simu_box_with_DLA_only)
    power_1D_with_DLA_only = fourier_instance_1D_with_DLA_only.get_flux_power_1D()


    #Dodged spectra
    spec_root = 'gridded_spectra_DLAs_dodged'
    simu_box_dodged,k_box,mu_box,box_instance_dodged = snapshot_to_boxes(snap_num, snap_dir, grid_samps, spectrum_resolution, reload_snapshot,spec_root,spectra_savedir,mean_flux_desired=mean_flux)

    fourier_instance_1D_dodged = FourierEstimator1D(simu_box_dodged)
    power_1D_dodged = fourier_instance_1D_dodged.get_flux_power_1D()

    simu_box_dodged_unmoved = box_instance_dodged.skewers_realisation_without_DLAs(mean_flux_desired=mean_flux,skewers_with_DLAs_bool_arr=box_instance_with_DLA._get_skewers_with_DLAs_bool_arr(box_instance_with_DLA.get_column_density()))
    fourier_instance_1D_dodged_unmoved = FourierEstimator1D(simu_box_dodged_unmoved)
    power_1D_dodged_unmoved = fourier_instance_1D_dodged_unmoved.get_flux_power_1D()

    simu_box_dodged_moved = box_instance_dodged.skewers_realisation_with_DLAs_only(mean_flux_desired=mean_flux,skewers_with_DLAs_bool_arr=box_instance_with_DLA._get_skewers_with_DLAs_bool_arr(box_instance_with_DLA.get_column_density()))
    fourier_instance_1D_dodged_moved = FourierEstimator1D(simu_box_dodged_moved)
    power_1D_dodged_moved = fourier_instance_1D_dodged_moved.get_flux_power_1D()'''

    #Load Flux power spectra
    '''power_loaded = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_flux.npz')
    power_binned = power_loaded['arr_0']
    k_binned_2D = power_loaded['arr_1']
    bin_counts = power_loaded['arr_2']
    model_power_binned = power_loaded['arr_3']

    #Load Hydrogen power spectra
    power_hydrogen_loaded = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_neutral_hydrogen.npz')
    hydrogen_power_binned = power_hydrogen_loaded['arr_0']
    k_binned_2D_2 = power_hydrogen_loaded['arr_1']
    bin_counts_2 = power_hydrogen_loaded['arr_2']
    npt.assert_array_equal(k_binned_2D_2,k_binned_2D)
    npt.assert_array_equal(bin_counts_2,bin_counts)

    #Load Total Hydrogen power spectra
    #Load Hydrogen power spectra
    power_total_hydrogen_loaded = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power.npz')
    total_hydrogen_power_binned = power_total_hydrogen_loaded['arr_0']
    k_binned_2D_3 = power_total_hydrogen_loaded['arr_1']
    bin_counts_3 = power_total_hydrogen_loaded['arr_2']
    npt.assert_array_equal(k_binned_2D_3,k_binned_2D)
    npt.assert_array_equal(bin_counts_3,bin_counts)

    #Ratios
    min_non_linear_k_bin = 5
    norm_fac = (box_instance.spectra_instance.box / 1000.) ** 3
    power_ratio_flux_model = power_binned * norm_fac / model_power_binned
    power_ratio_flux_total_hydrogen = power_binned / total_hydrogen_power_binned
    power_ratio_combined = np.concatenate((power_ratio_flux_total_hydrogen[:min_non_linear_k_bin], power_ratio_flux_model[min_non_linear_k_bin:]))'''

    #HACK TO FIX BINNING OF NYQUIST FREQUENCY
    '''bin_counts[-2,4] = bin_counts[-2,4] + 1
    bin_counts[-1,4] = bin_counts[-1,4] - 1
    power_binned[-1,4] = np.nan
    k_binned_2D[-1,4] = np.nan'''

    #Load GenPK power spectra
    '''genpk_raw_data = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/PK-DM-snap_064')
    genpk_baryon_raw_data = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/PK-by-snap_064')
    npt.assert_array_equal(genpk_baryon_raw_data[:,np.array([0,2,3])],genpk_raw_data[:,np.array([0,2,3])])

    genpk_power = [None] * n_bins_mu
    genpk_baryon_power = [None] * n_bins_mu
    genpk_k = [None] * n_bins_mu
    genpk_counts = [None] * n_bins_mu
    genpk_counts_mu_sum = [None] * n_bins_mu
    genpk_counts_total = 0.
    for i in range(n_bins_mu):
        genpk_mu_bool_array_geq_cond = genpk_raw_data[:,3] >= mu_bin_edges[i] #i / n_bins_mu
        if i <  n_bins_mu - 1:
            genpk_mu_bool_array = genpk_mu_bool_array_geq_cond * (genpk_raw_data[:,3] < mu_bin_edges[i+1]) #(i + 1) / n_bins_mu)
        else:
            genpk_mu_bool_array = genpk_mu_bool_array_geq_cond * (genpk_raw_data[:, 3] <= 1.)
        genpk_power[i] = genpk_raw_data[genpk_mu_bool_array][:,1] * ((box_instance.spectra_instance.box / 1000.)**3.) #Mpc^3 / h^3
        genpk_baryon_power[i] = genpk_baryon_raw_data[genpk_mu_bool_array][:,1] * ((box_instance.spectra_instance.box / 1000.)**3.) #Mpc^3 / h^3
        genpk_k[i] = genpk_raw_data[genpk_mu_bool_array][:,0] * 2. * mh.pi / (box_instance.spectra_instance.box / 1000.) #h / Mpc
        genpk_counts[i] = genpk_raw_data[genpk_mu_bool_array][:,2]

        print(genpk_counts[i], bin_counts[:,i][bin_counts[:,i] > 0.])
        npt.assert_array_equal(genpk_counts[i], bin_counts[:,i][bin_counts[:,i] > 0.])
        print(genpk_k[i], k_binned_2D[:, i][bin_counts[:, i] > 0.] / box_instance.spectra_instance.hubble)
        npt.assert_allclose(genpk_k[i], k_binned_2D[:, i][bin_counts[:, i] > 0.] / box_instance.spectra_instance.hubble, rtol=1.e-06)
        genpk_counts_mu_sum[i] = np.sum(genpk_counts[i])
        genpk_counts_total += np.sum(genpk_counts[i])

    genpk_counts_k = [None] * n_bins_k
    genpk_counts_k_sum = [None] * n_bins_k
    for i in range(n_bins_k):
        genpk_k_bool_array_geq_cond = genpk_raw_data[:, 0] * 2. * mh.pi / (box_instance.spectra_instance.box / 1000.) >= k_bin_edges[i].value / box_instance.spectra_instance.hubble  # i / n_bins_mu
        if i < n_bins_k - 1:
            genpk_k_bool_array = genpk_k_bool_array_geq_cond * (genpk_raw_data[:, 0] * 2. * mh.pi / (box_instance.spectra_instance.box / 1000.) < k_bin_edges[i + 1].value / box_instance.spectra_instance.hubble)  # (i + 1) / n_bins_mu)
        else:
            genpk_k_bool_array = genpk_k_bool_array_geq_cond * (genpk_raw_data[:, 0] * 2. * mh.pi / (box_instance.spectra_instance.box / 1000.) <= k_bin_edges[i + 1].value / box_instance.spectra_instance.hubble)
        genpk_counts_k[i] = genpk_raw_data[genpk_k_bool_array][:, 2]
        genpk_counts_k_sum[i] = np.sum(genpk_counts_k[i])

    print('Total number of samples = %f, %f, %f' %(genpk_counts_total, np.sum(bin_counts), box_instance._n_samp['x']*box_instance._n_samp['y']*box_instance._n_samp['z'] - 1.))
    '''

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
    #raw_model_isotropic_power = power_instance.evaluate3d_isotropic(k_box / box_instance.spectra_instance.hubble)
    #power_binned_model = bin_f_x_y_histogram(k_box.flatten()[1:],np.absolute(mu_box).flatten()[1:],raw_model_power.flatten()[1:],n_bins_k,n_bins_mu)
    #power_binned_isotropic_model = bin_f_x_y_histogram(k_box.flatten()[1:],np.absolute(mu_box).flatten()[1:],raw_model_isotropic_power.flatten()[1:],k_bin_edges,mu_bin_edges)
    #power_isotropic_model = power_instance.evaluate3d_isotropic(k_binned_2D / box_instance.spectra_instance.hubble / u.Mpc) #Crashes because of nan's

    #colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'brown', 'black'] * 2
