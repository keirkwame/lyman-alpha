# -*- coding: utf-8 -*-

import numpy as np
import astropy.units as u
import math as mh
import sys

import power_spectra as pos
import boxes as box
import fourier_estimators as fou
import utils as uti

if __name__ == "__main__":
    model_cosmology_filename = sys.argv[1]
    save_filename = sys.argv[2]

    #Input parameters
    box_size = {'x': 106.5 * u.Mpc, 'y': 106.5 * u.Mpc, 'z': 106.5 * u.Mpc} # = 75 Mpc / h
    n_samp = {'x': 21, 'y': 21, 'z': 21}
    redshift = 2.44
    H0 = (70.4 * u.km) / (u.s * u.Mpc)
    omega_m = 0.2726
    n_mu_bins = 8
    n_k_bins = 15

    #Input anisotropic functional form
    def mu_coefficients(k_para, k_perp):
        #return np.array([0. * k_para.value, 0. * k_para.value, 0. * k_para.value, 0. * k_para.value, 1. + 0. * k_para.value]) #Isotropic limit
        b = -0.122
        beta = 1.663
        #return np.array([0. * k_para.value, 0. * k_para.value, 0. * k_para.value, 0. * k_para.value, (b**2) + (0. * k_para.value)]) #Isotropic biased limit
        #return np.array([((b**2) * (beta**2)) + (0 * k_para.value), 0. * k_para.value, (2. * beta * (b**2)) + (0. * k_para.value), 0. * k_para.value, (b**2) + (0. * k_para.value)]) #Anisotropic biased limit

        #BOSS model
        b_HCD = -0.0288
        beta_HCD = 0.681
        L_HCD = 3.5 / 0.704 #Mpc

        F_HCD = np.sinc(k_para.value * L_HCD / mh.pi) #** 2 #1. + (0. * k_para.value)

        mu0_term = (2. * b * b_HCD * F_HCD) + ((b_HCD**2) * (F_HCD**2))
        mu2_term = (2. * b * b_HCD * F_HCD * (beta + beta_HCD)) + ((b_HCD**2) * (F_HCD**2) * 2. * beta_HCD)
        mu4_term = (2. * b * b_HCD * F_HCD * beta * beta_HCD) + ((b_HCD**2) * (F_HCD**2) * (beta_HCD**2))

        return np.array([mu4_term, 0. * k_para.value, mu2_term, 0. * k_para.value, mu0_term])

    #Gaussian box instances
    test_gaussian_ins = box.GaussianBox(box_size, n_samp, redshift, H0, omega_m)
    test_gaussian_ins.convert_fourier_units_to_distance = True

    #Co-ordinate boxes
    k_box = test_gaussian_ins.k_box()
    mu_box = test_gaussian_ins.mu_box()

    #Binning
    k_min = np.min(k_box[k_box > 0. / u.Mpc])
    k_max = k_max = 0.704 / u.Mpc #np.max(k_box)
    k_bin_max = mh.exp(mh.log(k_max.value) + ((mh.log(k_max.value) - mh.log(k_min.value)) / (n_k_bins - 1))) / u.Mpc
    k_bin_edges = np.exp(np.linspace(mh.log(k_min.value), mh.log(k_bin_max.value), n_k_bins + 1)) / u.Mpc
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)
    #mu_bin_edges = np.array([0., 0.5, 0.8, 0.95, 1.])

    print('Here')
    #Gaussian boxes
    test_gaussian_box = test_gaussian_ins.anisotropic_power_law_gauss_realisation(-3.,0.5 / u.Mpc,1.,mu_coefficients) #anisotropic_pre_computed_gauss_realisation(model_cosmology_filename, mu_coefficients)
    print('Here2')
    np.save('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/test_gaussian_box_anisotropic_scaleDepbiased3point5_minus3_power_21_21_21_num1.npy',test_gaussian_box)
    #test_gaussian_box = np.load('/home/keir/Data/Illustris_big_box_spectra/snapdir_064/test_gaussian_box_isotropic_751_751_751.npy')

    print('Here3')
    #Fourier estimator instances
    fourier_estimator_instance = fou.FourierEstimator3D(test_gaussian_box)

    #Bin theory power spectra
    power_spectrum_instance = pos.PowerLawPowerSpectrum(-3.,0.5 / u.Mpc,1.)
    power_theory_box = power_spectrum_instance.evaluate3d_isotropic(k_box)
    power_theory_binned = uti.bin_f_x_y_histogram(k_box.flatten()[1:],np.absolute(mu_box).flatten()[1:],power_theory_box.flatten()[1:],k_bin_edges,mu_bin_edges)

    #Power spectra
    power_bin,k_bin,mu_bin,bin_count = fourier_estimator_instance.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),k_bin_edges,mu_bin_edges,std_err=False)
    #power_raw = fourier_estimator_instance.get_flux_power_3D()[0]

    np.savez(save_filename, power_bin, k_bin, bin_count, mu_bin, power_theory_binned)
    #np.savez(save_filename, power_raw, k_box.value, np.absolute(mu_box.value))
