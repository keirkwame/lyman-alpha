import numpy as np
import astropy.units as u
import math as mh
import sys

import boxes as box
import fourier_estimators as fou

if __name__ == "__main__":
    model_cosmology_filename = sys.argv[1]
    save_filename = sys.argv[2]

    #Input parameters
    box_size = {'x': 106.5 * u.Mpc, 'y': 106.5 * u.Mpc, 'z': 106.5 * u.Mpc} # = 75 Mpc / h
    n_samp = {'x': 751, 'y': 751, 'z': 751}
    redshift = 2.44
    H0 = (70.4 * u.km) / (u.s * u.Mpc)
    omega_m = 0.2726
    n_mu_bins = 4
    n_k_bins = 6

    #Input anisotropic functional form
    def mu_coefficients(k_para, k_perp):
        return np.array([0. * k_para.value, 0. * k_para.value, 0. * k_para.value, 0. * k_para.value, 1. + 0. * k_para.value]) #Isotropic limit

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
    mu_bin_edges = np.array([0., 0.5, 0.8, 0.95, 1.])

    #Gaussian boxes
    test_gaussian_box = test_gaussian_ins.anisotropic_pre_computed_gauss_realisation(model_cosmology_filename, mu_coefficients)
    np.save('/home/keir/Data/Illustris_big_box_spectra/snapdir_064/test_gaussian_box_isotropic_751_751_751.npy',test_gaussian_box)

    #Fourier estimator instances
    fourier_estimator_instance = fou.FourierEstimator3D(test_gaussian_box)

    #Power spectra
    power_bin,k_bin,mu_bin,bin_count = fourier_estimator_instance.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),k_bin_edges,mu_bin_edges,std_err=False)

    np.savez(save_filename, power_bin, k_bin, bin_count, mu_bin)