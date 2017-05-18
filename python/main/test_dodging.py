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
    n_samp_sub_sampled = {'x': 751, 'y': 751, 'z': 751}
    sub_sampling_rate = 1 #WILL BE MORE!!!
    redshift = 2.44
    H0 = (70.4 * u.km) / (u.s * u.Mpc)
    omega_m = 0.2726
    n_mu_bins = 4
    n_k_bins = 15

    #Input anisotropic functional form
    def mu_coefficients(k_para, k_perp):
        return np.array([0. * k_para.value, 0. * k_para.value, 0. * k_para.value, 0. * k_para.value, 1. + 0. * k_para.value]) #Isotropic limit

    #Gaussian box instances
    test_gaussian_ins_sub_sampled = box.GaussianBox(box_size, n_samp_sub_sampled, redshift, H0, omega_m)
    test_gaussian_ins_sub_sampled.convert_fourier_units_to_distance = True
    test_gaussian_ins = box.GaussianBox(box_size, n_samp, redshift, H0, omega_m)
    test_gaussian_ins.convert_fourier_units_to_distance = True

    #Co-ordinate boxes
    k_box = test_gaussian_ins_sub_sampled.k_box()
    mu_box = test_gaussian_ins_sub_sampled.mu_box()

    #Binning
    k_min = np.min(k_box[k_box > 0. / u.Mpc])
    k_max = np.max(k_box)
    k_bin_max = mh.exp(mh.log(k_max.value) + ((mh.log(k_max.value) - mh.log(k_min.value)) / (n_k_bins - 1))) / u.Mpc
    k_bin_edges = np.exp(np.linspace(mh.log(k_min.value), mh.log(k_bin_max.value), n_k_bins + 1)) / u.Mpc
    #k_bin_edges[-2] = k_max
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)

    #Gaussian boxes
    test_gaussian_box = test_gaussian_ins.anisotropic_pre_computed_gauss_realisation(model_cosmology_filename, mu_coefficients)
    test_gaussian_box_orig = test_gaussian_box[::sub_sampling_rate,::sub_sampling_rate,:]
    test_gaussian_box_dodged = test_gaussian_box[::sub_sampling_rate,::sub_sampling_rate,:] #WILL ADD COFM!!!

    #Fourier estimator instances
    fourier_estimator_instance = fou.FourierEstimator3D(test_gaussian_box_orig)
    fourier_estimator_instance_dodged = fou.FourierEstimator3D(test_gaussian_box_dodged)

    #Power spectra
    power_bin,k_bin,bin_count = fourier_estimator_instance.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),k_bin_edges,mu_bin_edges,bin_coord2=False,std_err=False)
    power_bin_dodged = fourier_estimator_instance_dodged.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),k_bin_edges,mu_bin_edges,bin_coord1=False,bin_coord2=False,count=False,std_err=False)
    norm_fac = box_size['x'] * box_size['y'] * box_size['z']
    np.savez(save_filename, power_bin, k_bin, bin_count, power_bin_dodged)