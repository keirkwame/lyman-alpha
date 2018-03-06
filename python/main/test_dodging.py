import numpy as np
import astropy.units as u
import copy as cp
import math as mh
import sys

import boxes as box
import fourier_estimators as fou

if __name__ == "__main__":
    model_cosmology_filename = sys.argv[1]
    cofm_difference_filename = sys.argv[2]
    save_filename = sys.argv[3]

    #Input parameters
    box_size = {'x': 106.5 * u.Mpc, 'y': 106.5 * u.Mpc, 'z': 10.65 * u.Mpc} # = 75 Mpc / h
    n_samp = {'x': 7501, 'y': 751, 'z': 751}
    n_samp_sub_sampled = {'x': 751, 'y': 751, 'z': 751}
    sub_sampling_rate = 10
    redshift = 3.49 #2.44
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

    '''k_box = test_gaussian_ins.k_box()
    np.save('/home/keir/Data/Illustris_big_box_spectra/snapdir_064/k_box_7501_751_751.npy',k_box.value)'''

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
    '''test_gaussian_box = test_gaussian_ins.anisotropic_pre_computed_gauss_realisation(model_cosmology_filename, mu_coefficients)
    np.save('/home/keir/Data/Illustris_big_box_spectra/snapdir_057/test_gaussian_box_isotropic_7501_751_76_z_3_49.npy',test_gaussian_box)'''
    #test_gaussian_box = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/test_gaussian_box_isotropic_7501_751_76.npy') #* 1.e+8
    test_gaussian_box = np.load('/home/keir/Data/Illustris_big_box_spectra/snapdir_064/test_gaussian_box_isotropic_7501_751_76.npy')

    cofm_difference = ((np.load(cofm_difference_filename) / 10.).astype(np.int)).reshape(750,750)
    print(cofm_difference)

    test_gaussian_box_orig = cp.deepcopy(test_gaussian_box)[::sub_sampling_rate,:,:]
    test_gaussian_box_dodged = cp.deepcopy(test_gaussian_box)[::sub_sampling_rate,:,:]

    for i in range(cofm_difference.shape[0]):
        for j in range(cofm_difference.shape[1]):
            if cofm_difference[i,j] > 0:
                test_gaussian_box_dodged[i,j,:] = test_gaussian_box[i * sub_sampling_rate + cofm_difference[i,j], j, :]
                print(i, j, i * sub_sampling_rate, i * sub_sampling_rate + cofm_difference[i,j], test_gaussian_box[i * sub_sampling_rate, j, 0], test_gaussian_box[i * sub_sampling_rate + cofm_difference[i,j], j, 0], test_gaussian_box[i * sub_sampling_rate + cofm_difference[i,j] + 1, j, 0])

    #Fourier estimator instances
    fourier_estimator_instance = fou.FourierEstimator3D(test_gaussian_box_orig)
    fourier_estimator_instance_dodged = fou.FourierEstimator3D(test_gaussian_box_dodged)

    #Power spectra
    power_bin,k_bin,bin_count = fourier_estimator_instance.get_power_3D_two_coords_binned(k_box, np.absolute(mu_box), k_bin_edges, mu_bin_edges, bin_coord2=False, std_err=False)
    power_bin_dodged = fourier_estimator_instance_dodged.get_power_3D_two_coords_binned(k_box, np.absolute(mu_box), k_bin_edges, mu_bin_edges, bin_coord1=False, bin_coord2=False, count=False, std_err=False)

    '''power_unbinned = fourier_estimator_instance.get_flux_power_3D()
    power_unbinned_dodged = fourier_estimator_instance_dodged.get_flux_power_3D()'''

    norm_fac = box_size['x'] * box_size['y'] * box_size['z']
    np.savez(save_filename, power_bin, k_bin, bin_count, power_bin_dodged) #, power_unbinned, k_box.value, mu_box.value, power_unbinned_dodged)
