import os
import sys
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import astropy.units as u

from main import *
from power_spectra import *
from boxes import *
from fourier_estimators import *
from utils import *

#This is just the very start of the test suite!

#Null tests

def test_pre_computed_power_spectra_no_interpolation_limit():
    fname = os.path.dirname(os.path.abspath(__file__)) + '/P_k_z_4_default_CLASS.dat' #Make auto locate path to datafile
    pre_computed_power_instance = PreComputedPowerSpectrum(fname)
    power_interpolated = pre_computed_power_instance.evaluate3d_isotropic(pre_computed_power_instance.k_raw)
    return npt.assert_allclose(power_interpolated,pre_computed_power_instance.power_raw)

def test_anisotropic_power_law_power_spectra_isotropic_limit():
    test_k = np.arange(1.e-3,1.e+2,1.e-3) / u.Mpc
    isotropic_power_instance = PowerLawPowerSpectrum(-1., 1. / u.Mpc, 1.)
    anisotropic_power_instance = PowerLawPowerSpectrum(-1., 1. / u.Mpc, 1.)
    anisotropic_power_instance.set_anisotropic_functional_form(lambda a, b:np.array([0., 0., 0., 0., 1.]))
    npt.assert_allclose(anisotropic_power_instance.evaluate_multipole(0,test_k),isotropic_power_instance.evaluate3d_isotropic(test_k))

def test_choose_location_voigt_profiles_in_sky():
    test_box_size = {'x': 25. * u.Mpc, 'y': 25. * u.Mpc, 'z': 25. * u.Mpc}
    test_gaussian_box = GaussianBox(test_box_size,{'x': 250, 'y': 250, 'z': 117},3.993,(70.4*u.km)/(u.s*u.Mpc),0.2726)
    test_gaussian_box._num_voigt = 10000
    test_gaussian_box._choose_location_voigt_profiles_in_sky()
    assert test_gaussian_box._voigt_profile_skewers_index_arr.shape[0] == 10000 #np.sum(test_gaussian_box._voigt_profile_skewers_bool_arr) == 10000

def test_form_voigt_profile_box(): #SOME REPETITION OF TEST ABOVE!!!
    test_box_size = {'x': 25. * u.Mpc, 'y': 25. * u.Mpc, 'z': 25. * u.Mpc}
    test_gaussian_box = GaussianBox(test_box_size, {'x': 250, 'y': 250, 'z': 117}, 3.993, (70.4 * u.km) / (u.s * u.Mpc),0.2726)
    test_gaussian_box._num_voigt = 10000
    test_gaussian_box._choose_location_voigt_profiles_in_sky()
    no_voigt_profile_bool_arr = np.logical_not(test_gaussian_box._voigt_profile_skewers_bool_arr)
    test_zeros = np.zeros((test_gaussian_box.num_clean_skewers,117))
    npt.assert_array_equal(test_gaussian_box._form_voigt_profile_box(1.*(u.km/u.s),1.*(u.km/u.s),1.,wrap_around=10)[0][no_voigt_profile_bool_arr],test_zeros)

def test_3D_flux_power_zeros():
    test_box = np.zeros((100,150,200))
    test_estimator = FourierEstimator3D(test_box)
    npt.assert_array_equal(test_estimator.get_flux_power_3D()[0],test_box)

def test_cross_power_spectrum():
    test_box_1 = npr.rand(100, 150, 200)
    test_box_2 = npr.rand(100, 150, 200)
    test_estimator_1 = FourierEstimator3D(test_box_1)
    test_estimator_2 = FourierEstimator3D(test_box_2)
    test_estimator_cross = FourierEstimator3D(test_box_1,second_box=test_box_2)
    test_estimator_total = FourierEstimator3D(test_box_1 + test_box_2)
    total_power = test_estimator_1.get_flux_power_3D(norm=False)[0] + test_estimator_2.get_flux_power_3D(norm=False)[0] + (2. * test_estimator_cross.get_flux_power_3D(norm=False)[0])
    npt.assert_allclose(total_power,test_estimator_total.get_flux_power_3D(norm=False)[0])

def test_form_return_list():
    test_size = 10000
    n_bins_x_y = (10, 20)
    test_x_y = npr.rand(2, test_size-1)
    test_f = np.zeros(test_size)
    test_estimator = FourierEstimator3D(test_f.reshape(10,10,100))
    expected_zeros = [np.zeros(n_bins_x_y),np.zeros(n_bins_x_y)]
    npt.assert_array_equal(test_estimator._form_return_list(test_x_y[0],test_x_y[1],n_bins_x_y[0],n_bins_x_y[1],False,False,False,True),expected_zeros)

def test_bin_f_x_y_histogram():
    test_size = 10000
    n_bins_x_y = (10,20)
    test_x_y = npr.rand(2,test_size)
    test_f = np.ones(test_size)
    npt.assert_array_equal(bin_f_x_y_histogram(test_x_y[0],test_x_y[1],test_f,n_bins_x_y[0],n_bins_x_y[1]),np.ones(n_bins_x_y))

def test_calculate_local_average_of_array():
    test_box = np.zeros((100, 150, 200))
    bin_size = 5
    npt.assert_array_equal(calculate_local_average_of_array(test_box,bin_size),test_box[...,:get_end_index(bin_size)])

def test_make_box_hermitian():
    test_box = npr.rand(10,11,12)
    hermitian_box = make_box_hermitian(test_box)
    real_box = np.fft.ifftn(hermitian_box,s=(10,11,12),axes=(0,1,2))
    npt.assert_allclose(real_box.imag,np.zeros_like(real_box.imag),atol=1.e-16)


#Pipeline tests - will set up tests to test all combinations of boxes and Fourier estimators

'''def test_3D_flux_power_multipole(): #NEEDS MODIFYING
    pow_index = 0.
    pow_pivot = 1. / u.Mpc
    pow_amp = 1.
    box_size = {'x': 25 * u.Mpc, 'y': 25 * u.Mpc, 'z': 25 * u.Mpc}
    n_samp = {'x': 201, 'y': 201, 'z': 201}
    redshift = 2.
    H0 = (67.31 * u.km) / (u.s * u.Mpc)
    omega_m = 0.3149

    multipole = 0
    n_bins = 1000

    simu_box, k_box, mu_box = isotropic_power_spectrum_to_boxes(pow_index,pow_pivot,pow_amp,box_size,n_samp,redshift, H0,omega_m)
    power_binned_ell,k_binned_ell,power_mu_sorted = boxes_to_power_3D_multipole(multipole,simu_box,k_box,mu_box,n_bins)
    power_instance = IsotropicPowerLawPowerSpectrum(pow_index, pow_pivot, pow_amp)
    true_power = power_instance.evaluate3d(k_binned_ell)
    npt.assert_array_equal(power_binned_ell[1:]*(201**6),true_power[1:])'''