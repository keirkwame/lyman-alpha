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

def test_gauss_realisation():
    test_box_size = {'x': 25. * u.Mpc, 'y': 25. * u.Mpc, 'z': 25. * u.Mpc}
    test_n_samples = {'x': 250, 'y': 250, 'z': 117}
    test_gaussian_box = GaussianBox(test_box_size, test_n_samples, 3.993, (70.4 * u.km) / (u.s * u.Mpc), 0.2726)
    k_samples = test_gaussian_box.k_box()
    test_gaussian_realisation = test_gaussian_box._gauss_realisation(np.zeros_like(k_samples.value),k_samples)
    npt.assert_array_equal(test_gaussian_realisation,np.zeros((test_n_samples['x'],test_n_samples['y'],test_n_samples['z'])))

def test_isotropic_pre_computed_gauss_realisation():
    fname = os.path.dirname(os.path.abspath(__file__)) + '/P_k_z_4_default_CLASS.dat'
    test_box_size = {'x': 25. * u.Mpc, 'y': 25. * u.Mpc, 'z': 25. * u.Mpc}
    test_n_samples = {'x': 250, 'y': 250, 'z': 117}
    test_gaussian_box = GaussianBox(test_box_size, test_n_samples, 4., (67.11 * u.km) / (u.s * u.Mpc), 0.3161)
    test_gaussian_realisation = test_gaussian_box.isotropic_pre_computed_gauss_realisation(fname,n_interpolation_samples=250)
    assert test_gaussian_realisation.shape == (test_n_samples['x'],test_n_samples['y'],test_n_samples['z'])

def test_anisotropic_pre_computed_gauss_realisation_mean():
    fname = os.path.dirname(os.path.abspath(__file__)) + '/P_k_z_4_default_CLASS.dat'
    test_box_size = {'x': 25. * u.Mpc, 'y': 25. * u.Mpc, 'z': 25. * u.Mpc}
    test_n_samples = {'x': 250, 'y': 250, 'z': 117}
    anisotropic_function = lambda a, b:np.array([1., 1., 1., 1., 1.])
    test_gaussian_box = GaussianBox(test_box_size, test_n_samples, 4., (67.11 * u.km) / (u.s * u.Mpc), 0.3161)
    test_anisotropic_gaussian_realisation = test_gaussian_box.anisotropic_pre_computed_gauss_realisation(fname,anisotropic_function,n_interpolation_samples=250)
    assert np.absolute(np.mean(test_anisotropic_gaussian_realisation)) < 1.e-16

def test_choose_location_voigt_profiles_in_sky():
    test_box_size = {'x': 25. * u.Mpc, 'y': 25. * u.Mpc, 'z': 25. * u.Mpc}
    test_gaussian_box = GaussianBox(test_box_size,{'x': 250, 'y': 250, 'z': 117},4.,(67.11*u.km)/(u.s*u.Mpc),0.3161)
    test_gaussian_box._num_voigt = 10000
    test_gaussian_box._choose_location_voigt_profiles_in_sky()
    assert test_gaussian_box._voigt_profile_skewers_index_arr.shape[0] == 10000

def test_form_voigt_profile_box():
    test_box_size = {'x': 25. * u.Mpc, 'y': 25. * u.Mpc, 'z': 25. * u.Mpc}
    test_gaussian_box = GaussianBox(test_box_size, {'x': 250, 'y': 250, 'z': 117}, 4., (67.11 * u.km) / (u.s * u.Mpc),0.3161)
    test_gaussian_box._num_voigt = 10000
    test_gaussian_box._choose_location_voigt_profiles_in_sky()
    no_voigt_profile_bool_arr = np.logical_not(test_gaussian_box._voigt_profile_skewers_bool_arr)
    test_zeros = np.zeros((test_gaussian_box.num_clean_skewers,117))
    npt.assert_array_equal(test_gaussian_box._form_voigt_profile_box(1.*(u.km/u.s),1.*(u.km/u.s),1.,wrap_around=10)[0][no_voigt_profile_bool_arr],test_zeros)

def test_add_voigt_profiles():
    test_box_size = {'x': 25. * u.Mpc, 'y': 25. * u.Mpc, 'z': 25. * u.Mpc}
    test_gaussian_box_instance = GaussianBox(test_box_size, {'x': 25, 'y': 25, 'z': 11}, 4., (67.11 * u.km) / (u.s * u.Mpc),0.3161)
    test_gaussian_realisation = test_gaussian_box_instance.isotropic_power_law_gauss_realisation(0.,1./u.Mpc,1.)
    test_voigt_box = test_gaussian_box_instance.add_voigt_profiles(test_gaussian_realisation, 1, 1.*(u.km/u.s),1.*(u.km/u.s),0.)[0]
    npt.assert_array_equal(test_voigt_box, test_gaussian_realisation)

def test_3D_power_zeros():
    test_box = np.zeros((100,150,200))
    test_estimator = FourierEstimator3D(test_box)
    npt.assert_array_equal(test_estimator.get_power_3D()[0], test_box)

def test_cross_power_spectrum():
    test_box_1 = npr.rand(100, 150, 200)
    test_box_2 = npr.rand(100, 150, 200)
    test_estimator_1 = FourierEstimator3D(test_box_1)
    test_estimator_2 = FourierEstimator3D(test_box_2)
    test_estimator_cross = FourierEstimator3D(test_box_1, second_box = test_box_2)
    test_estimator_total = FourierEstimator3D(test_box_1 + test_box_2)
    total_power = test_estimator_1.get_power_3D(norm=False)[0] + test_estimator_2.get_power_3D(norm=False)[0] + (2. * test_estimator_cross.get_power_3D(norm=False)[0])
    npt.assert_allclose(total_power, test_estimator_total.get_power_3D(norm = False)[0])

def test_form_return_list():
    test_size = 10000
    n_bins_x_y = (10, 20)
    test_x_y = npr.rand(2, test_size-1)
    test_f = np.zeros(test_size) #Real-space test box
    test_estimator = FourierEstimator3D(test_f.reshape(10, 10, 100))
    expected_arrays = np.zeros((2, n_bins_x_y[0], n_bins_x_y[1]))
    npt.assert_array_equal(np.array(test_estimator._form_return_list(test_x_y[0],test_x_y[1],n_bins_x_y[0],n_bins_x_y[1],False,False,False,False,True)),expected_arrays)

def test_bin_f_x_y_histogram():
    test_size = 10000
    n_bins_x_y = (10, 20)
    test_x_y = npr.rand(2, test_size)
    test_f = np.ones(test_size)
    npt.assert_array_equal(bin_f_x_y_histogram(test_x_y[0],test_x_y[1],test_f,n_bins_x_y[0],n_bins_x_y[1]),np.ones(n_bins_x_y))

def test_standard_error():
    test_size = int(1.e+6)
    assert standard_error(np.ones(test_size)) == 0.

def test_bin_f_x_y_histogram_standard_error():
    test_size = 10000
    n_bins_x_y = (10, 20)
    test_x_y = npr.rand(2, test_size)
    test_f = np.ones(test_size)
    npt.assert_array_equal(bin_f_x_y_histogram_standard_error(test_x_y[0], test_x_y[1], test_f, n_bins_x_y[0], n_bins_x_y[1]),np.zeros(n_bins_x_y))

def test_calculate_local_average_of_array():
    test_box = np.zeros((100, 150, 200))
    bin_size = 5
    npt.assert_array_equal(calculate_local_average_of_array(test_box, bin_size) ,test_box[...,:get_end_index(bin_size)])

def test_make_box_hermitian():
    test_box = npr.rand(10, 11, 12)
    hermitian_box = make_box_hermitian(test_box)
    real_box = np.fft.ifftn(hermitian_box,s=(10, 11, 12),axes=(0, 1, 2))
    npt.assert_allclose(real_box.imag, np.zeros_like(real_box.imag), atol=1.e-16)

def test_gen_log_space():
    array_length = 100000
    npt.assert_array_equal(gen_log_space(array_length, array_length), np.arange(array_length))