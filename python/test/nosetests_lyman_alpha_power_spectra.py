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

def test_isotropic_power_law_power_spectra():
    test_k = np.arange(1.e-3, 1.e+2, 1.e-3) / u.Mpc
    isotropic_power_instance = PowerLawPowerSpectrum(0., 1. / u.Mpc, 1.)
    isotropic_power = isotropic_power_instance.evaluate3d_isotropic(test_k)
    npt.assert_array_equal(isotropic_power, np.ones_like(test_k.value))

def test_pre_computed_power_spectra_no_interpolation_limit():
    fname = os.path.dirname(os.path.abspath(__file__)) + '/P_k_z_4_default_CLASS.dat'
    pre_computed_power_instance = PreComputedPowerSpectrum(fname,n_interpolation_samples = 'default')
    power_interpolated = pre_computed_power_instance.evaluate3d_isotropic(pre_computed_power_instance.k_raw)
    npt.assert_allclose(power_interpolated,pre_computed_power_instance.power_raw)

def test_anisotropic_power_law_power_spectra_isotropic_limit():
    test_k = np.arange(1.e-3,1.e+2,1.e-3) / u.Mpc
    isotropic_power_instance = PowerLawPowerSpectrum(-1., 1. / u.Mpc, 1.)
    anisotropic_power_instance = PowerLawPowerSpectrum(-1., 1. / u.Mpc, 1.)
    anisotropic_power_instance.set_anisotropic_functional_form(lambda a, b:np.array([0., 0., 0., 0., 1.]))
    npt.assert_allclose(anisotropic_power_instance.evaluate_multipole(0,test_k),isotropic_power_instance.evaluate3d_isotropic(test_k))