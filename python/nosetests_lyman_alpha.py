import numpy.testing as npt
import astropy.units as u

from main import *
from power_spectra import *
from boxes import *
from fourier_estimators import *

#This is just the very start of the test suite!

#Null tests

def test_anisotropic_power_spectra_isotropic_limit():
    test_k = np.arange(1.e-3,1.e+2,1.e-3) / u.Mpc
    isotropic_power_instance = IsotropicPowerLawPowerSpectrum(-1.,1./u.Mpc,1.)
    anisotropic_power_instance = AnisotropicPowerLawPowerSpectrum(-1.,1./u.Mpc,1.,lambda a,b:np.array([0.,0.,0.,0.,1.]))
    npt.assert_allclose(anisotropic_power_instance.evaluate_multipole(0,test_k),isotropic_power_instance.evaluate3d(test_k))
    #return anisotropic_power_instance.evaluate_multipole(0,test_k),isotropic_power_instance.evaluate3d(test_k)

def test_3D_flux_power_zeros():
    test_box = np.zeros((100,150,200))
    test_estimator = FourierEstimator3D(test_box)
    npt.assert_array_equal(test_estimator.get_flux_power_3D()[0],test_box)


#Pipeline tests

def test_3D_flux_power_multipole(): #NEEDS MODIFYING
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
    npt.assert_array_equal(power_binned_ell[1:]*(201**6),true_power[1:])