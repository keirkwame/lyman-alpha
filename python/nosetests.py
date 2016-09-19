import numpy.testing as npt

from fourier_estimators import *

#This is just the very start of the test suite!

def test_3D_flux_power_zeros():
    test_box = np.zeros((100,150,200))
    test_estimator = FourierEstimator3D(test_box)
    npt.assert_array_equal(test_estimator.get_flux_power_3D()[0],test_box)