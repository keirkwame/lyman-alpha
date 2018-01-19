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

#Global variables
SNAPSHOT_NUM = 64
SNAPSHOT_DIR = '/Users/kwame/Simulations/Illustris_1'
GRID_WIDTH_IN_SAMPS = 25
SPECTRUM_RESOLUTION = 25 * u.km / u.s
RELOAD_SNAPSHOT = False
SPECTRA_SAVEFILE_ROOT = 'gridded_spectra'
SPECTRA_SAVEDIR = None

test_simulation_box_instance = SimulationBox(SNAPSHOT_NUM,SNAPSHOT_DIR,GRID_WIDTH_IN_SAMPS,SPECTRUM_RESOLUTION,reload_snapshot=RELOAD_SNAPSHOT,spectra_savefile_root=SPECTRA_SAVEFILE_ROOT,spectra_savedir=SPECTRA_SAVEDIR)

def test_generate_general_spectra_instance():
    test_cofm = np.array([[10.,10.,10.],[20., 20., 20.]])
    general_spectrum_instance = test_simulation_box_instance._generate_general_spectra_instance(test_cofm)
    assert general_spectrum_instance.NumLos == 2

def test_get_optical_depth():
    optical_depth = test_simulation_box_instance.get_optical_depth()
    assert optical_depth.shape == (GRID_WIDTH_IN_SAMPS ** 2, test_simulation_box_instance._n_samp['z'])

def test_get_column_density():
    column_density = test_simulation_box_instance.get_column_density()
    assert np.min(column_density) >= 0. / (u.cm ** 2)

def test_get_delta_flux():
    optical_depth = test_simulation_box_instance.get_optical_depth()
    delta_flux = test_simulation_box_instance._get_delta_flux(optical_depth, None, None, None)
    assert np.absolute(np.mean(delta_flux)) < 1.e-16