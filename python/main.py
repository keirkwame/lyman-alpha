import math as mh
import random as rd
import numpy as np
import numpy.random as npr
import scipy.integrate as spi
import copy as cp
import astropy.units as u
import spectra as sa
import griddedspectra as gs
import randspectra as rs
import sys

from utils import *
from power_spectra import *
from boxes import *
from fourier_estimators import *

def snapshot_to_power_3D_binned(snap_num,snap_dir,grid_samps,spectrum_resolution,n_bins,reload_snapshot=True,norm=True):
    box_instance = SimulationBox(snap_num,snap_dir,grid_samps,spectrum_resolution,reload_snapshot=reload_snapshot)
    box_instance.convert_fourier_units_to_distance = True
    print(box_instance._n_samp)
    simu_box = box_instance.skewers_realisation()
    power_instance = FourierEstimator3D(simu_box)
    k_box = box_instance.k_box()
    return power_instance.get_flux_power_3D_binned(k_box,n_bins,norm=norm)

if __name__ == "__main__":
    """Input arguments: Snapshot directory path; Snapshot number; grid_samps; Resolution of spectrum in km s^{-1}"""
    snap_dir = sys.argv[1]
    #snap_dir = '/Users/keir/Documents/lyman_alpha/simulations/illustris_Cosmo7_V6'
    #snap_dir = '/home/keir/Data/illustris_Cosmo7_V6'
    snap_num = int(sys.argv[2])
    grid_samps = int(sys.argv[3])
    spectrum_resolution = float(sys.argv[4])*(u.km / u.s)
    n_bins = 10000
    reload_snapshot = False
    norm = True
    power_binned, k_binned = snapshot_to_power_3D_binned(snap_num,snap_dir,grid_samps,spectrum_resolution,n_bins,reload_snapshot=reload_snapshot,norm=norm)
