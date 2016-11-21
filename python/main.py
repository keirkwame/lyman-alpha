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

def snapshot_to_boxes(snap_num,snap_dir,grid_samps,spectrum_resolution,reload_snapshot=True):
    box_instance = SimulationBox(snap_num,snap_dir,grid_samps,spectrum_resolution,reload_snapshot=reload_snapshot)
    box_instance.convert_fourier_units_to_distance = True
    print(box_instance._n_samp)
    return box_instance.skewers_realisation(), box_instance.k_box(), box_instance.mu_box()

def power_spectrum_to_boxes(pow_index, pow_pivot, pow_amp, box_size, n_samp, redshift, H0, omega_m):
    box_instance = GaussianBox(box_size, n_samp, redshift, H0, omega_m)
    box_instance.convert_fourier_units_to_distance = True
    print(box_instance.k_i('z'))
    return box_instance.isotropic_power_law_gauss_realisation(pow_index, pow_pivot, pow_amp), box_instance.k_box(), box_instance.mu_box()

def boxes_to_power_3D_binned(simu_box,k_box,n_bins,norm=True):
    power_instance = FourierEstimator3D(simu_box)
    return power_instance.get_flux_power_3D_binned(k_box,n_bins,norm=norm)

def boxes_to_power_3D_multipole(multipole,simu_box,k_box,mu_box,n_bins,norm=True):
    power_instance = FourierEstimator3D(simu_box)
    return power_instance.get_flux_power_3D_multipole(multipole, k_box, mu_box, n_bins, norm=norm)

if __name__ == "__main__":
    """Input arguments: Snapshot directory path; Snapshot number; grid_samps; Resolution of spectrum in km s^{-1}"""
    snap_dir = sys.argv[1]
    #snap_dir = '/Users/keir/Documents/lyman_alpha/simulations/illustris_Cosmo7_V6'
    #snap_dir = '/home/keir/Data/illustris_Cosmo7_V6'
    snap_num = int(sys.argv[2])
    grid_samps = int(sys.argv[3])
    spectrum_resolution = float(sys.argv[4])*(u.km / u.s)
    n_bins = 1000
    reload_snapshot = False
    norm = True

    pow_index = -1.
    pow_pivot = 1. / u.Mpc
    pow_amp = 1.
    box_size = {'x': 25 * u.Mpc, 'y': 25 * u.Mpc, 'z': 25 * u.Mpc}
    n_samp = {'x': 201, 'y': 201, 'z': 201}
    redshift = 2.
    H0 = (67.31 * u.km) / (u.s * u.Mpc)
    omega_m = 0.3149

    multipole = 4

    #simu_box, k_box, mu_box = snapshot_to_boxes(snap_num,snap_dir,grid_samps,spectrum_resolution,reload_snapshot=reload_snapshot)
    simu_box, k_box, mu_box = power_spectrum_to_boxes(pow_index,pow_pivot,pow_amp,box_size,n_samp,redshift,H0,omega_m)
    power_binned, k_binned, power_k_sorted = boxes_to_power_3D_binned(simu_box,k_box,n_bins,norm=norm)
    power_binned_ell, k_binned_ell, power_mu_sorted = boxes_to_power_3D_multipole(multipole,simu_box,k_box,mu_box,n_bins,norm=norm)

    power_instance = PowerLawPowerSpectrum(pow_index, pow_pivot, pow_amp)
    true_power = power_instance.evaluate3d(k_binned_ell)