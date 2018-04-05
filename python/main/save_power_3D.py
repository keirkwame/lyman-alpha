import numpy as np
import astropy.units as u
import math as mh
import sys

import power_spectra as spe
import boxes as box
import fourier_estimators as fou
import utils as uti

if __name__ == "__main__":
    """Input arguments: Snapshot number; Snapshot directory path; Width of skewer grid in samples;
    Resolution of spectra in km s^{-1}; Spectra directory path (with '/snapdir_XXX' if necessary)"""

    SNAPSHOT_NUM = int(sys.argv[1])
    SNAPSHOT_DIR = sys.argv[2]
    GRID_WIDTH_IN_SAMPS = int(sys.argv[3])
    SPECTRUM_RESOLUTION = int(sys.argv[4]) * u.km / u.s
    RELOAD_SNAPSHOT = False
    SPECTRA_SAVEFILE_ROOT = 'gridded_spectra'
    SPECTRA_SAVEDIR = sys.argv[5]
    POWER_SPECTRA_SAVEFILE = '/power_spectra.npz'

    simulation_box_instance = box.SimulationBox(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION, reload_snapshot=RELOAD_SNAPSHOT, spectra_savefile_root=SPECTRA_SAVEFILE_ROOT, spectra_savedir=SPECTRA_SAVEDIR)

    simulation_box_instance.convert_fourier_units_to_distance = True
    delta_flux_box = simulation_box_instance.skewers_realisation()
    k_box = simulation_box_instance.k_box()
    mu_box = simulation_box_instance.mu_box()

    #Binning to match GenPK
    n_k_bins = 15
    n_mu_bins = 4
    k_max = np.max(k_box) #0.704 / u.Mpc

    k_min = np.min(k_box[k_box > 0. / u.Mpc])
    k_bin_max = mh.exp(mh.log(k_max.value) + ((mh.log(k_max.value) - mh.log(k_min.value)) / (n_k_bins - 1))) / u.Mpc
    k_bin_edges = np.exp(np.linspace(mh.log(k_min.value), mh.log(k_bin_max.value), n_k_bins + 1)) / u.Mpc

    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)

    fourier_estimator_instance = fou.FourierEstimator3D(delta_flux_box)
    power_binned, k_binned, mu_binned, bin_counts = fourier_estimator_instance.get_power_3D_two_coords_binned(k_box,np.absolute(mu_box),k_bin_edges,mu_bin_edges,count=True)
    np.savez(SPECTRA_SAVEDIR + POWER_SPECTRA_SAVEFILE, power_binned, k_binned, mu_binned, bin_counts)