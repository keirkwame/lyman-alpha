import sys
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import astropy.units as u

import boxes as box

def plot_forest_spectrum(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION, SPECTRA_SAVEDIR, PLOTNAME, RELOAD_SNAPSHOT=False, SPECTRA_SAVEFILE_ROOT='gridded_spectra', SPECTRUM_NUM=0, FLUX_ASCII_FILENAME=None):
    simulation_box_instance = box.SimulationBox(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION,
                                                reload_snapshot=RELOAD_SNAPSHOT,
                                                spectra_savefile_root=SPECTRA_SAVEFILE_ROOT,
                                                spectra_savedir=SPECTRA_SAVEDIR)
    optical_depth = simulation_box_instance.get_optical_depth()
    transmitted_flux = np.exp(-1. * optical_depth)
    velocity_samples = simulation_box_instance.r_i('z')

    figure, axis = plt.subplots()
    axis.plot(velocity_samples.to(u.km / u.s), transmitted_flux[SPECTRUM_NUM])

    if flux_ascii_filename is not None:
        velocity_flux = np.loadtxt(flux_ascii_filename, skiprows=2)
        axis.plot(velocity_flux[:,0], velocity_flux[:,1], label=r'From ASCII file')
        axis.legend(frameon=False)

    axis.set_ylim([-0.025, 1.025])
    axis.set_xlabel(r'km / s')
    axis.set_ylabel(r'Transmitted flux')
    plt.savefig(SPECTRA_SAVEDIR + '/' + PLOTNAME)

    return simulation_box_instance

if __name__ == "__main__":
    snapshot_directory = sys.argv[1] #'/home/jsbolton/Sherwood/planck1_80_1024'
    spectra_directory = sys.argv[2] #'/home/keir/Data/Sherwood/planck1_80_1024/snapdir_011'

    plotname = 'spectrum_40_40_hi_res.pdf'
    flux_ascii_filename = None #'/Users/kwame/Simulations/Sherwood/planck1_80_1024/snapdir_011/spectest.txt'

    output = plot_forest_spectrum(11, snapshot_directory, 2, 4. * u.km / u.s, spectra_directory, plotname, RELOAD_SNAPSHOT=False, SPECTRUM_NUM=3, FLUX_ASCII_FILENAME=flux_ascii_filename)
