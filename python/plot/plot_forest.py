import sys
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import astropy.units as u

import boxes as box

def get_simulation_box_instance(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION, SPECTRA_SAVEDIR, RELOAD_SNAPSHOT=False, SPECTRA_SAVEFILE_ROOT='gridded_spectra', SPECTROGRAPH_FWHM='default'):
    simulation_box_instance = box.SimulationBox(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION,
                                                reload_snapshot=RELOAD_SNAPSHOT,
                                                spectra_savefile_root=SPECTRA_SAVEFILE_ROOT,
                                                spectra_savedir=SPECTRA_SAVEDIR, spectrograph_FWHM=SPECTROGRAPH_FWHM)
    return simulation_box_instance

def plot_forest_spectrum(plotname, simulation_box_instance, spectrum_num=0, flux_ascii_filename=None):
    optical_depth = simulation_box_instance.get_optical_depth()
    transmitted_flux = np.exp(-1. * optical_depth)
    velocity_samples = simulation_box_instance.r_i('z')

    figure, axis = plt.subplots()
    axis.plot(velocity_samples.to(u.km / u.s), transmitted_flux[spectrum_num])

    if flux_ascii_filename is not None:
        velocity_flux = np.loadtxt(flux_ascii_filename, skiprows=2)
        simulation_box_instance._velocity_flux_ascii = velocity_flux
        axis.plot(velocity_flux[:,0], velocity_flux[:,1], label=r'From ASCII file')
        axis.legend(frameon=False)

    axis.set_ylim([-0.025, 1.025])
    axis.set_xlabel(r'km / s')
    axis.set_ylabel(r'Transmitted flux')
    plt.savefig(plotname)

    return simulation_box_instance

def plot_CDDF(plotname, simulation_box_instance):
    column_density = simulation_box_instance.get_column_density() / (u.cm ** 2)

if __name__ == "__main__":
    snapshot_directory = sys.argv[1] #'/home/jsbolton/Sherwood/planck1_80_1024'
    spectra_directory = sys.argv[2] #'/home/keir/Data/Sherwood/planck1_80_1024/snapdir_011'

    plotname = spectra_directory + '/spectrum_750_25.pdf'
    flux_ascii_filename = None #'/Users/kwame/Simulations/Sherwood/planck1_80_1024/snapdir_011/spectest.txt'

    sim_box_ins = get_simulation_box_instance(11, snapshot_directory, 750, 25. * u.km / u.s, spectra_directory, RELOAD_SNAPSHOT=True) #, SPECTROGRAPH_FWHM=20.*u.km/u.s)
    output = plot_forest_spectrum(plotname, sim_box_ins, spectrum_num=0, flux_ascii_filename=flux_ascii_filename)
    plot_CDDF('', sim_box_ins)
