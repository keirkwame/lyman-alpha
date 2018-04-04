import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import boxes as box

def plot_forest_spectrum(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION, SPECTRA_SAVEDIR, PLOTNAME, RELOAD_SNAPSHOT=False, SPECTRA_SAVEFILE_ROOT='gridded_spectra'):
    simulation_box_instance = box.SimulationBox(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION,
                                                reload_snapshot=RELOAD_SNAPSHOT,
                                                spectra_savefile_root=SPECTRA_SAVEFILE_ROOT,
                                                spectra_savedir=SPECTRA_SAVEDIR)
    optical_depth = simulation_box_instance.get_optical_depth()
    transmitted_flux = np.exp(-1. * optical_depth)
    velocity_samples = simulation_box_instance.r_i('z')

    figure, axis = plt.subplots()
    axis.plot(velocity_samples.to(u.km / u.s), transmitted_flux[0])
    axis.set_xlabel(r'km / s')
    axis.set_ylabel(r'Transmitted flux')
    plt.savefig(SPECTRA_SAVEDIR + '/' + PLOTNAME)

if __name__ == "__main__":
    snapshot_directory = '/home/jsbolton/Sherwood/planck1_80_1024'
    spectra_directory = '/home/keir/Sherwood/planck1_80_1024/snapdir_011'

    plot_forest_spectrum(11, snapshot_directory, 2, 25. * u.km / u.s, spectra_directory, 'spectrum.pdf', RELOAD_SNAPSHOT=True)
