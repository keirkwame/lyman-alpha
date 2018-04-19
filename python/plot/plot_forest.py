import sys
import math as mh
import numpy as np
import numpy.testing as npt
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
    #plt.savefig(plotname)

    return simulation_box_instance

def plot_CDDF(plotname, cddf_savename, simulation_box_instance, load_cddf=False):
    figure, axis = plt.subplots()

    if load_cddf is False:
        column_density = simulation_box_instance.get_column_density()
        column_density_log10_cm2_no0 = np.log10(column_density[column_density > 0. / (u.cm ** 2)].value)

        histogram_bin_edges = np.arange(mh.floor(np.min(column_density_log10_cm2_no0)), mh.ceil(np.max(column_density_log10_cm2_no0))+0.1, 0.1)
        cddf = np.histogram(column_density_log10_cm2_no0, bins=histogram_bin_edges)
        np.savez(cddf_savename, cddf, histogram_bin_edges)

        #axis.hist(column_density.flatten(), bins='auto', normed=True, histtype='step')
        #axis.set_xscale('log')
        #axis.set_yscale('log')

        return column_density
    else:
        cddf_file = np.load(cddf_savename)
        cddf = cddf_file['arr_0']
        histogram_bin_edges = cddf_file['arr_1']
        npt.assert_array_equal(cddf[1], histogram_bin_edges)
        histogram_bin_centres = (histogram_bin_edges[:-1] + histogram_bin_edges[1:]) / 2

        #return cddf_file

        axis.scatter(histogram_bin_centres, np.log10(cddf[0]))
        axis.axvline(x=mh.log10(1.6e17), color='black', ls='--')
        axis.axvline(x=mh.log10(2.e20), color='black', ls='--')
        axis.set_xlabel(r'log[$N$(HI) ($\mathrm{cm}^{-2}$)]')
        axis.set_ylabel(r'CDDF (log[number of spectral pixels])')
        plt.savefig(plotname)

def plot_power_spectra(plotname, power_spectra_savename, simulation_box_instance=None, box_length=None, hubble_constant=None):
    if simulation_box_instance is not None:
        hubble_constant = simulation_box_instance.spectra_instance.hubble
        box_length = simulation_box_instance.spectra_instance.box * u.kpc
    power_spectra_file = np.load(power_spectra_savename)
    power_spectra = (power_spectra_file['arr_0'] * (box_length ** 3)).to(u.Mpc ** 3)
    k = power_spectra_file['arr_1'] / u.Mpc
    mu = power_spectra_file['arr_2']
    n_samples_per_bin = power_spectra_file['arr_3']
    error_bars = 2. * power_spectra / np.sqrt(n_samples_per_bin)

    figure, axis = plt.subplots()
    line_labels = [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$']
    for i in range(k.shape[1]): #Loop over mu bins
        print('Plotting mu bin number', i+1)
        x_plot = k[:,i] / hubble_constant
        print(x_plot)
        y_plot = power_spectra[:,i] * (hubble_constant ** 3)
        print(y_plot)
        axis.plot(x_plot, y_plot, label=line_labels[i])
        print(error_bars[:,i])
        #axis.errorbar(x_plot, y_plot.value, yerr=error_bars[:,i].value, ls='')
    axis.legend(frameon=False)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel(r'$k$ ($h\,\mathrm{Mpc}^{-1}$)')
    axis.set_ylabel(r'$P(k)$ ($\mathrm{Mpc}^3\,h^{-3}$)')
    plt.savefig(plotname)

if __name__ == "__main__":
    snapshot_directory = sys.argv[1] #'/home/jsbolton/Sherwood/planck1_80_1024'
    spectra_directory = sys.argv[2] #'/home/keir/Data/Sherwood/planck1_80_1024/snapdir_011'

    plotname = spectra_directory + '/power_spectra_font.pdf' #'/spectrum_750_25.pdf'
    power_spectra_savename = spectra_directory + '/power_spectra_font.npz'
    cddf_savename = spectra_directory + '/CDDF.npz'
    flux_ascii_filename = '/Users/kwame/Simulations/Sherwood/planck1_80_1024/snapdir_011/spectest.txt'

    sim_box_ins = get_simulation_box_instance(11, snapshot_directory, 2, 3. * u.km / u.s, spectra_directory, RELOAD_SNAPSHOT=False) #, SPECTROGRAPH_FWHM=1.*u.km/u.s)
    output = plot_forest_spectrum(plotname, sim_box_ins, spectrum_num=3, flux_ascii_filename=flux_ascii_filename)
    #output = plot_CDDF(plotname, cddf_savename, sim_box_ins, load_cddf=True)
    hubble_constant = 0.6724
    #plot_power_spectra(plotname, power_spectra_savename, box_length=30. * u.Mpc / hubble_constant, hubble_constant=hubble_constant) #sim_box_ins)
