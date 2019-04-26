import sys
import math as mh
import numpy as np
import numpy.testing as npt
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import astropy.units as u
import corner as co

import boxes as box
import fourier_estimators as fou
import save_power_3D as sav
import parametric_fit as pfit

import distinct_colours_py3 as dc

def get_simulation_box_instance(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION, SPECTRA_SAVEDIR, RELOAD_SNAPSHOT=False, SPECTRA_SAVEFILE_ROOT='gridded_spectra', SPECTROGRAPH_FWHM='default'):
    simulation_box_instance = box.SimulationBox(SNAPSHOT_NUM, SNAPSHOT_DIR, GRID_WIDTH_IN_SAMPS, SPECTRUM_RESOLUTION,
                                                reload_snapshot=RELOAD_SNAPSHOT,
                                                spectra_savefile_root=SPECTRA_SAVEFILE_ROOT,
                                                spectra_savedir=SPECTRA_SAVEDIR, spectrograph_FWHM=SPECTROGRAPH_FWHM)
    return simulation_box_instance

def plot_forest_spectrum(plotname, simulation_box_instance, spectrum_num=0, flux_ascii_filename=None, rescale_ascii=None, redshift_space=True):
    if redshift_space:
        optical_depth = simulation_box_instance.get_optical_depth()
    else:
        optical_depth = simulation_box_instance.get_optical_depth_real()
    transmitted_flux = np.exp(-1. * optical_depth)
    velocity_samples = simulation_box_instance.r_i('z')

    figure, axis = plt.subplots()
    axis.plot(velocity_samples.to(u.km / u.s), transmitted_flux[spectrum_num])

    if flux_ascii_filename is not None:
        velocity_flux = np.loadtxt(flux_ascii_filename, skiprows=2)
        simulation_box_instance._velocity_flux_ascii = velocity_flux
        if rescale_ascii is True:
            optical_depth_ascii = np.log(velocity_flux[:,1]) * -1.
            mean_rescaling_factor = np.mean(optical_depth_ascii) / np.mean(optical_depth[spectrum_num])
            print('mean[ASCII optical depth] / mean[optical depth] =', mean_rescaling_factor)
            transmitted_flux_ascii = np.exp(-1. * optical_depth_ascii / mean_rescaling_factor)
        else:
            transmitted_flux_ascii = velocity_flux[:,1]
        axis.plot(velocity_flux[:,0], transmitted_flux_ascii, label=r'From ASCII file')
        axis.legend(frameon=False)

    axis.set_ylim([-0.025, 1.025])
    axis.set_xlabel(r'km / s')
    axis.set_ylabel(r'Transmitted flux')
    plt.savefig(plotname)

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

def load_power_spectra(power_spectra_savename, box_length, power_spectra_linear_savename=None, hubble_constant=0.7):
    power_spectra_file = np.load(power_spectra_savename)
    power_spectra = (power_spectra_file['arr_0'] * ((box_length / hubble_constant) ** 3)).to(u.Mpc ** 3)
    k = power_spectra_file['arr_1'] / u.Mpc
    mu = power_spectra_file['arr_2']
    n_samples_per_bin = power_spectra_file['arr_3']
    error_bars = 2. * power_spectra / np.sqrt(n_samples_per_bin)

    if power_spectra_linear_savename is None:
        power_spectra_linear = None
    else:
        power_spectra_linear = np.load(power_spectra_savename2) / (hubble_constant ** 3) * (u.Mpc ** 3)

    return power_spectra, k, mu, n_samples_per_bin, error_bars, power_spectra_linear

def plot_power_spectra(plotname, power_spectra_savename, simulation_box_instance=None, box_length=None, hubble_constant=None, power_spectra_savename2=None, plot_errors=False):
    if simulation_box_instance is not None:
        hubble_constant = simulation_box_instance.spectra_instance.hubble
        box_length = simulation_box_instance.spectra_instance.box * u.kpc
    if power_spectra_savename2 is None:
        figure, axis = plt.subplots()
    else:
        figure, axes = plt.subplots(nrows=2, ncols=1)

    power_spectra, k, mu, n_samples_per_bin, error_bars, power_spectra2 = load_power_spectra(power_spectra_savename,
                    box_length, power_spectra_linear_savename=power_spectra_savename2, hubble_constant=hubble_constant)

    line_labels = [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$']
    distinct_colours = dc.get_distinct(k.shape[1])
    for i in range(k.shape[1]): #Loop over mu bins
        print('Plotting mu bin number', i+1)
        x_plot = k[:,i] / hubble_constant
        y_plot = power_spectra[:,i] * (hubble_constant ** 3)
        yerr_plot = error_bars[:,i] * (hubble_constant ** 3)
        if power_spectra_savename2 is not None:
            y2_plot = power_spectra2[:, i] * (hubble_constant ** 3)
            axes[1].plot(x_plot, y_plot / y2_plot, color=distinct_colours[i])
            axes[1].set_xscale('log')
            axes[1].set_yscale('log')
            axes[1].set_xlabel(r'$k$ ($h\,\mathrm{Mpc}^{-1}$)')
            axes[1].set_ylabel(r'$P(k) / P_\mathrm{linear} (k)$')
            '''if plot_errors is True:
                axes[1].errorbar(x_plot.value, (y_plot / y2_plot).value, yerr=(yerr_plot / y2_plot).value , ls='',
                      ecolor=distinct_colours[i])'''
            axis = axes[0]
        axis.plot(x_plot, y_plot, label=line_labels[i], color=distinct_colours[i])
        if plot_errors is True:
            axis.errorbar(x_plot.value, y_plot.value, yerr=yerr_plot.value, ls='', ecolor=distinct_colours[i])

    axis.legend(frameon=False)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel(r'$k$ ($h\,\mathrm{Mpc}^{-1}$)')
    axis.set_ylabel(r'$P(k)$ ($\mathrm{Mpc}^3\,h^{-3}$)')
    plt.savefig(plotname)

    return k

def bin_matter_power_spectrum(n_k_bins, n_mu_bins, simulation_box_instance, cosmo_name='base_plikHM_TTTEEE_lowTEB_2015', savename=None):
    simulation_box_instance.convert_fourier_units_to_distance = True
    redshift = simulation_box_instance._redshift
    k = simulation_box_instance.k_box()
    mu = np.absolute(simulation_box_instance.mu_box())
    hubble_constant = simulation_box_instance.spectra_instance.hubble

    k_bin_edges = sav.get_k_bin_edges_logspace(n_k_bins, k)
    mu_bin_edges = sav.get_mu_bin_edges_linspace(n_mu_bins)

    power = fou.get_matter_power_spectrum_two_coords_binned(redshift, k, k, mu, k_bin_edges, mu_bin_edges, hubble_constant, cosmology_name=cosmo_name)
    if savename is not None:
        np.save(savename, power)
    return power

def forest_linear_bias_model(k_mu_tuple, b_forest_weighted, beta_forest):
    (k, mu) = k_mu_tuple
    b_forest = b_forest_weighted / (1. + beta_forest)
    return ((b_forest * (1. + beta_forest * (mu ** 2))) ** 2) * pfit.forest_non_linear_function(k.value, mu)

def ln_likelihood_gaussian_forest_linear_bias(parameter_array, x, y, yerr, power_linear=None):
    model_evaluation = forest_linear_bias_model(x, parameter_array[0], parameter_array[1])
    return -0.5 * np.sum(((y - model_evaluation) ** 2) / ((yerr * model_evaluation) ** 2))

def ln_prior_forest_linear_bias(parameter_array):
    prior_limits = get_prior_limits_forest_linear_bias()
    if prior_limits[0, 0] < parameter_array[0] < prior_limits[0, 1] and prior_limits[1, 0] < parameter_array[1] < prior_limits[1, 1]: #b_F (1 + beta_F); beta_F
        return 0.
    else:
        return -np.inf

def get_prior_limits_forest_linear_bias():
    return np.array([[-100., 0.], [0., 100.]])

def sample_posterior_distribution(power_spectra_savename, power_spectra_linear_savename, lnlike_function,
        lnprior_function, prior_limits, n_walkers=100, n_steps=200, n_burn_in_steps=50, k_h_max=np.inf / u.Mpc,
        simulation_box_instance=None, box_length=None, hubble_constant=None, chains_savename=None):
    n_params = prior_limits.shape[0]
    if simulation_box_instance is not None:
        hubble_constant = simulation_box_instance.spectra_instance.hubble
        box_length = simulation_box_instance.spectra_instance.box * u.kpc
    power_spectra, k, mu, n_samples_per_bin, error_bars, power_spectra_linear = load_power_spectra(power_spectra_savename,
            box_length, power_spectra_linear_savename=power_spectra_linear_savename, hubble_constant=hubble_constant)

    k_h = k / hubble_constant
    k_h_cut_bool_array = k_h < k_h_max #Also removes nan's
    x = (k_h[k_h_cut_bool_array], mu[k_h_cut_bool_array])
    y = (power_spectra / power_spectra_linear)[k_h_cut_bool_array]
    yerr = (error_bars / power_spectra)[k_h_cut_bool_array]

    walker_starting_positions = pfit.get_starting_positions_in_uniform_prior(prior_limits, n_walkers)
    samples_flattened, chains_without_burn_in, sampler_instance = pfit.get_posterior_samples(lnlike_function,
        lnprior_function, x, y, yerr, n_params, n_walkers, n_steps, n_burn_in_steps, walker_starting_positions, lnprob=pfit.lnprob)

    gelman_rubin_statistic = pfit.gelman_rubin_convergence_statistic(chains_without_burn_in)
    print('Gelman-Rubin statistic =', gelman_rubin_statistic)
    b_F_weighted, beta_F = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples_flattened, [16, 50, 84], axis = 0)))
    maximum_posterior_parameter_array = np.array([b_F_weighted[0], beta_F[0]]) #np.array(posterior_summary_statistics[i][0] for i in len(posterior_summary_statistics))
    print('Posterior summary statistics =', b_F_weighted, beta_F) #posterior_summary_statistics)
    n_degrees_of_freedom = np.sum(k_h_cut_bool_array) - n_params
    reduced_chi_squared_statistic = -2. * lnlike_function(maximum_posterior_parameter_array, x, y, yerr) / n_degrees_of_freedom
    print('Reduced chi-squared statistic =', reduced_chi_squared_statistic)

    if chains_savename is not None:
        np.save(chains_savename, samples_flattened)
    return samples_flattened

def make_corner_plot(plotname, samples, parameter_names):
    n_params = samples.shape[1]
    figure, axes = plt.subplots(nrows=n_params, ncols=n_params)
    figure = co.corner(samples, labels=parameter_names, fig=figure)
    plt.savefig(plotname)

def plot_mean_flux_evolution(plotname, mean_flux_redshift):
    figure, axis = plt.subplots()
    axis.plot(mean_flux_redshift[:, 1], mean_flux_redshift[:, 0])
    axis.set_xlabel(r'Redshift')
    axis.set_ylabel(r'Mean transmitted flux')
    plt.savefig(plotname)


if __name__ == "__main__":
    snapshot_directory = sys.argv[1] #'/home/jsbolton/Sherwood/planck1_80_1024'
    spectra_directory = sys.argv[2] #'/home/keir/Data/Sherwood/planck1_80_1024/snapdir_011'

    plotname = spectra_directory + '/spectrum_750_10_real.pdf'
    power_spectra_savename = spectra_directory + '/power_spectra.npz'
    power_spectra_savename2 = spectra_directory + '/power_spectra_matter_linear.npy'
    cddf_savename = spectra_directory + '/CDDF.npz'
    flux_ascii_filename = '/Users/kwame/Simulations/Sherwood/planck1_80_1024/snapdir_011/spectest.txt'

    mean_flux_redshift = np.array([[0.132087971214,4.20000007681],[0.316660922444,3.60000011841],[0.463744789112,3.19999998751],
                                   [0.600909218823,2.80000009225],[0.709329306751,2.39999998443],[0.788885808313,2.0000000305]])

    sim_box_ins = get_simulation_box_instance(11, snapshot_directory, 750, 10. * u.km / u.s, spectra_directory, RELOAD_SNAPSHOT=False) #, SPECTROGRAPH_FWHM=40.*u.km/u.s)
    #print("Mean flux =", sim_box_ins.get_mean_flux())
    output = plot_forest_spectrum(plotname, sim_box_ins, spectrum_num=3, redshift_space=False) #, flux_ascii_filename=flux_ascii_filename, rescale_ascii=True)
    #output = plot_CDDF(plotname, cddf_savename, sim_box_ins, load_cddf=True)
    hubble_constant = 0.678 #24
    #output = plot_power_spectra(plotname, power_spectra_savename, simulation_box_instance=sim_box_ins, power_spectra_savename2=power_spectra_savename2, plot_errors=True) #box_length=80. * u.Mpc / hubble_constant, hubble_constant=hubble_constant) #sim_box_ins)
    '''posterior_samples = sample_posterior_distribution(power_spectra_savename, power_spectra_savename2,
        ln_likelihood_gaussian_forest_linear_bias, ln_prior_forest_linear_bias, get_prior_limits_forest_linear_bias(),
        k_h_max=1. / u.Mpc, simulation_box_instance=sim_box_ins, n_steps=200, n_burn_in_steps=50)'''
    #make_corner_plot(plotname, posterior_samples, [r'b_F (1 + beta_F)', r'beta_F'])
    #plot_mean_flux_evolution(plotname, mean_flux_redshift)

    #output = bin_matter_power_spectrum(15, 4, sim_box_ins, cosmo_name='base_planck_lowl_lowLike_highL_post_BAO_2013', savename=power_spectra_savename)
