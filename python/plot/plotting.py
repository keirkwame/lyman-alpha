import math as mh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import astropy.units as u
import distinct_colours_py3 as dc

from parametric_fit import *
from utils import *

def make_plot_voigt_power_spectrum(f_name):
    spectrum_length = 92000 * u.km / u.s
    velocity_bin_width = 2.5 * u.km / u.s
    col_den = [1.e+19, 1.e+20, 10.**(21.)] / (u.cm ** 2)
    sigma = [14., 14., 14.] * u.km / u.s
    gamma = [14., 14., 14.] * u.km / u.s
    amp = [10., 100., 1000.]
    mean_flux = 0.68 #CHECK!!!
    n_curves = 3

    '''contaminant_power_1D_f_names = [None] * 4
    contaminant_power_1D_f_names[0] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_068/contaminant_power_1D_z_2_00.npy'
    contaminant_power_1D_f_names[1] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/contaminant_power_1D_z_2_44.npy'
    contaminant_power_1D_f_names[2] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/contaminant_power_1D_z_3_49.npy'
    contaminant_power_1D_f_names[3] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_052/contaminant_power_1D_z_4_43.npy'
    contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43 = _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names)
    '''

    optical_depth = [None] * n_curves

    power_spectra = [None] * n_curves
    for i in range(n_curves):
        power_spectra[i], k_samples, vel_samples, optical_depth[i], del_lambda_D, z, wavelength_samples = voigt_power_spectrum(spectrum_length, velocity_bin_width, mean_flux, column_density=col_den[i], sigma=sigma[i], gamma=gamma[i], amp=amp[i])
        power_spectra[i] = power_spectra[i][1:] * 10. * k_samples[1:] / mh.pi #/ contaminant_power_1D_z_4_43[2] / 9199
    k_samples_list = [k_samples[1:],] * n_curves

    plot_voigt_power_spectrum(k_samples_list, power_spectra, f_name)

    return vel_samples, optical_depth, del_lambda_D, z, wavelength_samples

def plot_voigt_power_spectrum(k_samples_list, power_spectra, f_name):
    line_labels = [r'$\log N(\mathrm{HI}) = 19$', r'$\log N(\mathrm{HI}) = 20$', r'$\log N(\mathrm{HI}) = 21$']
    #[r'$\tau_0 = 10$',r'$\tau_0 = 100$',r'$\tau_0 = 1000$']
    line_colours = ['black'] * 3
    line_styles = ['-', ':', '--']
    x_label = r'$k_{||}$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    y_label = r'$P^\mathrm{1D} k_{||} / \pi$'
    x_log_scale = True
    y_log_scale = True

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_samples_list, power_spectra, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles)
    fig.subplots_adjust(right=0.97)
    ax.set_xlim([6.e-4, 1.e-1])
    ax.set_ylim([1.e-4, 1.e+0]) #1.e-2, 5.e+3])
    ax.legend(frameon=False, fontsize=13.0) #, loc='upper right')
    plt.savefig(f_name)


def _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names):
    contaminant_power_1D_z_2_00 = np.load(contaminant_power_1D_f_names[0])[:, 1:]
    contaminant_power_1D_z_2_44 = np.load(contaminant_power_1D_f_names[1])[:, 1:]
    contaminant_power_1D_z_3_01 = np.load(contaminant_power_1D_f_names[2])[:, 1:]
    contaminant_power_1D_z_3_49 = np.load(contaminant_power_1D_f_names[3])[:, 1:]
    contaminant_power_1D_z_4_43 = np.load(contaminant_power_1D_f_names[4])[:, 1:]

    return contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_01, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43

def _get_k_z_mod(contaminant_power_1D_z_2_00,contaminant_power_1D_z_2_44,contaminant_power_1D_z_3_01,contaminant_power_1D_z_3_49,contaminant_power_1D_z_4_43):
    return [contaminant_power_1D_z_2_00[0], contaminant_power_1D_z_2_44[0], contaminant_power_1D_z_3_01[0], contaminant_power_1D_z_3_49[0],contaminant_power_1D_z_4_43[0]]

def make_plot_contaminant_power_absolute_1D(f_name, contaminant_power_1D_f_names):
    contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_01, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43 = _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names)
    k_z_mod = _get_k_z_mod(contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43)
    k_z_mod = k_z_mod[0] #,k_z_mod[-1]] #Just want minimum and maximum redshift slices

    box_size_z_2_00 = 7111 #km / s
    #box_size_z_4_43 = 9199 #km / s
    contaminant_power_absolute_1D = [None] * 6 #12
    for i in range(6):
        contaminant_power_absolute_1D[i] = contaminant_power_1D_z_2_00[i+1] * box_size_z_2_00 * k_z_mod / mh.pi #First row is k_z_mod
        #contaminant_power_absolute_1D[6 + i] = contaminant_power_1D_z_4_43[i+1] * box_size_z_4_43 * k_z_mod[1] / mh.pi

    plot_contaminant_power_absolute_1D(k_z_mod, contaminant_power_absolute_1D, f_name)

def make_plot_contaminant_power_absolute_redshift_evolution_1D(f_name, contaminant_power_1D_f_names):
    contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43 = _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names)
    k_z_mod = _get_k_z_mod(contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43)

    box_size_z_2_00 = 7111  # km / s
    box_size_z_2_44 = 7501  # km / s
    box_size_z_3_49 = 8420  # km / s
    box_size_z_4_43 = 9199  # km / s
    contaminant_power_absolute_1D = [None] * 8
    for i in range(2):
        slice_array = np.array([2,6]) #Forest, large DLA
        contaminant_power_absolute_1D[i] = contaminant_power_1D_z_2_00[slice_array[i]] * box_size_z_2_00 * k_z_mod[0] / mh.pi
        contaminant_power_absolute_1D[2 + i] = contaminant_power_1D_z_2_44[slice_array[i]] * box_size_z_2_44 * k_z_mod[1] / mh.pi
        contaminant_power_absolute_1D[4 + i] = contaminant_power_1D_z_3_49[slice_array[i]] * box_size_z_3_49 * k_z_mod[2] / mh.pi
        contaminant_power_absolute_1D[6 + i] = contaminant_power_1D_z_4_43[slice_array[i]] * box_size_z_4_43 * k_z_mod[3] / mh.pi

    plot_contaminant_power_absolute_redshift_evolution_1D(k_z_mod, contaminant_power_absolute_1D, f_name)

def _get_contaminant_power_ratios_1D(contaminant_power_1D_list):
    contaminant_power_ratios_1D_z_2_00 = contaminant_power_1D_list[0][3:, :] / contaminant_power_1D_list[0][2, :][np.newaxis, :]
    contaminant_power_ratios_1D_z_2_44 = contaminant_power_1D_list[1][3:, :] / contaminant_power_1D_list[1][2, :][np.newaxis, :]
    contaminant_power_ratios_1D_z_3_01 = contaminant_power_1D_list[2][3:, :] / contaminant_power_1D_list[2][2, :][np.newaxis, :]
    contaminant_power_ratios_1D_z_3_49 = contaminant_power_1D_list[3][3:, :] / contaminant_power_1D_list[3][2, :][np.newaxis, :]
    contaminant_power_ratios_1D_z_4_43 = contaminant_power_1D_list[4][3:, :] / contaminant_power_1D_list[4][2, :][np.newaxis, :]
    return contaminant_power_ratios_1D_z_2_00,contaminant_power_ratios_1D_z_2_44,contaminant_power_ratios_1D_z_3_01,contaminant_power_ratios_1D_z_3_49,contaminant_power_ratios_1D_z_4_43

def make_plot_contaminant_power_ratios_1D(f_name, contaminant_power_1D_f_names):
    #f_name = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D_less_z_diff_colours.png'
    contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43 = _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names)
    k_z_mod = _get_k_z_mod(contaminant_power_1D_z_2_00,contaminant_power_1D_z_2_44,contaminant_power_1D_z_3_49,contaminant_power_1D_z_4_43)
    contaminant_power_1D_list = [contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43]
    contaminant_power_ratios_1D_z_2_00, contaminant_power_ratios_1D_z_2_44, contaminant_power_ratios_1D_z_3_49, contaminant_power_ratios_1D_z_4_43 = _get_contaminant_power_ratios_1D(contaminant_power_1D_list)

    contaminant_power_ratios_1D = [None] * 16
    for i in range(4):
        contaminant_power_ratios_1D[i] = contaminant_power_ratios_1D_z_2_00[i]
        contaminant_power_ratios_1D[4 + i] = contaminant_power_ratios_1D_z_2_44[i]
        contaminant_power_ratios_1D[8 + i] = contaminant_power_ratios_1D_z_3_49[i]
        contaminant_power_ratios_1D[12 + i] = contaminant_power_ratios_1D_z_4_43[i]

    plot_contaminant_power_ratios_1D(k_z_mod, contaminant_power_ratios_1D, f_name)

def make_plot_model_1D_comparison(save_f_name):
    k_z_mod = np.arange(8.e-4,1.e-1,1.e-4) #s / km
    redshift_expanded = np.ones_like(k_z_mod) * 2.00
    k_redshift_tuple = (k_z_mod,redshift_expanded)

    param_array_lls = [2.20011070,0.01337873,36.4492434,-0.06740389] #,0.98491499,-0.06307393]
    rogers_model_lls_ratio = parametric_ratio_growth_factor_model_final(k_redshift_tuple,param_array_lls[0],param_array_lls[1],param_array_lls[2],param_array_lls[3]) #,param_array_lls[4],param_array_lls[5])
    param_array_sub_dla = [1.50826019,0.09936676,81.3877693,-0.22873123] #,0.98491499,-0.06307393]
    rogers_model_sub_dla_ratio = parametric_ratio_growth_factor_model_final(k_redshift_tuple,param_array_sub_dla[0],param_array_sub_dla[1],param_array_sub_dla[2],param_array_sub_dla[3]) #,param_array_lls[4],param_array_lls[5])
    param_array_small_dla = [1.14153374,0.09370901,162.948304,0.01262570] #,0.98491499,-0.06307393]
    rogers_model_small_dla_ratio = parametric_ratio_growth_factor_model_final(k_redshift_tuple,param_array_small_dla[0],param_array_small_dla[1],param_array_small_dla[2],param_array_small_dla[3]) #,param_array_lls[4],param_array_lls[5])
    param_array_large_dla = [0.86333925,0.29429816,429.580169,-0.49637168] #,0.98491499,-0.06307393]
    rogers_model_large_dla_ratio = parametric_ratio_growth_factor_model_final(k_redshift_tuple,param_array_large_dla[0],param_array_large_dla[1],param_array_large_dla[2],param_array_large_dla[3]) #,param_array_lls[4],param_array_lls[5])

    alpha_params = [0.106,0.059,0.031,0.027] #LLS, sub-DLA, small DLA, large DLA
    small_scale_correction = 0. #(alpha_params[0] * 0.98491499) + (alpha_params[1] * 0.86666716) + (alpha_params[2] * 0.65720658) + (alpha_params[3] * 0.33391171) + 1. - np.sum(alpha_params)
    rogers_model_total_ratio = 1. + (alpha_params[0] * rogers_model_lls_ratio) + (alpha_params[1] * rogers_model_sub_dla_ratio) + (alpha_params[2] * rogers_model_small_dla_ratio) + (alpha_params[3] * rogers_model_large_dla_ratio) + small_scale_correction
    rogers_model_clipped_ratio = 1. + (alpha_params[0] * rogers_model_lls_ratio) + (alpha_params[1] * rogers_model_sub_dla_ratio) + small_scale_correction

    alpha_mcdonald = (rogers_model_total_ratio[0] - 1.) / mcdonald_model(k_z_mod[0])
    mcdonald_model_evaluation = 1. + alpha_mcdonald * mcdonald_model(k_z_mod)

    print(k_z_mod[2], k_z_mod[92])
    print(mcdonald_model_evaluation[92] / mcdonald_model_evaluation[2], rogers_model_clipped_ratio[92] / rogers_model_clipped_ratio[2])

    plot_model_1D_comparison(np.vstack((k_z_mod,k_z_mod,k_z_mod)), np.vstack((rogers_model_total_ratio,rogers_model_clipped_ratio,mcdonald_model_evaluation)), save_f_name)

def make_plot_contaminant_power_ratios_1D_with_templates(f_name_list, contaminant_power_1D_f_names):
    #f_name = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D_less_z_diff_colours.png'
    contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_01, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43 = _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names)
    k_z_mod = _get_k_z_mod(contaminant_power_1D_z_2_00,contaminant_power_1D_z_2_44,contaminant_power_1D_z_3_01,contaminant_power_1D_z_3_49,contaminant_power_1D_z_4_43)
    contaminant_power_1D_list = [contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_01, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43]
    contaminant_power_ratios_1D_z_2_00, contaminant_power_ratios_1D_z_2_44, contaminant_power_ratios_1D_z_3_01, contaminant_power_ratios_1D_z_3_49, contaminant_power_ratios_1D_z_4_43 = _get_contaminant_power_ratios_1D(contaminant_power_1D_list)

    contaminant_power_ratios_1D = [None] * 40
    for i in range(4):
        contaminant_power_ratios_1D[i] = contaminant_power_ratios_1D_z_2_00[i]
        contaminant_power_ratios_1D[4 + i] = contaminant_power_ratios_1D_z_2_44[i]
        contaminant_power_ratios_1D[8 + i] = contaminant_power_ratios_1D_z_3_01[i]
        contaminant_power_ratios_1D[12 + i] = contaminant_power_ratios_1D_z_3_49[i]
        contaminant_power_ratios_1D[16 + i] = contaminant_power_ratios_1D_z_4_43[i]

    #Template fitting
    '''for i in range(16):
        param_array = fit_parametric_ratio_models(k_z_mod[int(i / 4)],contaminant_power_ratios_1D[i])
        print(param_array)
        contaminant_power_ratios_1D[i+16] = parametric_ratio_model(k_z_mod[int(i / 4)],param_array[0],param_array[1],param_array[2])'''

    initial_param_values = np.zeros((4,6))
    initial_param_values[0] = np.array([2.2,0.,37.7,0.,0.99,0.]) #,-3.6]) #,0.]) #np.array([2.2, 37.7, 0.99, 0., 0.11]) #np.array([2.2,0.,37.7,0.,0.99,0.])
    initial_param_values[1] = np.array([1.5, 0., 81.0, 0., 0.86, 0.]) #, -3.6]) #, 0.]) #np.array([1.5, 81.0, 0.86, 0., 0.11]) #np.array([1.5, 0., 81.0, 0., 0.86, 0.])
    initial_param_values[2] = np.array([1.1, 0., 162.2, 0., 0.65, 0.]) #, -3.6]) #, 0.]) #np.array([1.1, 162.2, 0.65, 0., 0.11]) #np.array([1.1, 0., 162.2, 0., 0.65, 0.])
    initial_param_values[3] = np.array([0.87, 0., 422.0, 0., 0.33, 0.]) #, -3.6]) #, 0.]) #np.array([0.87, 422.0, 0.33, 0., 0.11]) #np.array([0.87, 0., 422.0, 0., 0.33, 0.])
    #initial_param_values = [None] * 4
    for i in range(4):
        k_z_mod_expanded = np.concatenate((k_z_mod[0], k_z_mod[1], k_z_mod[2], k_z_mod[3], k_z_mod[4]))
        redshift_expanded = np.array(([2.00] * k_z_mod[0].shape[0]) + ([2.44] * k_z_mod[1].shape[0]) + ([3.01] * k_z_mod[2].shape[0]) + ([3.49] * k_z_mod[3].shape[0]) + ([4.43] * k_z_mod[4].shape[0]))
        data_expanded = np.concatenate((contaminant_power_ratios_1D[i], contaminant_power_ratios_1D[4 + i], contaminant_power_ratios_1D[8 + i], contaminant_power_ratios_1D[12 + i], contaminant_power_ratios_1D[16 + i]))
        param_array = fit_parametric_ratio_redshift_evolution_models(k_z_mod_expanded, redshift_expanded, data_expanded, initial_param_values = initial_param_values[i])
        print(param_array)
        model_evaluation = parametric_ratio_growth_factor_model((k_z_mod_expanded,redshift_expanded),param_array[0],param_array[1],param_array[2],param_array[3],param_array[4],param_array[5]) #,param_array[6])
        contaminant_power_ratios_1D[20 + i] = model_evaluation[:k_z_mod[0].shape[0]]
        contaminant_power_ratios_1D[20 + 4 + i] = model_evaluation[k_z_mod[0].shape[0]:k_z_mod[0].shape[0]+k_z_mod[1].shape[0]]
        contaminant_power_ratios_1D[20 + 8 + i] = model_evaluation[k_z_mod[0].shape[0]+k_z_mod[1].shape[0]:(-1*k_z_mod[3].shape[0])+(-1*k_z_mod[4].shape[0])]
        contaminant_power_ratios_1D[20 + 12 + i] = model_evaluation[(-1*k_z_mod[3].shape[0])+(-1*k_z_mod[4].shape[0]):-1*k_z_mod[4].shape[0]]
        contaminant_power_ratios_1D[20 + 16 + i] = model_evaluation[-1*k_z_mod[4].shape[0]:]

    plot_contaminant_power_ratios_1D_with_templates(k_z_mod, contaminant_power_ratios_1D, f_name_list)

def make_plot_linear_flux_power_3D():
    f_name = '/Users/keir/Documents/dla_papers/paper_3D/linear_flux_power_3D_high_res.png'

    k_pk_linear = np.loadtxt('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_4_snap64.dat')
    k_pk_dark_matter = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/PK-DM-snap_064') #,usecols=[0,1])
    k_pk_flux = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power.npz')
    box_size = 75.
    hubble = 0.70399999999999996

    k_mod = [k_pk_linear[:,0],k_pk_dark_matter[:,0] * 2. * mh.pi / box_size,k_pk_flux['arr_1'][:,0] / hubble]
    power = [k_pk_linear[:,1],k_pk_dark_matter[:,1] * (box_size**3),k_pk_flux['arr_0'][:,0] * (box_size**3)]

    plot_linear_flux_power_3D(k_mod, power, f_name)

def plot_contaminant_power_absolute_1D(k_z_mod,power_absolute,f_name):
    k_z_mod_list = [k_z_mod,] * 6 #) + ([k_z_mod[1],] * 6)
    line_labels = ['Total','Forest','LLS','Sub-DLA','Small DLA','Large DLA'] #+ ([None] * 6)
    dis_cols = ['black'] + dc.get_distinct(5)
    dis_cols[4] = '#CCCC00'  # Dark yellow1
    line_colours = dis_cols #* 2
    line_styles = ['-'] * 6 #) + (['--'] * 6)
    x_label = r'$k_{||}$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    y_label = r'$P_i^\mathrm{1D} k_{||} / \pi$'
    x_log_scale = True
    y_log_scale = True

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_z_mod_list, power_absolute, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles)
    fig.subplots_adjust(right=0.97)
    ax.set_xlim([6.e-4, 1.e-1])
    ax.axvline(x=1.e-3, color='black', ls=':')
    #ax.plot([], label=r'$z = 2.00$', color='gray', ls='-')
    #ax.plot([], label=r'$z = 4.43$', color='gray', ls='--')
    ax.legend(frameon=False, fontsize=13.0) #, ncol=2) #, loc='upper right')
    plt.savefig(f_name)

def plot_contaminant_power_absolute_redshift_evolution_1D(k_z_mod,power_absolute,f_name):
    k_z_mod_list = ([k_z_mod[0], ] * 2) + ([k_z_mod[1], ] * 2) + ([k_z_mod[2],] * 2) + ([k_z_mod[3],] * 2)
    line_labels = ['Forest', 'Large DLA'] + ([None] * 6)
    dis_cols = dc.get_distinct(5)
    line_colours = [dis_cols[0],dis_cols[-1]] * 4
    line_styles = (['-'] * 2) + ([':'] * 2) + (['-.'] * 2) + (['--'] * 2)
    x_label = r'$k_{||}$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    y_label = r'$P_i^\mathrm{1D} k_{||} / \pi$'
    x_log_scale = True
    y_log_scale = True

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_z_mod_list, power_absolute, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles)
    fig.subplots_adjust(right=0.97)
    ax.set_xlim([6.e-4, 1.e-1])
    ax.axvline(x=1.e-3, color='black', ls=':')
    ax.plot([], label=r'$z = 2.00$', color='gray', ls='-')
    ax.plot([], label=r'$z = 2.44$', color='gray', ls=':')
    ax.plot([], label=r'$z = 3.49$', color='gray', ls='-.')
    ax.plot([], label=r'$z = 4.43$', color='gray', ls='--')
    ax.legend(frameon=False, fontsize=13.0, ncol=2, loc='lower center')
    plt.savefig(f_name)

def plot_contaminant_power_ratios_1D_with_templates(k_z_mod,power_ratios,f_name_list):
    k_z_mod_list = ([k_z_mod[0],] * 8) + ([k_z_mod[1],] * 8) + ([k_z_mod[2],] * 8) + ([k_z_mod[3],] * 8) + ([k_z_mod[4],] * 8)
    power_ratios = power_ratios[0:4] + power_ratios[20:24] + power_ratios[4:8] + power_ratios[24:28] + power_ratios[8:12] + power_ratios[28:32] + power_ratios[12:16] + power_ratios[32:36] + power_ratios[16:20] + power_ratios[36:40]
    line_labels = ['LLS','Sub-DLA','Small DLA','Large DLA',None,None,None,None] * 5
    dis_cols = dc.get_distinct(5)
    dis_cols[3] = '#CCCC00' #Dark yellow1
    line_colours = [dis_cols[1],dis_cols[2],dis_cols[3],dis_cols[4]] * 10
    line_styles = ['-','-','-','-','--','--','--','--'] * 5
    x_label = r'$k_{||}$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    y_label = r'$P_i^\mathrm{1D} / P_\mathrm{Forest}^\mathrm{1D}$'
    x_log_scale = True
    y_log_scale = True

    figure, axes = plt.subplots(nrows = 5, ncols = 1, figsize = (6.4,14)) #6.4*2,12.)) #4.8*3)) #, sharex = True)
    #axes[2,1].axis('off')

    plot_instance = Plot()
    figure, axes[0] = plot_instance.plot_lines(k_z_mod_list[0:8], power_ratios[0:8], line_labels[0:8], line_colours[0:8], x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles[0:8], fig=figure, ax=axes[0])
    figure.subplots_adjust(right=0.97)
    axes[0].set_xlim([6.e-4, 1.e-1])
    axes[0].axhline(y=1.0, color='black', ls=':')
    axes[0].axvline(x=1.e-3, color='black', ls=':')
    plt.text(0.15, 0.9, r'\textbf{(a)}: $z = 2.00$', transform = axes[0].transAxes)
    axes[0].plot([], label='Simulation', color='gray', ls='-') #:')
    axes[0].plot([], label='Template', color='gray', ls='--') #.')
    axes[0].legend(frameon=False, fontsize=13.0) #, loc='upper right')
    axes[0].set_xticklabels([])
    axes[0].set_xlabel('')
    #plt.savefig(f_name_list[0])

    figure, axes[1] = plot_instance.plot_lines(k_z_mod_list[8:16], power_ratios[8:16], line_labels[8:16], line_colours[8:16], x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles[8:16], fig=figure, ax=axes[1])
    figure.subplots_adjust(right=0.97)
    axes[1].set_xlim([6.e-4, 1.e-1])
    axes[1].axhline(y=1.0, color='black', ls=':')
    axes[1].axvline(x=1.e-3, color='black', ls=':')
    plt.text(0.15, 0.9, r'\textbf{(b)}: $z = 2.44$', transform=axes[1].transAxes)
    axes[1].plot([], label='Simulation', color='gray', ls='-') #:')
    axes[1].plot([], label='Template', color='gray', ls='--') #.')
    axes[1].legend_.remove() #(frameon=False, fontsize=13.0) #, loc='upper right')
    axes[1].set_xticklabels([])
    axes[1].set_xlabel('')
    #plt.savefig(f_name_list[1])

    figure, axes[2] = plot_instance.plot_lines(k_z_mod_list[16:24], power_ratios[16:24], line_labels[16:24], line_colours[16:24], x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles[16:24], fig=figure, ax=axes[2])
    figure.subplots_adjust(right=0.97)
    axes[2].set_xlim([6.e-4, 1.e-1])
    axes[2].axhline(y=1.0, color='black', ls=':')
    axes[2].axvline(x=1.e-3, color='black', ls=':')
    axes[2].yaxis.labelpad = -26
    axes[2].set_yticks(np.array([0.4, 0.6, 1., 4., 6.]))
    axes[2].set_yticklabels(np.array([r'$4 \times 10^{-1}$', r'$6 \times 10^{-1}$', r'$10^0$', r'$4 \times 10^0$', r'$6 \times 10^0$']))
    plt.text(0.15, 0.9, r'\textbf{(c)}: $z = 3.01$', transform=axes[2].transAxes)
    axes[2].plot([], label='Simulation', color='gray', ls='-') #:')
    axes[2].plot([], label='Template', color='gray', ls='--') #.')
    axes[2].legend_.remove() #(frameon=False, fontsize=13.0) #, loc='upper right')
    axes[2].set_xticklabels([])
    axes[2].set_xlabel('')
    #plt.savefig(f_name_list[2])

    figure, axes[3] = plot_instance.plot_lines(k_z_mod_list[24:32], power_ratios[24:32], line_labels[24:32], line_colours[24:32], x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles[24:32], fig=figure, ax=axes[3])
    figure.subplots_adjust(right=0.97)
    axes[3].set_xlim([6.e-4, 1.e-1])
    axes[3].axhline(y=1.0, color='black', ls=':')
    axes[3].axvline(x=1.e-3, color='black', ls=':')
    axes[3].yaxis.labelpad = -26
    axes[3].set_yticks(np.array([0.4, 0.6, 1., 2., 4.]))
    axes[3].set_yticklabels(np.array([r'$4 \times 10^{-1}$', r'$6 \times 10^{-1}$', r'$10^0$', r'$2 \times 10^0$', r'$4 \times 10^0$']))
    #ax.yaxis.set_major_formatter(tic.FormatStrFormatter('%.1f'))
    #ax.yaxis.set_minor_formatter(tic.FormatStrFormatter('%.1f'))
    plt.text(0.15, 0.9, r'\textbf{(d)}: $z = 3.49$', transform=axes[3].transAxes)
    axes[3].plot([], label='Simulation', color='gray', ls='-') #:')
    axes[3].plot([], label='Template', color='gray', ls='--') #.')
    axes[3].legend_.remove() #(frameon=False, fontsize=13.0) #, ncol=2) #, loc='upper right')
    axes[3].set_xticklabels([])
    axes[3].set_xlabel('')
    #plt.savefig(f_name_list[3])

    figure, axes[4] = plot_instance.plot_lines(k_z_mod_list[32:40], power_ratios[32:40], line_labels[32:40], line_colours[32:40], x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles[32:40], fig=figure, ax=axes[4])
    figure.subplots_adjust(right=0.97)
    axes[4].set_xlim([6.e-4, 1.e-1])
    axes[4].axhline(y=1.0, color='black', ls=':')
    axes[4].axvline(x=1.e-3, color='black', ls=':')
    axes[4].yaxis.labelpad = -24
    #ax.yaxis.set_major_formatter(tic.FormatStrFormatter('%.1f'))
    #ax.yaxis.set_minor_formatter(tic.FormatStrFormatter('%.1f'))
    plt.text(0.15, 0.9, r'\textbf{(e)}: $z = 4.43$', transform=axes[4].transAxes)
    axes[4].plot([], label='Simulation', color='gray', ls='-') #:')
    axes[4].plot([], label='Template', color='gray', ls='--') #.')
    axes[4].legend_.remove() #(frameon=False, fontsize=13.0, ncol=2) #, loc='upper right')

    figure.subplots_adjust(hspace = 0., left = 0.13, top = 0.99, bottom = 0.05)
    plt.savefig(f_name_list[4])

def plot_contaminant_power_ratios_1D(k_z_mod,power_ratios,f_name):
    k_z_mod_list = [k_z_mod[0],k_z_mod[0],k_z_mod[0],k_z_mod[0],k_z_mod[3],k_z_mod[3],k_z_mod[3],k_z_mod[3]] #k_z_mod[1],k_z_mod[1],k_z_mod[1],k_z_mod[1],k_z_mod[2],k_z_mod[2],k_z_mod[2],k_z_mod[2],k_z_mod[3],k_z_mod[3],k_z_mod[3],k_z_mod[3]]
    power_ratios = [power_ratios[0],power_ratios[1],power_ratios[2],power_ratios[3],power_ratios[12],power_ratios[13],power_ratios[14],power_ratios[15]]
    line_labels = ['LLS','Sub-DLA','Small DLA','Large DLA',None,None,None,None] #,None,None,None,None] #None,None,None,None,
    dis_cols = dc.get_distinct(5)
    dis_cols[3] = '#CCCC00' #Dark yellow1
    line_colours = [dis_cols[1],dis_cols[2],dis_cols[3],dis_cols[4],dis_cols[1],dis_cols[2],dis_cols[3],dis_cols[4]] #,dis_cols[2],dis_cols[3],dis_cols[4],dis_cols[5],dis_cols[2],dis_cols[3],dis_cols[4],dis_cols[5]]
    line_styles = ['-','-','-','-','--','--','--','--'] #,'-.','-.','-.','-.'] #':',':',':',':',
    x_label = r'$k_{||}$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    y_label = r'$P_i^\mathrm{1D} / P_\mathrm{Forest}^\mathrm{1D}$'
    x_log_scale = True
    y_log_scale = True

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_z_mod_list, power_ratios, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles)
    fig.subplots_adjust(right=0.97)
    ax.set_xlim([6.e-4, 1.e-1])
    ax.axhline(y=1.0, color='black', ls=':')
    ax.axvline(x=1.e-3, color='black', ls=':')
    ax.plot([], label=r'$z = 2.00$', color='gray', ls='-') #:')
    #ax.plot([], label=r'$z = 2.44$', color='black', ls='-')
    #ax.plot([], label=r'$z = 3.49$', color='black', ls='--')
    ax.plot([], label=r'$z = 4.43$', color='gray', ls='--') #.')
    ax.legend(frameon=False, fontsize=13.0) #, loc='upper right')
    plt.savefig(f_name)

def plot_model_1D_comparison(k_z_mod, model_evaluations, save_f_name):
    line_labels = ['Rogers et al. (2017) model [total contamination]','Rogers et al. (2017) model [residual contamination]','McDonald et al. (2005) model']
    line_colours = dc.get_distinct(3)
    x_label = r'$k_{||}$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    y_label = r'$P_\mathrm{Total}^\mathrm{1D} / P_\mathrm{Forest}^\mathrm{1D}$'
    x_log_scale = True
    y_log_scale = False #True

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_z_mod, model_evaluations, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale)
    fig.subplots_adjust(right=0.97)
    ax.set_xlim([6.e-4, 1.e-1])
    ax.axhline(y=1.0, color='black', ls=':')
    ax.legend(frameon=False, fontsize=13.0)
    plt.savefig(save_f_name)

def plot_linear_flux_power_3D(k_mod,power,f_name):
    line_labels = ['Linear theory', 'Dark matter', 'Lyman-alpha forest flux']
    dis_cols = dc.get_distinct(2)
    line_colours = ['black',dis_cols[0],dis_cols[1]]
    x_label = r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]'
    y_label = r'$P^\mathrm{3D}(|\mathbf{k}|)$ [$\mathrm{Mpc}^3\,h^{-3}$]'
    x_log_scale = True
    y_log_scale = True

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_mod, power, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale)
    ax.set_xlim([5.e-2,50.])
    ax.set_ylim([8.e-6,1.e+4])
    plt.savefig(f_name)

class Plot():
    """Class to make plots"""
    def __init__(self, font_size = 13.0):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=font_size)

        plt.rc('axes', linewidth=1.5)
        plt.rc('xtick.major', width=1.5)
        plt.rc('xtick.minor', width=1.5)
        plt.rc('ytick.major', width=1.5)
        plt.rc('ytick.minor', width=1.5)
        #plt.rc('lines', linewidth=1.0)

    def plot_lines(self, x, y, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles='default', plot_title='', fig = None, ax = None):
        n_lines = len(line_labels)
        if line_styles == 'default':
            line_styles = ['-'] * n_lines
        if fig == None:
            fig, ax = plt.subplots(1) #, figsize=(8, 12))
        for i in range(n_lines):
            print(i)
            ax.plot(x[i], y[i], label=line_labels[i], color=line_colours[i], ls=line_styles[i])
        ax.legend(frameon=False, fontsize=15.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        if x_log_scale == True:
            ax.set_xscale('log')
        if y_log_scale == True:
            ax.set_yscale('log')
        fig.subplots_adjust(right=0.99)

        return fig, ax

if __name__ == "__main__":
    contaminant_power_ratios_1D_save_f_names = [None] * 5
    contaminant_power_ratios_1D_save_f_names[0] = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D_templates_z_2_00_growth_fac2.pdf'
    contaminant_power_ratios_1D_save_f_names[1] = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D_templates_z_2_44_growth_fac2.pdf'
    contaminant_power_ratios_1D_save_f_names[2] = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D_templates_z_3_01_growth_fac2.pdf'
    contaminant_power_ratios_1D_save_f_names[3] = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D_templates_z_3_49_growth_fac2.pdf'
    contaminant_power_ratios_1D_save_f_names[4] = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D_templates_all_z_column.pdf' #z_4_43_growth_fac2.pdf'


    #f_name = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_absolute_1D_limit_z_2_00_only.pdf'

    contaminant_power_1D_f_names = [None] * 5
    contaminant_power_1D_f_names[0] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_068/contaminant_power_1D_z_2_00.npy'
    contaminant_power_1D_f_names[1] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/contaminant_power_1D_z_2_44.npy'
    contaminant_power_1D_f_names[2] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_060/contaminant_power_1D_z_3_01.npy'
    contaminant_power_1D_f_names[3] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/contaminant_power_1D_z_3_49.npy'
    contaminant_power_1D_f_names[4] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_052/contaminant_power_1D_z_4_43.npy'
    #make_plot_contaminant_power_ratios_1D_with_templates(contaminant_power_ratios_1D_save_f_names, contaminant_power_1D_f_names)
    #make_plot_contaminant_power_absolute_1D(f_name, contaminant_power_1D_f_names)

    save_f_name = '/Users/keir/Documents/dla_papers/paper_1D/mcdonald_model_comparison2.png'
    make_plot_model_1D_comparison(save_f_name)

    #make_plot_linear_flux_power_3D()
    #vel_samps, tau, del_lambda_D, z, wavelength_samples = make_plot_voigt_power_spectrum('/Users/keir/Documents/dla_papers/paper_1D/voigt_power_spectrum.pdf')