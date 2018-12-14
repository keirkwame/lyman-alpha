import math as mh
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import astropy.units as u
import distinct_colours_py3 as dc

from parametric_fit import *
from utils import *

def make_plot_voigt_power_spectrum(f_name, col_den_min = 1.e+18 / (u.cm ** 2), col_den_max = 1.e+19 / (u.cm ** 2)):
    #col_den_min = 1.e+18 / (u.cm ** 2)
    #col_den_max = 1.e+19 / (u.cm ** 2)

    cddf = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/CDDF.npy')
    col_den_bin_edges = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/CDDF_bin_edges.npy') #/ (u.cm ** 2)
    col_den_samples = np.mean(np.vstack((col_den_bin_edges[:-1],col_den_bin_edges[1:])),axis=0) / (u.cm ** 2)
    col_den = col_den_samples[(col_den_samples >= col_den_min) * (col_den_samples < col_den_max)] #/ (u.cm ** 2)
    weights = cddf[(col_den_samples >= col_den_min) * (col_den_samples < col_den_max)]

    spectrum_length = 200000 * u.km / u.s #92000 * u.km / u.s
    velocity_bin_width = 10. * u.km / u.s
    mean_flux = 0.675940542622
    n_curves = col_den.shape[0]
    print(n_curves)

    #k_d = -23.09
    #k_d = -22.41 #Sub-DLAs
    #N_d = 21.27
    #N_d = 20.90 #Sub-DLAs
    #alpha_d_low_col_den = -1.60 #N < 21.27
    #alpha_d_low_col_den = -1.13 #Sub-DLAs
    #alpha_d_high_col_den = -3.48 #N >= 21.27
    #weights = 10. ** (k_d + (alpha_d_low_col_den * (np.log10(col_den.value) - N_d)))
    #weights[np.log10(col_den.value) >= N_d] = 10. ** (k_d + (alpha_d_high_col_den * (np.log10(col_den.value)[np.log10(col_den.value) >= N_d] - N_d)))

    '''colden_numer =((col_den_max**(alpha_d_low_col_den+2))-(col_den_min**(alpha_d_low_col_den+2)))*(alpha_d_low_col_den+1)
    colden_denom =((col_den_max**(alpha_d_low_col_den+1))-(col_den_min**(alpha_d_low_col_den+1)))*(alpha_d_low_col_den+2)

    colden_numer_2 = ((col_den_max_2**(alpha_d_high_col_den+2))-(col_den_max**(alpha_d_high_col_den+2)))*(alpha_d_high_col_den+1)
    colden_denom_2 = ((col_den_max_2 ** (alpha_d_high_col_den + 1)) - (col_den_max ** (alpha_d_high_col_den + 1))) * (alpha_d_high_col_den + 2)'''

    #col_den = [colden_numer / colden_denom, 2.e+20 / (u.cm ** 2), 1.e+21 / (u.cm ** 2)]
    #col_den = [(colden_numer / colden_denom) + (colden_numer_2 / colden_denom_2), 1.e+21 / (u.cm ** 2)]

    '''contaminant_power_1D_f_names = [None] * 4
    contaminant_power_1D_f_names[0] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_068/contaminant_power_1D_z_2_00.npy'
    contaminant_power_1D_f_names[1] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/contaminant_power_1D_z_2_44.npy'
    contaminant_power_1D_f_names[2] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/contaminant_power_1D_z_3_49.npy'
    contaminant_power_1D_f_names[3] = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_052/contaminant_power_1D_z_4_43.npy'
    contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43 = _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names)
    '''

    optical_depth = [None] * n_curves
    delta_flux_FT = [None] * n_curves
    delta_flux = [None] * n_curves
    power_spectra = [None] * n_curves
    for i in range(n_curves):
        print(i)
        power_spectra[i], k_samples, vel_samples, optical_depth[i], del_lambda_D, z, wavelength_samples, delta_flux_FT[i], delta_flux[i] = voigt_power_spectrum(spectrum_length, velocity_bin_width, mean_flux, column_density=col_den[i])
        power_spectra[i] = power_spectra[i][1:] * 10. * k_samples[1:] / mh.pi

        sign_correction_array = np.ones_like(delta_flux_FT[i].real)
        sign_correction_array[::2] = -1.
        delta_flux_FT[i] = delta_flux_FT[i] * sign_correction_array

    #k_samples_list = [k_samples[1:],] * n_curves
    #plot_voigt_power_spectrum(k_samples_list, power_spectra, f_name)

    #Integration
    equivalent_widths = spi.trapz(1. - np.exp(-1. * np.array(optical_depth)), wavelength_samples.value, axis=1) / 1215.67

    F_Voigt_numer = spi.trapz(np.array(delta_flux_FT) * weights[:,np.newaxis], col_den.value, axis=0)
    #F_Voigt_denom = spi.trapz(weights,col_den.value) #* equivalent_widths
    F_Voigt = F_Voigt_numer #/ F_Voigt_denom
    #F_Voigt = np.fft.rfft(F_Voigt) / F_Voigt.shape[0] * sign_correction_array

    return vel_samples, optical_depth, del_lambda_D, z, wavelength_samples, power_spectra, k_samples[1:] * (7501. * u.km / u.s / (75. * u.Mpc)), delta_flux_FT, delta_flux, weights, col_den, F_Voigt, equivalent_widths #k in h / Mpc

#def voigt_F_HCD(k_parallel, col_den_min, col_den_max):


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
    f_name = '/Users/kwame/Documents/dla_papers/paper_3D/figures/linear_flux_power_3D_bw.pdf'

    k_pk_linear = np.loadtxt('/Users/kwame/Software/lyman-alpha/python/test/P_k_z_2_4_snap64.dat')
    k_pk_dark_matter = np.loadtxt('/Users/kwame/Simulations/Illustris/snapdir_064/PK-DM-snap_064') #,usecols=[0,1])
    k_pk_flux = np.load('/Users/kwame/Simulations/Illustris/snapdir_064/power.npz')
    box_size = 75.
    hubble = 0.70399999999999996

    k_mod = [k_pk_linear[:,0],k_pk_dark_matter[:,0] * 2. * mh.pi / box_size,k_pk_flux['arr_1'][:,0] / hubble]
    power = [k_pk_linear[:,1],k_pk_dark_matter[:,1] * (box_size**3),k_pk_flux['arr_0'][:,0] * (box_size**3)]

    plot_linear_flux_power_3D(k_mod, power, f_name)

def make_plot_F_HCD_Voigt():
    f_name = '/Users/kwame/Papers/dla_papers/paper_3D/F_HCD_Voigt_sinc.pdf'

    output_col_den_1 = make_plot_voigt_power_spectrum(0, col_den_min = 1.6e+17 / (u.cm ** 2), col_den_max = 1.e+19 / (u.cm ** 2))
    output_col_den_2 = make_plot_voigt_power_spectrum(0, col_den_min=1.6e+17 / (u.cm ** 2), col_den_max=2.e+20 / (u.cm ** 2))
    output_col_den_3 = make_plot_voigt_power_spectrum(0, col_den_min=1.6e+17 / (u.cm ** 2), col_den_max=1.e+21 / (u.cm ** 2))

    k_mod = [output_col_den_1[6], output_col_den_2[6], output_col_den_3[6]] #h / Mpc
    F_HCD_Voigt = np.array([output_col_den_1[11], output_col_den_2[11], output_col_den_3[11]]).real
    F_HCD_Voigt_norm = F_HCD_Voigt[:,1:] / F_HCD_Voigt[:,1][:, np.newaxis]

    #Insert sinc function
    #F_HCD_Voigt_norm[-1] = np.sinc(k_mod[-1] * 24.341)

    #plot_F_HCD_Voigt(k_mod, F_HCD_Voigt_norm, f_name)
    return k_mod, F_HCD_Voigt_norm

def plot_F_HCD_Voigt(k_mod, F_HCD_Voigt, f_name):
    line_labels = [r'$\log [N(\mathrm{HI})_\mathrm{min}, N(\mathrm{HI})_\mathrm{max}] = [17, 21]$',r'$\log [N(\mathrm{HI})_\mathrm{min}, N(\mathrm{HI})_\mathrm{max}] = [17, 22]$',r'$\log [N(\mathrm{HI})_\mathrm{min}, N(\mathrm{HI})_\mathrm{max}] = [sinc]$']
    line_colours = ['black'] * 3
    line_styles = ['-', ':', '--']
    x_label = r'$k_{||}$ [$h\,\mathrm{Mpc}^{-1}$]'
    y_label = r'$F_\mathrm{HCD}^\mathrm{Voigt}$'
    x_log_scale = True
    y_log_scale = False

    figure, axis = plt.subplots(1)
    plot_instance = Plot() #font_size = 18.0)
    figure, axis = plot_instance.plot_lines(k_mod, F_HCD_Voigt, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles, fig=figure, ax=axis)
    axis.set_xlim([5.e-3, 1.])
    axis.set_ylim([-0.15, 1.15])
    #axis.axhline(y = 0.0, color = 'black', ls='-.')
    axis.legend(frameon = False, fontsize = 13.0)
    figure.subplots_adjust(right=0.97, left=0.12, bottom=0.13)
    plt.savefig(f_name)

def make_plot_sample_forest_spectra():
    f_name = '/Users/kwame/Papers/dla_papers/paper_3D/sample_forest_spectra.pdf'

    f_name_low_z = '/Users/kwame/Simulations/Illustris_1/snapdir_064/sample_forest_sightline.npy'
    f_name_high_z = '/Users/kwame/Simulations/Illustris_1/snapdir_057/sample_forest_sightline.npy'

    optical_depth_low_z = np.load(f_name_low_z)
    transmitted_flux_low_z = np.exp(-1. * optical_depth_low_z)
    optical_depth_high_z = np.load(f_name_high_z)
    transmitted_flux_high_z = np.exp(-1. * optical_depth_high_z)

    hubble = 0.70399999999999996
    box_size = 75. / hubble #Mpc
    position_low_z = np.linspace(0., box_size, num = transmitted_flux_low_z.shape[0], endpoint = False)
    position_high_z = np.linspace(0., box_size, num = transmitted_flux_high_z.shape[0], endpoint = False)

    transmitted_flux = [transmitted_flux_low_z, transmitted_flux_high_z]
    position = [position_low_z, position_high_z]

    plot_sample_forest_spectra(position, transmitted_flux, f_name)

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

def plot_3D_power_ratio_difference(k_mod, power, f_name): #UNFINISHED
    line_labels = [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$'] + ([None]*12)
    dis_cols = dc.get_distinct(4)
    line_colours = dis_cols * 4
    x_label = r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]'
    y_label = r'$P^\mathrm{3D}(|\mathbf{k}|)$'

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_mod, power, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale)

def make_plot_anisotropic_linear_flux_power_3D():
    f_name = '/Users/keir/Documents/dla_papers/paper_3D/figures/linear_flux_power_3D_anisotropic2.pdf'

    k_pk_linear = np.loadtxt('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_44_snap64_more_modes.dat')
    k_pk_linear_binned = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_4_snap64_binned_eval_4_15.npy')
    k_pk_flux_total = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_undodged_64_750_10_4_15.npz')
    k_pk_flux_forest = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_LLS_dodged_specify_flux_64_750_10_4_15.npz')
    box_size = 75.
    hubble = 0.70399999999999996

    k_mod = [k_pk_flux_total['arr_1'][:,0] / hubble,k_pk_flux_total['arr_1'][:,1] / hubble,k_pk_flux_total['arr_1'][:,2] / hubble,k_pk_flux_total['arr_1'][:,3] / hubble,k_pk_flux_forest['arr_1'][:,0] / hubble,k_pk_flux_forest['arr_1'][:,1] / hubble,k_pk_flux_forest['arr_1'][:,2] / hubble,k_pk_flux_forest['arr_1'][:,3] / hubble,k_pk_linear[:,0]]
    power = [k_pk_flux_total['arr_0'][:,0] * (box_size**3),k_pk_flux_total['arr_0'][:,1] * (box_size**3),k_pk_flux_total['arr_0'][:,2] * (box_size**3),k_pk_flux_total['arr_0'][:,3] * (box_size**3),k_pk_flux_forest['arr_0'][:,0] * (box_size**3),k_pk_flux_forest['arr_0'][:,1] * (box_size**3),k_pk_flux_forest['arr_0'][:,2] * (box_size**3),k_pk_flux_forest['arr_0'][:,3] * (box_size**3),k_pk_linear[:,1],k_pk_linear_binned[:,0],k_pk_linear_binned[:,1],k_pk_linear_binned[:,2],k_pk_linear_binned[:,3]]
    errorbars = ([None]*9) + [power[0] * mh.sqrt(1.) / power[9] / np.sqrt(k_pk_flux_total['arr_2'][:,0]), power[1] * mh.sqrt(1.) / power[10] / np.sqrt(k_pk_flux_total['arr_2'][:,1]), power[2] * mh.sqrt(1.) / power[11] / np.sqrt(k_pk_flux_total['arr_2'][:,2]), power[3] * mh.sqrt(1.) / power[12] / np.sqrt(k_pk_flux_total['arr_2'][:,3])] + ([None]*4)

    plot_anisotropic_linear_flux_power_3D(k_mod, power, errorbars, f_name)

def make_plot_fractional_hcd_effect():
    f_name = '/Users/keir/Documents/dla_papers/paper_3D/figures/fractional_hcd_effect2.pdf'

    k_pk_flux_total = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_undodged_64_750_10_4_6_evenMu_kMax_1.00.npz')
    k_pk_flux_forest = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_LLS_dodged_64_750_10_4_6_evenMu_kMax_1.00.npz')
    k_pk_flux_lls_forest = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_dodged_64_750_10_4_6_evenMu_kMax_1.00.npz')
    k_pk_flux_total_high_z = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/power_undodged_57_750_10_4_6_evenMu_kMax_1.00.npz')
    k_pk_flux_forest_high_z = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/power_DLAs_LLS_dodged_57_750_10_4_6_evenMu_kMax_1.00.npz')
    k_pk_flux_lls_forest_high_z = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/power_DLAs_dodged_57_750_10_4_6_evenMu_kMax_1.00.npz')
    hubble = 0.70399999999999996

    pk_total_fraction = (k_pk_flux_total['arr_0'] - k_pk_flux_forest['arr_0']) / k_pk_flux_forest['arr_0']
    pk_lls_forest_fraction = (k_pk_flux_lls_forest['arr_0'] - k_pk_flux_forest['arr_0']) / k_pk_flux_forest['arr_0']
    pk_total_fraction_high_z = (k_pk_flux_total_high_z['arr_0'] - k_pk_flux_forest_high_z['arr_0']) / k_pk_flux_forest_high_z['arr_0']
    pk_lls_forest_fraction_high_z = (k_pk_flux_lls_forest_high_z['arr_0'] - k_pk_flux_forest_high_z['arr_0']) / k_pk_flux_forest_high_z['arr_0']

    k_mod = [k_pk_flux_total['arr_1'][:,0] / hubble,k_pk_flux_total['arr_1'][:,1] / hubble,k_pk_flux_total['arr_1'][:,2] / hubble,k_pk_flux_total['arr_1'][:,3] / hubble] * 4
    power = [pk_total_fraction[:,0],pk_total_fraction[:,1],pk_total_fraction[:,2],pk_total_fraction[:,3],pk_total_fraction_high_z[:,0],pk_total_fraction_high_z[:,1],pk_total_fraction_high_z[:,2],pk_total_fraction_high_z[:,3],pk_lls_forest_fraction[:,0],pk_lls_forest_fraction[:,1],pk_lls_forest_fraction[:,2],pk_lls_forest_fraction[:,3],pk_lls_forest_fraction_high_z[:,0],pk_lls_forest_fraction_high_z[:,1],pk_lls_forest_fraction_high_z[:,2],pk_lls_forest_fraction_high_z[:,3]]

    plot_fractional_hcd_effect(k_mod, power, f_name)

def make_plot_fractional_dodging_effect():
    f_name = '/Users/keir/Documents/dla_papers/paper_3D/figures/fractional_dodging_effect2.pdf'

    k_pk_flux_total = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_test_dodge_7501_751_751_4_15_DLAs_LLS.npz')
    k_pk_flux_total_high_z = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/power_test_dodge_7501_751_751_4_15_DLAs_LLS_z_3_49.npz')
    hubble = 0.70399999999999996

    pk_total_fraction = (k_pk_flux_total['arr_0'] - k_pk_flux_total['arr_3'][0]) / k_pk_flux_total['arr_3'][0]
    pk_total_fraction_high_z = (k_pk_flux_total_high_z['arr_0'] - k_pk_flux_total_high_z['arr_3'][0]) / k_pk_flux_total_high_z['arr_3'][0]

    k_mod = [k_pk_flux_total['arr_1'][:,0] / hubble,k_pk_flux_total['arr_1'][:,1] / hubble,k_pk_flux_total['arr_1'][:,2] / hubble,k_pk_flux_total['arr_1'][:,3] / hubble] * 2
    power = [pk_total_fraction[:,0],pk_total_fraction[:,1],pk_total_fraction[:,2],pk_total_fraction[:,3],pk_total_fraction_high_z[:,0],pk_total_fraction_high_z[:,1],pk_total_fraction_high_z[:,2],pk_total_fraction_high_z[:,3]]

    plot_fractional_dodging_effect(k_mod, power, f_name)

def make_plot_dodging_statistics():
    f_name = '/Users/keir/Documents/dla_papers/paper_3D/figures/dodging_statistics.png'

    dodge_low_z = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/gridded_spectra_750_10_cofm_difference_DLAs_LLS.npy')
    dodge_high_z = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/gridded_spectra_57_750_10_cofm_difference_DLAs_LLS.npy')

    bin_edges_low_z = np.linspace(0,1200,121)
    bin_edges_high_z = np.linspace(0,2900,291)

    plot_dodging_statistics(dodge_low_z, dodge_high_z, bin_edges_low_z, bin_edges_high_z, f_name)

def make_plot_residual_contamination():
    f_name = '/Users/keir/Documents/dla_papers/paper_3D/figures/residual_contamination2.pdf'

    low_z_files = [None]*3
    low_z_files[0] = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_dodged_64_750_10_4_6_evenMu_kMax_1.00.npz')
    low_z_files[1] = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_LLS_dodged_64_750_10_4_6_evenMu_kMax_1.00.npz')

    high_z_files = [None]*3
    high_z_files[0] = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/power_DLAs_dodged_57_750_10_4_6_evenMu_kMax_1.00.npz')
    high_z_files[1] = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/power_DLAs_LLS_dodged_57_750_10_4_6_evenMu_kMax_1.00.npz')

    low_z_files[2] = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_44_snap64_750_10_4_6_evenMu_k_raw_max_1_pow_k_not_binned.npz')
    high_z_files[2] = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_3_49_snap57_750_10_4_6_evenMu_k_raw_max_1_pow_k_mu_binned.npz')

    low_z_samples = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/samples_residual_z_2_44.npy')
    high_z_samples = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/samples_residual_z_3_49.npy')

    low_z_F_Voigt = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/F_Voigt_residual_z_2_44_plot.dat')
    high_z_F_Voigt = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/F_Voigt_residual_z_3_49_plot.dat')

    n_modes = low_z_files[0]['arr_2']

    box_size = 75.
    hubble = 0.70399999999999996

    k_mod_plot = np.linspace(8.e-2, 1, 1000)
    mu_plot = np.nanmean(low_z_files[0]['arr_3'], axis = 0)

    parameter_percentiles_low_z = np.percentile(low_z_samples, [16, 50, 84], axis=0)  # Percentiles x Parameters
    parameter_percentiles_high_z = np.percentile(high_z_samples, [16, 50, 84], axis=0) #Percentiles x Parameters
    power_max_posterior = [None]*8

    power = [None]*8
    errorbars = [None]*8
    power_diff_low_z = (low_z_files[0]['arr_0'] - low_z_files[1]['arr_0']) * (box_size ** 3.) / low_z_files[2]['arr_0']
    power_diff_high_z = (high_z_files[0]['arr_0'] - high_z_files[1]['arr_0']) * (box_size ** 3.) / high_z_files[2]['arr_0']
    for j in range(4): #Mu bin
        power[j] = power_diff_low_z[:,j]
        power[j + 4] = power_diff_high_z[:,j]
        errorbars[j] = power[j] * mh.sqrt(2.) / np.sqrt(n_modes[:,j])
        errorbars[j + 4] = power[j + 4] * mh.sqrt(2.) / np.sqrt(n_modes[:, j])

        power_max_posterior[j] = forest_HCD_linear_bias_and_parametric_wings_model((k_mod_plot, mu_plot[j]*np.ones_like(k_mod_plot)), parameter_percentiles_low_z[1, 0], parameter_percentiles_low_z[1, 1], parameter_percentiles_low_z[1, 2], parameter_percentiles_low_z[1, 3],0.,0., F_Voigt=low_z_F_Voigt[:,j])
        power_max_posterior[j + 4] = forest_HCD_linear_bias_and_parametric_wings_model((k_mod_plot, mu_plot[j] * np.ones_like(k_mod_plot)), parameter_percentiles_high_z[1, 0], parameter_percentiles_high_z[1, 1], parameter_percentiles_high_z[1, 2], parameter_percentiles_high_z[1, 3], 0., 0., F_Voigt=high_z_F_Voigt[:,j])

    k_mod = [low_z_files[0]['arr_1'][:,0]/hubble,low_z_files[0]['arr_1'][:,1]/hubble,low_z_files[0]['arr_1'][:,2]/hubble,low_z_files[0]['arr_1'][:,3]/hubble]*2

    plot_residual_contamination(k_mod, power, errorbars, k_mod_plot, power_max_posterior, f_name)

def make_plot_bias_tests():
    f_name = '/Users/keir/Documents/dla_papers/paper_3D/figures/bias_tests.png'

    k_max = np.linspace(0.5, 1., 6)
    bias = np.array([-0.261,-0.262,-0.266,-0.269,-0.272,-0.270])
    bias_plus_one_sigma = np.array([0.013,0.009,0.007,0.007,0.005,0.004])
    bias_minus_one_sigma = np.array([0.015, 0.011, 0.008, 0.007, 0.005, 0.004])
    beta = np.array([1.439,1.566,1.653,1.683,1.732,1.722])
    beta_plus_one_sigma = np.array([0.213,0.158,0.119,0.096,0.083,0.072])
    beta_minus_one_sigma = np.array([0.199,0.150,0.117,0.095,0.080,0.072])

    plot_bias_tests(k_max, bias, beta, bias_plus_one_sigma, bias_minus_one_sigma, beta_plus_one_sigma, beta_minus_one_sigma, f_name)

def plot_bias_tests(k_max, bias, beta, bias_plus_one_sigma, bias_minus_one_sigma, beta_plus_one_sigma, beta_minus_one_sigma, f_name):
    labels = [None]
    colours = ['black']
    x_label = [None, r'$|\mathbf{k}|_\mathrm{max}$ [$h\,\mathrm{Mpc}^{-1}$]']
    y_label = [r'$b_\mathrm{Forest} (1 + \beta_\mathrm{Forest})$', r'$\beta_\mathrm{Forest}$']
    x_log_scale = False
    y_log_scale = False
    line_styles = ['']
    marker_styles = ['X']
    errorbar_widths = [1.,]

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4*1., 7.5*1.))
    plot_instance = Plot()

    figure, axis = plot_instance.plot_lines([k_max,], [bias,], labels, colours, x_label[0], y_label[0], x_log_scale, y_log_scale, line_styles=line_styles, marker_styles=marker_styles, fig=figure, ax=axes[0], errorbars=[np.vstack((bias_minus_one_sigma,bias_plus_one_sigma)),], errorbar_widths=errorbar_widths)
    figure, axis = plot_instance.plot_lines([k_max,], [beta,], labels, colours, x_label[1], y_label[1], x_log_scale, y_log_scale, line_styles=line_styles, marker_styles=marker_styles, fig=figure, ax=axes[1], errorbars=[np.vstack((beta_minus_one_sigma,beta_plus_one_sigma)),], errorbar_widths=errorbar_widths)

    axes[0].set_xticklabels([])
    figure.subplots_adjust(hspace=0., top=0.99, bottom=0.08, left=0.15) #right=0.98,
    plt.savefig(f_name)

def make_plot_BOSS_comparison():
    f_name = '/Users/kwame/Documents/dla_papers/paper_3D/BOSS_comparison_sim_fitted2.pdf'

    low_z_file = np.load('/Users/kwame/Simulations/Illustris/snapdir_064/power_DLAs_dodged_64_750_10_4_6_evenMu_kMax_1.00.npz')
    low_z_samples = np.load('/Users/kwame/Simulations/Illustris/snapdir_064/samples_residual_z_2_44.npy')
    low_z_F_Voigt = np.loadtxt('/Users/kwame/Simulations/Illustris/snapdir_064/F_Voigt_residual_z_2_44_plot_large_scales.dat')

    L_HCD = 6. #24.341 #Mpc / h

    k_mod_plot = np.linspace(1.e-2, 1, 10000)
    mu_plot = np.nanmean(low_z_file['arr_3'], axis = 0)

    parameter_percentiles_low_z = np.percentile(low_z_samples, [16, 50, 84], axis=0)  # Percentiles x Parameters
    power_max_posterior = [None]*4
    power_BOSS = [None]*4

    for j in range(mu_plot.shape[0]): #Mu bin
        power_max_posterior[j] = forest_HCD_linear_bias_and_parametric_wings_model((k_mod_plot, mu_plot[j]*np.ones_like(k_mod_plot)), parameter_percentiles_low_z[1, 0], parameter_percentiles_low_z[1, 1], parameter_percentiles_low_z[1, 2], parameter_percentiles_low_z[1, 3],0.,0., F_Voigt=low_z_F_Voigt[:,j])
        power_BOSS[j] = forest_HCD_linear_bias_and_sinc_model((k_mod_plot, mu_plot[j] * np.ones_like(k_mod_plot)), parameter_percentiles_low_z[1, 2], parameter_percentiles_low_z[1, 3], parameter_percentiles_low_z[1, 0], parameter_percentiles_low_z[1, 1], L_HCD)

    plot_BOSS_comparison(k_mod_plot, power_max_posterior, power_BOSS, f_name)

def plot_BOSS_comparison(k_mod_plot, power_max_posterior, power_BOSS, f_name):
    line_labels = [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$'] + ([None] * 4)
    dis_cols = dc.get_distinct(4)
    dis_cols[2] = 'orange' #'#CCCC00'  # Dark yellow1
    line_colours = dis_cols*2
    x_label = r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]'
    y_label = r'$(P^\mathrm{3D}_\mathrm{Contaminated} - P^\mathrm{3D}_\mathrm{Forest}) / P^\mathrm{3D}_\mathrm{Linear}$'
    x_log_scale = True
    y_log_scale = False
    line_styles = (['-'] * 4) + (['--'] * 4)

    figure, axis = plt.subplots(1)
    plot_instance = Plot()

    axis.plot([], label='Voigt model [Rogers et al. 2018]', color='gray', ls='-')
    axis.plot([], label='BOSS model [Bautista et al. 2017]', color='gray', ls='--')

    figure, axis = plot_instance.plot_lines([k_mod_plot,]*4, power_max_posterior, line_labels[:4], line_colours[:4], x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles[:4], fig=figure, ax=axis)
    figure, axis = plot_instance.plot_lines([k_mod_plot,]*4, power_BOSS, line_labels[4:], line_colours[4:], x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles[4:], fig=figure, ax=axis)

    axis.axhline(y=0., color='black', ls=':')
    axis.set_xlim([1.e-2, 1.])
    axis.set_ylim([-0.00155, 0.00595])
    axis.legend(frameon=True, fontsize=9.3, edgecolor='white', facecolor='white', framealpha=1.) #, ncol=2) #, loc='upper right')
    #axis.annotate('', xy = (0.35, 0.35), xytext = (0.48, 0.77), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    axis.annotate('', xy = (0.35, 0.35), xytext = (0.61, 0.91), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'}) #sim_fitted
    #plt.text(0.35, 0.35, r'Decreasing $\mu$', transform=axis.transAxes, horizontalalignment = 'center', verticalalignment = 'top', fontsize = 12.0)
    plt.text(0.61, 0.91, r'Decreasing $\mu$', transform=axis.transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', fontsize = 12.0) #sim_fitted

    figure.subplots_adjust(right=0.97, left=0.17, bottom=0.13)  # figure.subplots_adjust(hspace=0., right=0.98, top=0.99, bottom=0.04, left=0.08)
    plt.savefig(f_name)

def plot_residual_contamination(k_mod,power,errorbars, k_mod_plot, power_max_posterior, f_name):
    line_labels = [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$'] + ([None]*4)
    dis_cols = dc.get_distinct(4)
    dis_cols[2] = 'orange' #'#CCCC00'  # Dark yellow1
    line_colours = dis_cols*2
    x_label = [None] + [r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]']
    y_label = ([r'$(P^\mathrm{3D}_\mathrm{Contaminated} - P^\mathrm{3D}_\mathrm{Forest}) / P^\mathrm{3D}_\mathrm{Linear}$']*2)
    x_log_scale = True
    y_log_scale = False
    line_styles = ['']*8
    marker_styles = ['X']*8
    line_weight_thick = 2.5

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4*1., 7.5*1.))
    plot_instance = Plot()

    axes[0].plot([], label='Simulation', color='gray', ls='', marker='X')
    axes[0].plot([], label='Model', color='gray', ls='-', lw=line_weight_thick)
    #axes[0, 0].plot([], label=r'$+/- 1 \sigma$', color='gray', ls=':', lw=1.5)

    for i in range(axes.shape[0]):
        idx0 = int(i * 4)
        # print(idx0)
        idx1 = int(idx0 + 4)
        power_max_posterior_plot = power_max_posterior[idx0:idx1]
        figure, axes[i] = plot_instance.plot_lines([k_mod_plot,]*4, power_max_posterior_plot, line_labels[idx0:idx1], line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_weights=[line_weight_thick,]*4, fig=figure, ax=axes[i])
        #figure, axes[i, j] = plot_instance.plot_lines([k_mod_plot, ] * 4, np.array(power_percentiles[idx0:idx1])[:,0,:], [None]*4, line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_styles=[':']*4, line_weights=[line_weight_thin,]*4, fig=figure, ax=axes[i, j])
        #figure, axes[i, j] = plot_instance.plot_lines([k_mod_plot, ] * 4, np.array(power_percentiles[idx0:idx1])[:,1,:], [None]*4, line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_styles=[':']*4, line_weights=[line_weight_thin,]*4, fig=figure, ax=axes[i, j])
        figure, axes[i] = plot_instance.plot_lines(k_mod[idx0:idx1], power[idx0:idx1], [None]*4, line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_styles=line_styles[idx0:idx1], marker_styles=marker_styles[idx0:idx1], fig=figure, ax=axes[i], errorbars=errorbars[idx0:idx1])

        axes[i].axhline(y=0., color='black', ls=':')
        axes[i].set_xlim([8.e-2, 1.])
        '''if j > 0:
                axes[i, j].set_yticklabels([])
        '''
        if i == 0:
            axes[i].set_xticklabels([])

    #axes[0,0].legend(frameon=False, fontsize=13.0)

    axes[0].set_ylim([-0.0009, 0.0055])
    axes[1].set_ylim([-0.0025, 0.0099]) #-0.00009

    plt.text(0.03, 0.04, r'\textbf{(a)}: residual contamination, $z = 2.44$', transform=axes[0].transAxes)
    plt.text(0.03, 0.04, r'\textbf{(b)}: residual contamination, $z = 3.49$', transform=axes[1].transAxes)

    axes[0].annotate('', xy = (0.795, 0.4), xytext = (0.795, 0.11), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.81, 0.39, r'Decreasing $\mu$', transform=axes[0].transAxes, horizontalalignment = 'left', verticalalignment = 'top', fontsize = 12.0)
    axes[1].annotate('', xy = (0.795, 0.46), xytext = (0.795, 0.17), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.795, 0.46, r'Decreasing $\mu$', transform=axes[1].transAxes, horizontalalignment = 'center', verticalalignment = 'bottom', fontsize = 12.0)

    figure.subplots_adjust(hspace=0., right=0.97, top=0.99, bottom=0.08, left=0.17) #figure.subplots_adjust(hspace=0., right=0.98, top=0.99, bottom=0.04, left=0.08)
    plt.savefig(f_name)

def make_plot_categories():
    f_name = '/Users/kwame/Papers/dla_papers/paper_3D/categories_posteriors_no_limits_dotted6.pdf'

    low_z_files = [None]*6
    low_z_files[0] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/power_LLS_forest_64_750_10_4_6_evenMu_kMax_1.00.npz')
    low_z_files[1] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/power_subDLAs_forest_64_750_10_4_6_evenMu_kMax_1.00.npz')
    low_z_files[2] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/power_smallDLAs_forest_64_750_10_4_6_evenMu_kMax_1.00.npz')
    low_z_files[3] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/power_largeDLAs_forest_64_750_10_4_6_evenMu_kMax_1.00.npz')
    low_z_files[4] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/power_DLAs_LLS_dodged_64_750_10_4_6_evenMu_kMax_1.00.npz')

    high_z_files = [None] * 6
    high_z_files[0] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/power_LLS_forest_57_750_10_4_6_evenMu_kMax_1.00.npz')
    high_z_files[1] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/power_subDLAs_forest_57_750_10_4_6_evenMu_kMax_1.00_2.npz')
    high_z_files[2] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/power_smallDLAs_forest_57_750_10_4_6_evenMu_kMax_1.00.npz')
    high_z_files[3] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/power_largeDLAs_forest_57_750_10_4_6_evenMu_kMax_1.00.npz')
    high_z_files[4] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/power_DLAs_LLS_dodged_57_750_10_4_6_evenMu_kMax_1.00.npz')

    low_z_files[5] = np.load('/Users/kwame/Software/lyman-alpha/python/test/P_k_z_2_44_snap64_750_10_4_6_evenMu_k_raw_max_1_pow_k_not_binned.npz')
    high_z_files[5] = np.load('/Users/kwame/Software/lyman-alpha/python/test/P_k_z_3_49_snap57_750_10_4_6_evenMu_k_raw_max_1_pow_k_mu_binned.npz')

    low_z_samples = [None]*4
    low_z_samples[0] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/samples_LLS_z_2_44.npy')
    low_z_samples[1] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/samples_sub_DLAs_z_2_44.npy')
    low_z_samples[2] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/samples_small_DLAs_z_2_44.npy')
    low_z_samples[3] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_064/samples_large_DLAs_z_2_44.npy')

    high_z_samples = [None]*4
    high_z_samples[0] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/samples_LLS_z_3_49.npy')
    high_z_samples[1] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/samples_sub_DLAs_z_3_49.npy')
    high_z_samples[2] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/samples_small_DLAs_z_3_49.npy')
    high_z_samples[3] = np.load('/Users/kwame/Simulations/Illustris_1/snapdir_057/samples_large_DLAs_z_3_49.npy')

    low_z_F_Voigt = [None]*4
    low_z_F_Voigt[0] = np.loadtxt('/Users/kwame/Simulations/Illustris_1/snapdir_064/F_Voigt_LLS_z_2_44_plot.dat')
    low_z_F_Voigt[1] = np.loadtxt('/Users/kwame/Simulations/Illustris_1/snapdir_064/F_Voigt_sub_DLAs_z_2_44_plot.dat')
    low_z_F_Voigt[2] = np.loadtxt('/Users/kwame/Simulations/Illustris_1/snapdir_064/F_Voigt_small_DLAs_z_2_44_plot.dat')
    low_z_F_Voigt[3] = np.loadtxt('/Users/kwame/Simulations/Illustris_1/snapdir_064/F_Voigt_large_DLAs_z_2_44_plot.dat')

    high_z_F_Voigt = [None]*4
    high_z_F_Voigt[0] = np.loadtxt('/Users/kwame/Simulations/Illustris_1/snapdir_057/F_Voigt_LLS_z_3_49_plot.dat')
    high_z_F_Voigt[1] = np.loadtxt('/Users/kwame/Simulations/Illustris_1/snapdir_057/F_Voigt_sub_DLAs_z_3_49_plot.dat')
    high_z_F_Voigt[2] = np.loadtxt('/Users/kwame/Simulations/Illustris_1/snapdir_057/F_Voigt_small_DLAs_z_3_49_plot.dat')
    high_z_F_Voigt[3] = np.loadtxt('/Users/kwame/Simulations/Illustris_1/snapdir_057/F_Voigt_large_DLAs_z_3_49_plot.dat')

    model_percentiles_fname_root_low_z = [None]*4
    model_percentiles_fname_root_low_z[0] = '/Users/kwame/Simulations/Illustris_1/snapdir_064/model_percentiles_LLS_z_2_44_'
    model_percentiles_fname_root_low_z[1] = '/Users/kwame/Simulations/Illustris_1/snapdir_064/model_percentiles_sub_DLAs_z_2_44_'
    model_percentiles_fname_root_low_z[2] = '/Users/kwame/Simulations/Illustris_1/snapdir_064/model_percentiles_small_DLAs_z_2_44_'
    model_percentiles_fname_root_low_z[3] = '/Users/kwame/Simulations/Illustris_1/snapdir_064/model_percentiles_large_DLAs_z_2_44_'

    model_percentiles_fname_root_high_z = [None] * 4
    model_percentiles_fname_root_high_z[0] = '/Users/kwame/Simulations/Illustris_1/snapdir_057/model_percentiles_LLS_z_3_49_'
    model_percentiles_fname_root_high_z[1] = '/Users/kwame/Simulations/Illustris_1/snapdir_057/model_percentiles_sub_DLAs_z_3_49_'
    model_percentiles_fname_root_high_z[2] = '/Users/kwame/Simulations/Illustris_1/snapdir_057/model_percentiles_small_DLAs_z_3_49_'
    model_percentiles_fname_root_high_z[3] = '/Users/kwame/Simulations/Illustris_1/snapdir_057/model_percentiles_large_DLAs_z_3_49_'

    n_modes = low_z_files[0]['arr_2']

    box_size = 75.
    hubble = 0.70399999999999996

    k_mod_plot = np.linspace(8.e-2, 1, 1000)
    mu_plot = np.nanmean(low_z_files[0]['arr_3'], axis = 0)

    parameter_percentiles_low_z = np.percentile(np.array(low_z_samples), [16, 50, 84], axis=1)  # Percentiles x Categories x Parameters
    parameter_percentiles_high_z = np.percentile(np.array(high_z_samples), [16, 50, 84], axis=1) #Percentiles x Categories x Parameters
    power_max_posterior = [None]*32
    power_percentiles = [None]*32

    power = [None]*32
    errorbars = [None]*32
    power_diff_low_z = [None]*4
    power_diff_high_z = [None] * len(power_diff_low_z)
    for i in range(len(power_diff_low_z)): #Category
        power_diff_low_z[i] = (low_z_files[i]['arr_0'] - low_z_files[4]['arr_0']) * (box_size ** 3.) / low_z_files[5]['arr_0']
        power_diff_high_z[i] = (high_z_files[i]['arr_0'] - high_z_files[4]['arr_0']) * (box_size ** 3.) / high_z_files[5]['arr_0']
        for j in range(4): #Mu bin
            power[i * len(power_diff_low_z) + j] = power_diff_low_z[i][:,j]
            power[i * len(power_diff_low_z) + j + 16] = power_diff_high_z[i][:,j]
            errorbars[i * len(power_diff_low_z) + j] = power[i * len(power_diff_low_z) + j] * mh.sqrt(2.) / np.sqrt(n_modes[:,j])
            errorbars[i * len(power_diff_low_z) + j + 16] = power[i * len(power_diff_low_z) + j + 16] * mh.sqrt(2.) / np.sqrt(n_modes[:, j])

            power_max_posterior[i * len(power_diff_low_z) + j] = forest_HCD_linear_bias_and_parametric_wings_model((k_mod_plot, mu_plot[j]*np.ones_like(k_mod_plot)), parameter_percentiles_low_z[1, i, 0], parameter_percentiles_low_z[1, i, 1], parameter_percentiles_low_z[1, i, 2], parameter_percentiles_low_z[1, i, 3],0.,0., F_Voigt=low_z_F_Voigt[i][:,j])
            power_max_posterior[i * len(power_diff_low_z) + j + 16] = forest_HCD_linear_bias_and_parametric_wings_model((k_mod_plot, mu_plot[j] * np.ones_like(k_mod_plot)), parameter_percentiles_high_z[1, i, 0], parameter_percentiles_high_z[1, i, 1], parameter_percentiles_high_z[1, i, 2], parameter_percentiles_high_z[1, i, 3], 0., 0., F_Voigt=high_z_F_Voigt[i][:,j])

            power_percentiles[i * len(power_diff_low_z) + j] = np.load(model_percentiles_fname_root_low_z[i] + str(j) + '.npy')[np.array([0,2]),:]
            power_percentiles[i * len(power_diff_low_z) + j + 16] = np.load(model_percentiles_fname_root_high_z[i] + str(j) + '.npy')[np.array([0, 2]), :]

    k_mod = [low_z_files[0]['arr_1'][:,0]/hubble,low_z_files[0]['arr_1'][:,1]/hubble,low_z_files[0]['arr_1'][:,2]/hubble,low_z_files[0]['arr_1'][:,3]/hubble]*8

    plot_categories(k_mod, power, errorbars, k_mod_plot, power_max_posterior, power_percentiles, f_name)

def plot_categories(k_mod,power,errorbars, k_mod_plot, power_max_posterior, power_percentiles, f_name):
    line_labels = ([None]*16) + [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$'] + ([None]*12)
    dis_cols = dc.get_distinct(4)
    dis_cols[2] = 'orange' #'#CCCC00'  # Dark yellow1
    line_colours = dis_cols*8
    x_label = ([None]*3 + [r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]'])*2
    y_label = ([r'$(P^\mathrm{3D}_\mathrm{Contaminated} - P^\mathrm{3D}_\mathrm{Forest}) / P^\mathrm{3D}_\mathrm{Linear}$']*8)
    x_log_scale = True
    y_log_scale = False
    line_styles = ['']*32
    marker_styles = ['X']*32
    line_weight_thick = 2.5
    line_weight_thin = 0.5

    figure, axes = plt.subplots(nrows=4, ncols=2, figsize=(6.4*2., 7.5*2.))
    plot_instance = Plot()

    axes[0, 0].plot([], label='Simulation', color='gray', ls='', marker='X')
    axes[0,0].plot([], label='Model', color='gray', ls='-', lw=line_weight_thick)
    #axes[0, 0].plot([], label=r'$+/- 1 \sigma$', color='gray', ls=':', lw=1.5)

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            idx0 = int((i * 4) + (j * 16))
            #print(idx0)
            idx1 = int(idx0 + 4)
            if i == 2:
                k_plot = k_mod_plot[k_mod_plot <= 0.4]
                k_plot_small_scales = k_mod_plot[k_mod_plot > 0.4]
                power_max_posterior_plot = np.array(power_max_posterior[idx0:idx1])[:,k_mod_plot <= 0.4]
                power_max_posterior_plot_small_scales = np.array(power_max_posterior[idx0:idx1])[:, k_mod_plot > 0.4]
                figure, axes[i, j] = plot_instance.plot_lines([k_plot_small_scales, ] * 4, power_max_posterior_plot_small_scales, line_labels[idx0:idx1], line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_styles=[':']*4, fig=figure, ax=axes[i, j])
            elif i == 3:
                k_plot = k_mod_plot[k_mod_plot <= 0.25]
                k_plot_small_scales = k_mod_plot[k_mod_plot > 0.25]
                power_max_posterior_plot = np.array(power_max_posterior[idx0:idx1])[:, k_mod_plot <= 0.25]
                power_max_posterior_plot_small_scales = np.array(power_max_posterior[idx0:idx1])[:, k_mod_plot > 0.25]
                figure, axes[i, j] = plot_instance.plot_lines([k_plot_small_scales, ] * 4, power_max_posterior_plot_small_scales, line_labels[idx0:idx1], line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_styles=[':'] * 4, fig=figure, ax=axes[i, j])
            else:
                k_plot = k_mod_plot
                power_max_posterior_plot = power_max_posterior[idx0:idx1]
            figure, axes[i, j] = plot_instance.plot_lines([k_plot,]*4, power_max_posterior_plot, line_labels[idx0:idx1], line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_weights=[line_weight_thick,]*4, fig=figure, ax=axes[i, j])
            #figure, axes[i, j] = plot_instance.plot_lines([k_mod_plot, ] * 4, np.array(power_percentiles[idx0:idx1])[:,0,:], [None]*4, line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_styles=[':']*4, line_weights=[line_weight_thin,]*4, fig=figure, ax=axes[i, j])
            #figure, axes[i, j] = plot_instance.plot_lines([k_mod_plot, ] * 4, np.array(power_percentiles[idx0:idx1])[:,1,:], [None]*4, line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_styles=[':']*4, line_weights=[line_weight_thin,]*4, fig=figure, ax=axes[i, j])

            #Use posterior errorbars
            slice_array = np.argmin(np.fabs(k_mod_plot[:,np.newaxis,np.newaxis] - np.array(k_mod[idx0:idx1])[np.newaxis,:,:]), axis=0)
            errorbars_raw = np.array(power_percentiles[idx0:idx1])[:,0,:] - power_max_posterior[idx0:idx1]
            errorbars[idx0:idx1] = [errorbars_raw[0,slice_array[0]],errorbars_raw[1,slice_array[1]],errorbars_raw[2,slice_array[2]],errorbars_raw[3,slice_array[3]]]

            figure, axes[i, j] = plot_instance.plot_lines(k_mod[idx0:idx1], power[idx0:idx1], [None]*4, line_colours[idx0:idx1], x_label[int(idx0 / 4)], y_label[int(idx0 / 4)], x_log_scale, y_log_scale, line_styles=line_styles[idx0:idx1], marker_styles=marker_styles[idx0:idx1], fig=figure, ax=axes[i, j], errorbars=errorbars[idx0:idx1])

            axes[i,j].axhline(y=0., color='black', ls=':')
            axes[i,j].set_xlim([8.e-2, 1.])
            '''if j > 0:
                axes[i, j].set_yticklabels([])
            '''
            if i < 3:
                axes[i,j].set_xticklabels([])

    #axes[0,0].legend(frameon=False, fontsize=13.0)

    axes[0,0].set_ylim([-0.00009, 0.00056])
    axes[1, 0].set_ylim([-0.0006, 0.0026])
    axes[2, 0].set_ylim([-0.0016, 0.0046])
    axes[3, 0].set_ylim([-0.0026, 0.0046])

    axes[0, 1].set_ylim([-0.00019, 0.00161]) #-0.00009
    axes[1, 1].set_ylim([-0.003, 0.009])
    axes[2, 1].set_ylim([-0.008, 0.014])
    axes[3, 1].set_ylim([-0.011, 0.014])

    axes[2, 0].axvline(x=0.4, color='black', ls=':')
    axes[2, 1].axvline(x=0.4, color='black', ls=':')
    axes[3, 0].axvline(x=0.25, color='black', ls=':')
    axes[3, 1].axvline(x=0.25, color='black', ls=':')

    plt.text(0.03, 0.04, r'\textbf{(a)}: LLS, $z = 2.44$', transform=axes[0,0].transAxes)
    plt.text(0.03, 0.04, r'\textbf{(b)}: sub-DLAs, $z = 2.44$', transform=axes[1, 0].transAxes)
    plt.text(0.03, 0.04, r'\textbf{(c)}: small DLAs, $z = 2.44$', transform=axes[2, 0].transAxes)
    plt.text(0.5, 0.92, r'\textbf{(d)}: large DLAs, $z = 2.44$', transform=axes[3, 0].transAxes)

    plt.text(0.03, 0.04, r'\textbf{(e)}: LLS, $z = 3.49$', transform=axes[0, 1].transAxes)
    plt.text(0.03, 0.04, r'\textbf{(f)}: sub-DLAs, $z = 3.49$', transform=axes[1, 1].transAxes)
    plt.text(0.03, 0.04, r'\textbf{(g)}: small DLAs, $z = 3.49$', transform=axes[2, 1].transAxes)
    plt.text(0.5, 0.92, r'\textbf{(h)}: large DLAs, $z = 3.49$', transform=axes[3, 1].transAxes)

    axes[0, 0].annotate('', xy = (0.82, 0.24), xytext = (0.82, 0.51), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.82, 0.24, r'Decreasing $\mu$', transform=axes[0, 0].transAxes, horizontalalignment = 'center', verticalalignment = 'top', fontsize = 12.0)
    axes[0, 1].annotate('', xy = (0.82, 0.18), xytext = (0.82, 0.51), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.82, 0.18, r'Decreasing $\mu$', transform=axes[0, 1].transAxes, horizontalalignment = 'center', verticalalignment = 'top', fontsize = 12.0)

    axes[1, 0].annotate('', xy = (0.82, 0.45), xytext = (0.82, 0.15), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.82, 0.45, r'Decreasing $\mu$', transform=axes[1, 0].transAxes, horizontalalignment = 'center', verticalalignment = 'bottom', fontsize = 12.0)
    axes[1, 1].annotate('', xy = (0.82, 0.5), xytext = (0.82, 0.18), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.82, 0.5, r'Decreasing $\mu$', transform=axes[1, 1].transAxes, horizontalalignment = 'center', verticalalignment = 'bottom', fontsize = 12.0)

    axes[2, 0].annotate('', xy = (0.5, 0.52), xytext = (0.39, 0.27), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.5, 0.52, r'Decreasing $\mu$', transform=axes[2, 0].transAxes, horizontalalignment = 'center', verticalalignment = 'bottom', fontsize = 12.0)
    axes[2, 1].annotate('', xy = (0.5, 0.63), xytext = (0.39, 0.39), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.5, 0.63, r'Decreasing $\mu$', transform=axes[2, 1].transAxes, horizontalalignment = 'center', verticalalignment = 'bottom', fontsize = 12.0)

    axes[3, 0].annotate('', xy = (0.41, 0.8), xytext = (0.41, 0.2), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.41, 0.2, r'Decreasing $\mu$', transform=axes[3, 0].transAxes, horizontalalignment = 'right', verticalalignment = 'top', fontsize = 12.0)
    axes[3, 1].annotate('', xy = (0.41, 0.82), xytext = (0.41, 0.28), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.41, 0.28, r'Decreasing $\mu$', transform=axes[3, 1].transAxes, horizontalalignment = 'right', verticalalignment = 'top', fontsize = 12.0)

    figure.subplots_adjust(hspace=0., wspace=0.25, right=0.98, top=0.99, bottom=0.04, left=0.1)
    plt.savefig(f_name)

def plot_dodging_statistics(dodge_low_z, dodge_high_z, bin_edges_low_z, bin_edges_high_z, f_name):
    labels = [None]*2
    colours = dc.get_distinct(2)[::-1]
    x_label = r'Transverse dodging distance ($\mathrm{kpc}\,\,h^{-1}$)'
    y_label = 'Number of spectra'
    x_log_scale = False
    y_log_scale = True

    plot_instance = Plot()
    figure, axis = plt.subplots(1)
    axis.plot([], label=r'$z = 2.44$', color=colours[1], ls='-', lw=10.)
    axis.plot([], label=r'$z = 3.49$', color=colours[0], ls='-', lw=10.)
    figure, axis = plot_instance.plot_histograms([dodge_high_z,dodge_low_z],[bin_edges_high_z,bin_edges_low_z],labels,colours,x_label,y_label,x_log_scale,y_log_scale,fig=figure,ax=axis)

    axis.set_xlim([0., 2900.])
    axis.set_ylim([1.e-1, 4.e+4])

    plt.savefig(f_name) #, dpi=1000)

def plot_fractional_dodging_effect(k_mod,power,f_name):
    line_labels = [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$'] + ([None]*4)
    dis_cols = dc.get_distinct(4)
    dis_cols[2] = 'orange' #'#CCCC00'  # Dark yellow1
    line_colours = dis_cols*2
    x_label = [None, r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]']
    y_label = [r'$(P^\mathrm{3D}_\mathrm{Original} - P^\mathrm{3D}_\mathrm{Dodged}) / P^\mathrm{3D}_\mathrm{Dodged}$']*2
    x_log_scale = True
    y_log_scale = False
    line_styles = (['-']*8)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 7.5))
    plot_instance = Plot()

    #axes[0].plot([], label=r'$z = 2.44$', color='gray', ls='-')
    #axes[0].plot([], label=r'$z = 3.49$', color='gray', ls='--')
    figure, axes[0] = plot_instance.plot_lines(k_mod[:4], power[:4], line_labels[:4], line_colours[:4], x_label[0], y_label[0], x_log_scale, y_log_scale, line_styles=line_styles[:4], fig=figure, ax=axes[0])
    plt.text(0.55, 0.9, r'\textbf{(a)}: $z = 2.44$', transform=axes[0].transAxes)
    axes[0].axhline(y=0., color='black', ls=':')
    axes[0].axvline(x=1., color='black', ls=':')
    axes[0].set_xlim([8.e-2,10.])
    axes[0].set_ylim([-0.075,0.075])
    axes[0].set_xticklabels([])
    #axes[0].legend(frameon=False, fontsize=13.0)

    figure, axes[1] = plot_instance.plot_lines(k_mod[4:], power[4:], line_labels[4:], line_colours[4:], x_label[1], y_label[1], x_log_scale, y_log_scale, line_styles=line_styles[4:], fig=figure, ax=axes[1])
    plt.text(0.55, 0.9, r'\textbf{(b)}: $z = 3.49$', transform=axes[1].transAxes)
    axes[1].axhline(y=0., color='black', ls=':')
    axes[1].axvline(x=1., color='black', ls=':')
    axes[1].set_xlim([8.e-2, 10.])
    axes[1].set_ylim([-0.075,0.075])

    figure.subplots_adjust(hspace=0., right=0.98, top=0.99, bottom=0.08, left=0.16)
    plt.savefig(f_name)

def plot_fractional_hcd_effect(k_mod,power,f_name):
    line_labels = [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$'] + ([None]*12)
    dis_cols = dc.get_distinct(4)
    dis_cols[2] = 'orange' #'#CCCC00'  # Dark yellow1
    line_colours = dis_cols*4
    x_label = [None, r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]']
    y_label = [r'$(P^\mathrm{3D}_\mathrm{Contaminated} - P^\mathrm{3D}_\mathrm{Forest}) / P^\mathrm{3D}_\mathrm{Forest}$']*2
    x_log_scale = True
    y_log_scale = False
    line_styles = (['-']*4) + (['--']*4) + (['-']*4) + (['--']*4)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 7.5))
    plot_instance = Plot()

    axes[0].plot([], label=r'$z = 2.44$', color='gray', ls='-')
    axes[0].plot([], label=r'$z = 3.49$', color='gray', ls='--')
    figure, axes[0] = plot_instance.plot_lines(k_mod[:8], power[:8], line_labels[:8], line_colours[:8], x_label[0], y_label[0], x_log_scale, y_log_scale, line_styles=line_styles[:8], fig=figure, ax=axes[0])
    plt.text(0.05, 0.05, r'\textbf{(a)}: total contamination', transform=axes[0].transAxes)
    axes[0].axhline(y=0., color='black', ls=':')
    axes[0].set_xlim([8.e-2,1.])
    #axes[0].set_ylim([4.e-4,3.e+3])
    axes[0].set_xticklabels([])
    axes[0].annotate('', xy = (0.88, 0.45), xytext = (0.78, 0.06), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.78, 0.06, r'Decreasing $\mu$', transform=axes[0].transAxes, horizontalalignment = 'right', verticalalignment = 'top', fontsize = 12.0)
    axes[0].legend(frameon=False, fontsize=13.0)

    figure, axes[1] = plot_instance.plot_lines(k_mod[8:], power[8:], line_labels[8:], line_colours[8:], x_label[1], y_label[1], x_log_scale, y_log_scale, line_styles=line_styles[8:], fig=figure, ax=axes[1])
    plt.text(0.05, 0.05, r'\textbf{(b)}: residual contamination', transform=axes[1].transAxes)
    axes[1].axhline(y=0., color='black', ls=':')
    axes[1].set_xlim([8.e-2, 1.])
    axes[1].set_ylim([-0.025, 0.16])
    axes[1].annotate('', xy = (0.92, 0.6), xytext = (0.78, 0.06), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.78, 0.06, r'Decreasing $\mu$', transform=axes[1].transAxes, horizontalalignment = 'right', verticalalignment = 'top', fontsize = 12.0)

    figure.subplots_adjust(hspace=0., right=0.98, top=0.99, bottom=0.08, left=0.17)
    plt.savefig(f_name)

def plot_sample_forest_spectra(position, transmitted_flux, f_name):
    line_labels = [''] * 2
    line_colours = ['black'] * 2
    x_label = [None, r'Comoving position [Mpc]']
    y_label = [r'Normalised transmitted flux'] * 2
    x_log_scale = False
    y_log_scale = False

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 7.5))
    plot_instance = Plot()

    figure, axes[0] = plot_instance.plot_lines([position[0],], [transmitted_flux[0],], [line_labels[0],], [line_colours[0],], x_label[0], y_label[0], x_log_scale, y_log_scale, fig=figure, ax=axes[0])
    plt.text(0.05, 0.03, r'$z = 2.44$', transform=axes[0].transAxes)
    #\textbf{(a)}:
    #axes[0].set_xlim([0., 107.])
    axes[0].set_ylim([-0.12, 1.05])
    axes[0].axhline(y=0., color='black', ls=':')
    axes[0].axhline(y=1., color='black', ls=':')
    axes[0].set_xticklabels([])

    figure, axes[1] = plot_instance.plot_lines([position[1],], [transmitted_flux[1],], [line_labels[1],], [line_colours[1],], x_label[1], y_label[1], x_log_scale, y_log_scale, fig=figure, ax=axes[1])
    plt.text(0.05, 0.03, r'$z = 3.49$', transform=axes[1].transAxes)
    #\textbf{(b)}:
    #axes[1].set_xlim([0., 107.])
    axes[1].set_ylim([-0.12, 1.05])
    axes[1].axhline(y=0., color='black', ls=':')
    axes[1].axhline(y=1., color='black', ls=':')

    figure.subplots_adjust(hspace=0., right=0.99, top=0.99, bottom=0.08, left=0.1)
    plt.savefig(f_name)

def plot_anisotropic_linear_flux_power_3D(k_mod,power,errorbars,f_name):
    line_labels = [r'$0 < \mu < 0.25$', r'$0.25 < \mu < 0.5$', r'$0.5 < \mu < 0.75$', r'$0.75 < \mu < 1$'] + ([None]*4) + ['Linear theory'] + ([None]*8)
    dis_cols = dc.get_distinct(4)
    dis_cols[2] = 'orange' #'#CCCC00'  # Dark yellow1
    line_colours = (dis_cols*2) + ['black'] + (dis_cols*2)
    x_label = [None, r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]']
    y_label = [r'$P^\mathrm{3D}(|\mathbf{k}|)$ [$\mathrm{Mpc}^3\,h^{-3}$]', r'$P^\mathrm{3D}_\mathrm{Flux} / P^\mathrm{3D}_\mathrm{Linear}$']
    x_log_scale = True
    y_log_scale = True
    line_styles = (['-']*4) + (['--']*4) + [':'] + (['-']*4) + (['--']*4)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 7.5))
    plot_instance = Plot()

    axes[0].plot([], label='Total flux', color='gray', ls='-')
    axes[0].plot([], label='Forest flux', color='gray', ls='--')
    figure, axes[0] = plot_instance.plot_lines(k_mod, power[:9], line_labels[:9], line_colours[:9], x_label[0], y_label[0], x_log_scale, y_log_scale, line_styles=line_styles[:9], fig=figure, ax=axes[0], reverse_legend=[7,6,5,4,3,2,1,0,8])
    axes[0].set_xlim([8.e-2,10.])
    axes[0].set_ylim([4.e-4,3.e+3])
    axes[0].set_xticklabels([])
    axes[0].annotate('', xy = (0.6, 0.32), xytext = (0.65, 0.5), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    plt.text(0.6, 0.32, r'Decreasing $\mu$', transform=axes[0].transAxes, horizontalalignment = 'center', verticalalignment = 'top', fontsize = 12.0)
    axes[0].legend(frameon=False, fontsize=13.0)

    figure, axes[1] = plot_instance.plot_lines(k_mod[:8], np.array(power[:8]) / np.array(power[9:]*2), line_labels[9:], line_colours[9:], x_label[1], y_label[1], x_log_scale, y_log_scale, line_styles=line_styles[9:], fig=figure, ax=axes[1], errorbars=errorbars[9:], reverse_legend=True)
    axes[1].plot()
    axes[1].set_xlim([8.e-2, 10.])
    axes[1].set_ylim([5.e-3, 1.e-1])

    figure.subplots_adjust(hspace=0., right=0.98, top=0.99, bottom=0.08, left=0.14) #left=0.13
    plt.savefig(f_name)

def plot_linear_flux_power_3D(k_mod,power,f_name):
    line_labels = ['Linear theory', 'Dark matter', 'Lyman-alpha forest flux']
    dis_cols = dc.get_distinct(2)
    line_colours = ['black']*3 #,dis_cols[0],dis_cols[1]]
    line_styles = [':', '--', '-']
    x_label = r'$|\mathbf{k}|$ [$h\,\mathrm{Mpc}^{-1}$]'
    y_label = r'$P^\mathrm{3D}(|\mathbf{k}|)$ [$\mathrm{Mpc}^3\,h^{-3}$]'
    x_log_scale = True
    y_log_scale = True

    figure, axis = plt.subplots(1)
    plot_instance = Plot(font_size = 18.0)
    figure, axis = plot_instance.plot_lines(k_mod, power, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles, fig=figure, ax=axis)
    axis.set_xlim([5.e-2,50.])
    axis.set_ylim([8.e-6,1.e+4])
    axis.legend(frameon=False, fontsize=18.0)
    figure.subplots_adjust(right=0.99, left=0.11, bottom=0.11)
    plt.savefig(f_name)



class Plot():
    """Class to make plots"""
    def __init__(self, font_size = 15.0):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=font_size)

        plt.rc('axes', linewidth=1.5)
        plt.rc('xtick.major', width=1.5)
        plt.rc('xtick.minor', width=1.5)
        plt.rc('ytick.major', width=1.5)
        plt.rc('ytick.minor', width=1.5)
        #plt.rc('lines', linewidth=1.0)

    def plot_lines(self, x, y, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles='default', line_weights='default', marker_styles = None, plot_title='', fig = None, ax = None, errorbars=False, errorbar_widths='default', reverse_legend=False):
        n_lines = len(line_labels)
        if line_styles == 'default':
            line_styles = ['-'] * n_lines
        if line_weights == 'default':
            line_weights = [1.5,] * n_lines
        if marker_styles == None:
            marker_styles = [''] * n_lines
        if fig == None:
            fig, ax = plt.subplots(1) #, figsize=(8, 12))
        if errorbar_widths == 'default':
            errorbar_widths = [1.5,] * n_lines
        if reverse_legend == False:
            line_iterator = range(n_lines)
        elif reverse_legend == True:
            line_iterator = range(n_lines)[::-1]
        else:
            line_iterator = reverse_legend
        for i in line_iterator:
            print(i)
            ax.plot(x[i], y[i], label=line_labels[i], color=line_colours[i], ls=line_styles[i], lw=line_weights[i], marker=marker_styles[i])
            if errorbars is not False:
                if errorbars[i] is not None:
                    ax.errorbar(x[i], y[i], yerr=errorbars[i], ecolor=line_colours[i], ls='', elinewidth=errorbar_widths[i])
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

    def plot_histograms(self, arrays, bin_edges, labels, colours, x_label, y_label, x_log_scale, y_log_scale, plot_title='', fig = None, ax = None):
        n_histograms = len(labels)
        if fig == None:
            fig, ax = plt.subplots(1) #, figsize=(8, 12))
        for i in range(n_histograms):
            print('Plotting histogram', str(i+1))
            ax.hist(arrays[i], bins = bin_edges[i], label = labels[i], color = colours[i])
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

    #save_f_name = '/Users/keir/Documents/dla_papers/paper_1D/mcdonald_model_comparison2.png'
    #make_plot_model_1D_comparison(save_f_name)

    #make_plot_linear_flux_power_3D()
    #vel_samps, tau, del_lambda_D, z, wavelength_samps, power_spectra, k_samps, delta_flux_FT, delta_flux, weights, col_den, F_Voigt, equiv_widths = make_plot_voigt_power_spectrum('/Users/keir/Documents/dla_papers/paper_1D/voigt_power_spectrum.pdf')


    #3D paper
    #make_plot_categories()
    #make_plot_BOSS_comparison()
    k, F_HCD = make_plot_F_HCD_Voigt()

    #Thesis
    #make_plot_sample_forest_spectra()