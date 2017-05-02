import math as mh
import numpy as np
import matplotlib.pyplot as plt
import distinct_colours_py3 as dc

def _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names):
    contaminant_power_1D_z_2_00 = np.load(contaminant_power_1D_f_names[0])[:, 1:]
    contaminant_power_1D_z_2_44 = np.load(contaminant_power_1D_f_names[1])[:, 1:]
    contaminant_power_1D_z_3_49 = np.load(contaminant_power_1D_f_names[2])[:, 1:]
    contaminant_power_1D_z_4_43 = np.load(contaminant_power_1D_f_names[3])[:, 1:]

    return contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43

def _get_k_z_mod(contaminant_power_1D_z_2_00,contaminant_power_1D_z_2_44,contaminant_power_1D_z_3_49,contaminant_power_1D_z_4_43):
    return [contaminant_power_1D_z_2_00[0], contaminant_power_1D_z_2_44[0], contaminant_power_1D_z_3_49[0],contaminant_power_1D_z_4_43[0]]

def make_plot_contaminant_power_absolute_1D(f_name, contaminant_power_1D_f_names):
    contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43 = _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names)
    k_z_mod = _get_k_z_mod(contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43)
    k_z_mod = [k_z_mod[0],k_z_mod[-1]] #Just want minimum and maximum redshift slices

    box_size_z_2_00 = 7111 #km / s
    box_size_z_4_43 = 9199 #km / s
    contaminant_power_absolute_1D = [None] * 12
    for i in range(6):
        contaminant_power_absolute_1D[i] = contaminant_power_1D_z_2_00[i+1] * box_size_z_2_00 * k_z_mod[0] / mh.pi #First row is k_z_mod
        contaminant_power_absolute_1D[6 + i] = contaminant_power_1D_z_4_43[i+1] * box_size_z_4_43 * k_z_mod[1] / mh.pi

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

def make_plot_contaminant_power_ratios_1D(f_name, contaminant_power_1D_f_names):
    #f_name = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D_less_z_diff_colours.png'
    contaminant_power_1D_z_2_00, contaminant_power_1D_z_2_44, contaminant_power_1D_z_3_49, contaminant_power_1D_z_4_43 = _load_contaminant_power_1D_arrays(contaminant_power_1D_f_names)

    k_z_mod = _get_k_z_mod(contaminant_power_1D_z_2_00,contaminant_power_1D_z_2_44,contaminant_power_1D_z_3_49,contaminant_power_1D_z_4_43)
    contaminant_power_ratios_1D_z_2_00 = contaminant_power_1D_z_2_00[3:, :] / contaminant_power_1D_z_2_00[2, :][np.newaxis, :]
    contaminant_power_ratios_1D_z_2_44 = contaminant_power_1D_z_2_44[3:, :] / contaminant_power_1D_z_2_44[2, :][np.newaxis, :]
    contaminant_power_ratios_1D_z_3_49 = contaminant_power_1D_z_3_49[3:, :] / contaminant_power_1D_z_3_49[2, :][np.newaxis, :]
    contaminant_power_ratios_1D_z_4_43 = contaminant_power_1D_z_4_43[3:, :] / contaminant_power_1D_z_4_43[2, :][np.newaxis, :]
    contaminant_power_ratios_1D = [None] * 16
    for i in range(4):
        contaminant_power_ratios_1D[i] = contaminant_power_ratios_1D_z_2_00[i]
        contaminant_power_ratios_1D[4 + i] = contaminant_power_ratios_1D_z_2_44[i]
        contaminant_power_ratios_1D[8 + i] = contaminant_power_ratios_1D_z_3_49[i]
        contaminant_power_ratios_1D[12 + i] = contaminant_power_ratios_1D_z_4_43[i]

    plot_contaminant_power_ratios_1D(k_z_mod, contaminant_power_ratios_1D, f_name)

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
    k_z_mod_list = ([k_z_mod[0],] * 6) + ([k_z_mod[1],] * 6)
    line_labels = ['Total','Forest','LLS','Sub-DLA','Small DLA','Large DLA'] + ([None] * 6)
    dis_cols = ['black'] + dc.get_distinct(5)
    dis_cols[4] = '#CCCC00'  # Dark yellow1
    line_colours = dis_cols * 2
    line_styles = (['-'] * 6) + (['--'] * 6)
    x_label = r'$k_{||}$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    y_label = r'$P_i^\mathrm{1D} k_{||} / \pi$'
    x_log_scale = True
    y_log_scale = True

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_z_mod_list, power_absolute, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles=line_styles)
    fig.subplots_adjust(right=0.97)
    ax.set_xlim([6.e-4, 1.e-1])
    ax.plot([], label=r'$z = 2.00$', color='gray', ls='-')
    ax.plot([], label=r'$z = 4.43$', color='gray', ls='--')
    ax.legend(frameon=False, fontsize=13.0, ncol=2) #, loc='upper right')
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
    ax.plot([], label=r'$z = 2.00$', color='gray', ls='-')
    ax.plot([], label=r'$z = 2.44$', color='gray', ls=':')
    ax.plot([], label=r'$z = 3.49$', color='gray', ls='-.')
    ax.plot([], label=r'$z = 4.43$', color='gray', ls='--')
    ax.legend(frameon=False, fontsize=13.0, ncol=2, loc='lower center')
    plt.savefig(f_name)

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
    ax.plot([], label=r'$z = 2.00$', color='gray', ls='-') #:')
    #ax.plot([], label=r'$z = 2.44$', color='black', ls='-')
    #ax.plot([], label=r'$z = 3.49$', color='black', ls='--')
    ax.plot([], label=r'$z = 4.43$', color='gray', ls='--') #.')
    ax.legend(frameon=False, fontsize=13.0) #, loc='upper right')
    plt.savefig(f_name)

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

    def plot_lines(self, x, y, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles='default', plot_title=''):
        n_lines = len(line_labels)
        if line_styles == 'default':
            line_styles = ['-'] * n_lines
        fig, ax = plt.subplots(1) #, figsize=(8, 12))
        for i in range(n_lines):
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
    contaminant_power_ratios_1D_save_f_name = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D.pdf'
    contaminant_power_1D_f_names = [None] * 4
    contaminant_power_1D_f_names[0] = '/Users/keir/Documents/lyman_alpha_data/contaminant_power_1D_z_2_00.npy'
    contaminant_power_1D_f_names[1] = '/Users/keir/Documents/lyman_alpha_data/contaminant_power_1D_z_2_44.npy'
    contaminant_power_1D_f_names[2] = '/Users/keir/Documents/lyman_alpha_data/contaminant_power_1D_z_3_49.npy'
    contaminant_power_1D_f_names[3] = '/Users/keir/Documents/lyman_alpha_data/contaminant_power_1D_z_4_43.npy'
    make_plot_contaminant_power_ratios_1D(contaminant_power_ratios_1D_save_f_name,contaminant_power_1D_f_names)

    #make_plot_linear_flux_power_3D()