import numpy as np
import matplotlib.pyplot as plt
import distinct_colours_py3 as dc

def plot_contaminant_power_ratios_1D(k_z_mod,power_ratios,f_name):
    k_z_mod_list = [k_z_mod[0],k_z_mod[0],k_z_mod[0],k_z_mod[0],k_z_mod[1],k_z_mod[1],k_z_mod[1],k_z_mod[1],k_z_mod[2],k_z_mod[2],k_z_mod[2],k_z_mod[2],k_z_mod[3],k_z_mod[3],k_z_mod[3],k_z_mod[3]]
    line_labels = [None,None,None,None,'LLS','Sub-DLA','Small DLA','Large DLA',None,None,None,None,None,None,None,None]
    dis_cols = dc.get_distinct(4)
    line_colours = [dis_cols[0],dis_cols[1],dis_cols[2],dis_cols[3],dis_cols[0],dis_cols[1],dis_cols[2],dis_cols[3],dis_cols[0],dis_cols[1],dis_cols[2],dis_cols[3],dis_cols[0],dis_cols[1],dis_cols[2],dis_cols[3]]
    line_styles = [':',':',':',':','-','-','-','-','--','--','--','--','-.','-.','-.','-.']
    x_label = r'$|k_\mathrm{LOS}|$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    y_label = r'$P_i^\mathrm{1D} / P_\mathrm{Forest}^\mathrm{1D}$'
    plot_title = ''
    x_log_scale = True
    y_log_scale = True

    plot_instance = Plot()
    fig, ax = plot_instance.plot_lines(k_z_mod_list, power_ratios, line_labels, line_colours, line_styles, x_label, y_label, plot_title, x_log_scale, y_log_scale)
    plt.savefig(f_name)

class Plot():
    """Class to make plots"""
    def __init__(self, font_size = 16.0):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=font_size)

        '''plt.rc('axes', linewidth=2.0)
        plt.rc('xtick.major', width=2.0)
        plt.rc('xtick.minor', width=2.0)
        plt.rc('ytick.major', width=2.0)
        plt.rc('ytick.minor', width=2.0)
        plt.rc('lines', linewidth=1.5)'''

    def plot_lines(self, x, y, line_labels, line_colours, line_styles, x_label, y_label, plot_title, x_log_scale, y_log_scale):
        n_lines = len(line_labels)
        fig, ax = plt.subplots(1) #, figsize=(8, 12))
        for i in range(n_lines):
            ax.plot(x[i], y[i], label=line_labels[i], color=line_colours[i], ls=line_styles[i])
        ax.legend(frameon=False)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        if x_log_scale == True:
            ax.set_xscale('log')
        if y_log_scale == True:
            ax.set_yscale('log')

        return fig, ax

if __name__ == "__main__":
    f_name = '/Users/keir/Documents/dla_papers/paper_1D/contaminant_power_ratios_1D.png'

    contaminant_power_1D_z_2_00 = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_068/contaminant_power_1D_z_2_00.npy')[:,1:]
    contaminant_power_1D_z_2_44 = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/contaminant_power_1D_z_2_44.npy')[:,1:]
    contaminant_power_1D_z_3_49 = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/contaminant_power_1D_z_3_49.npy')[:,1:]
    contaminant_power_1D_z_4_43 = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_052/contaminant_power_1D_z_4_43.npy')[:,1:]

    k_z_mod = [contaminant_power_1D_z_2_00[0],contaminant_power_1D_z_2_44[0],contaminant_power_1D_z_3_49[0],contaminant_power_1D_z_4_43[0]]
    contaminant_power_ratios_1D_z_2_00 = contaminant_power_1D_z_2_00[3:,:] / contaminant_power_1D_z_2_00[2,:][np.newaxis,:]
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