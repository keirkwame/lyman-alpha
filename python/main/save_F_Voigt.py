# coding: utf-8

import numpy as np
import scipy.interpolate as spp
from matplotlib.pyplot import *

k_F_HCD_Voigt = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/k_h_Mpc_F_HCD_Voigt_sub_DLAs_LLS_sim_CDDF_short.txt')
figure()
plot(k_F_HCD_Voigt[:,0],k_F_HCD_Voigt[:,1])
xscale('log')
axhline(y = 0.,color='black',lw=0.5,ls=':')
k_mod_plot = np.linspace(1.e-2, 1, 10000)
low_z_files = [None]*6
low_z_files[0] = np.load('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_LLS_forest_64_750_10_4_6_evenMu_kMax_1.00.npz')
mu_plot = np.nanmean(low_z_files[0]['arr_3'], axis = 0)
Voigt_interpolating_function = spp.interp1d(k_F_HCD_Voigt[:8000:5,0],k_F_HCD_Voigt[:8000:5,1],kind='cubic')
F_Voigt = np.ones_like(k_mod_plot)
F_Voigt = Voigt_interpolating_function(k_mod_plot*mu_plot[0])
scatter(k_mod_plot*mu_plot[0],F_Voigt,color='red')
F_Voigt2 = np.ones_like(k_mod_plot)
F_Voigt2 = Voigt_interpolating_function(k_mod_plot*mu_plot[1])
scatter(k_mod_plot*mu_plot[1],F_Voigt2,color='green')
F_Voigt3 = np.ones_like(k_mod_plot)
F_Voigt3 = Voigt_interpolating_function(k_mod_plot*mu_plot[2])
scatter(k_mod_plot*mu_plot[2],F_Voigt3,color='black')
F_Voigt4 = np.ones_like(k_mod_plot)
F_Voigt4 = Voigt_interpolating_function(k_mod_plot*mu_plot[3])
scatter(k_mod_plot*mu_plot[3],F_Voigt4,color='yellow')
np.savetxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/F_Voigt_residual_z_2_44_plot_large_scales.dat',np.vstack((F_Voigt,F_Voigt2,F_Voigt3,F_Voigt4)).T)
