import numpy as np
import astropy.units as u
import math as mh
import sys

import power_spectra as spe
import boxes as box
import fourier_estimators as fou
import utils as uti

if __name__ == "__main__":
    """Input arguments: Snapshot directory path; Snapshot number; Width of grid in samples;
    Resolution of spectra in km s^{-1}; Spectra directory path (with '/snapdir_XXX' if necessary)"""

    snapshot_dir = sys.argv[1]
    snapshot_num = int(sys.argv[2])
    grid_width = int(sys.argv[3])
    spectral_res = float(sys.argv[4]) * (u.km / u.s)
    spectra_full_dir_path = sys.argv[5]
    model_cosmology_filename = sys.argv[6]

    undodged_spectra_ins = box.SimulationBox(snapshot_num, snapshot_dir, grid_width, spectral_res, reload_snapshot=False, spectra_savedir=spectra_full_dir_path, spectra_savefile_root='gridded_spectra_LLS_forest')
    #print(np.mean(np.exp(-1. * undodged_spectra_ins.get_optical_depth())))

    undodged_spectra_ins.convert_fourier_units_to_distance = True
    spectra_box = undodged_spectra_ins.skewers_realisation(mean_flux_specified=0.675940542622) #0.67573418185771716) #_hydrogen_overdensity(ion = -1)
    #spectra_box = undodged_spectra_ins.skewers_realisation(tau_scaling_specified = 1.5)
    k_box = undodged_spectra_ins.k_box()
    mu_box = undodged_spectra_ins.mu_box()

    for i in [1.]: #np.arange(0.5,0.9,0.1):
        print(i)
        n_mu_bins = 4
        n_k_bins = 6
        k_min = np.min(k_box[k_box > 0. / u.Mpc])
        k_max = i * 0.704 / u.Mpc #np.max(k_box)
        k_bin_max = mh.exp(mh.log(k_max.value) + ((mh.log(k_max.value) - mh.log(k_min.value)) / (n_k_bins - 1))) / u.Mpc
        k_bin_edges = np.exp(np.linspace(mh.log(k_min.value), mh.log(k_bin_max.value), n_k_bins + 1)) / u.Mpc
        #k_bin_edges[-2] = k_max
        mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)
        #mu_bin_edges = np.array([0., 0.5, 0.8, 0.95, 1.])

        fourier_estimator_instance = fou.FourierEstimator3D(spectra_box)
        #power,df_hat = fourier_estimator_instance.get_flux_power_3D()
        power_binned, k_binned, mu_binned, bin_counts = fourier_estimator_instance.get_flux_power_3D_two_coords_hist_binned(k_box,np.absolute(mu_box),k_bin_edges,mu_bin_edges,bin_coord2=True,std_err=False)

        np.savez('/home/keir/Data/Illustris_big_box_spectra/snapdir_064/power_LLS_forest_64_750_10_4_6_evenMu_kMax_%.2f.npz'%i,power_binned,k_binned,bin_counts,mu_binned) #,model_power_binned)'''
