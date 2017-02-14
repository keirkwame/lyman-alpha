import astropy.units as u
import sys

import boxes as box

if __name__ == "__main__":
    """Input arguments: Snapshot directory path; Snapshot number; Width of grid in samples;
    Resolution of spectra in km s^{-1}; Spectra directory path (with '/snapdir_XXX' if necessary)"""

    snapshot_dir = sys.argv[1]
    snapshot_num = int(sys.argv[2])
    grid_width = int(sys.argv[3])
    spectral_res = float(sys.argv[4]) * (u.km / u.s)
    spectra_full_dir_path = sys.argv[5]

    undodged_spectra_ins = box.SimulationBox(snapshot_num, snapshot_dir, grid_width, spectral_res, reload_snapshot=False, spectra_savedir=spectra_full_dir_path, spectra_savefile_root='gridded_spectra')

    col_den_thresh = 2.e+20 / (u.cm * u.cm)
    dodge_dist = 10. * u.kpc
    dodged_spectra_savefile_root = 'gridded_spectra_DLAs_dodged'

    undodged_spectra_ins.form_skewers_realisation_dodging_DLAs(col_dens_threshold=col_den_thresh, dodge_dist=dodge_dist, savefile_root=dodged_spectra_savefile_root)
