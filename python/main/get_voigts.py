import voigtfit as vf
import astropy.units as u

import boxes as box

if __name__ == "__main__":
    box_instance = box.SimulationBox(64, '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra', 300, 25 * u.km / u.s, reload_snapshot=False, spectra_savefile_root='gridded_spectra', spectra_savedir='/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064')
    #_smallDLAs_forest
    optical_depth = box_instance.get_optical_depth()

    #voigt_output = vf.get_voigt_systems(optical_depth[0], 25)
    voigt_instance = vf.Profiles(optical_depth[23424], 25)
    voigt_instance.do_fit(tol = 1.e-2)
    voigt_output = voigt_instance.get_fitted_profile()