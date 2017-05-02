import numpy as np

def parametric_ratio_model(k_z_mod, a, b, c):
    return 1. / ((a * np.exp(b * k_z_mod) - 1.)**2) + c

def fit_parametric_ratio_models(contaminant_power_1D):
