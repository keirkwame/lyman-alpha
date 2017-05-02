import numpy as np
import scipy.optimize as spo

def parametric_ratio_model(k_z_mod, a, b, c):
    return 1. / ((a * np.exp(b * k_z_mod) - 1.)**2) + c
    #return 1. / ((a * k_z_mod + b)**2) + c
    #return a * np.exp(b * (k_z_mod**2)) + c

def fit_parametric_ratio_models(x, y):
    return spo.curve_fit(parametric_ratio_model, x, y)[0]