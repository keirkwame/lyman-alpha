import numpy as np
import scipy.optimize as spo

#from plotting import *

def mcdonald_model(k_z_mod):
    return 0.2 * (1. / (15000. * k_z_mod - 8.9)) #+ 0.018)

def mcdonald_model_full(k_z_mod):
    return 0.2 * ((1. / (15000. * k_z_mod - 8.9)) + 0.018)

def parametric_ratio_model(k_z_mod, a, b, c):
    return 1. / ((a * np.exp(b * k_z_mod) - 1.)**2) + c
    #return 1. / ((a * k_z_mod + b)**2) + c
    #return a * np.exp(b * (k_z_mod**2)) + c

def parametric_ratio_redshift_evolution_model(k_redshift_tuple, a0, a1, b0, b1, c0, c1, redshift_pivot = 2.00):
    (k_z_mod, redshift) = k_redshift_tuple #WRONG DIMENSIONS??? - should be same long 1D???
    a = a0 * (((1. + redshift) / (1. + redshift_pivot)) ** a1)
    b = b0 * (((1. + redshift) / (1. + redshift_pivot)) ** b1)
    c = c0 * (((1. + redshift) / (1. + redshift_pivot)) ** c1)
    #print(a.shape,b.shape,c.shape,k_z_mod.shape,redshift.shape)
    return 1. / ((a * np.exp(b * k_z_mod) - 1.) ** 2) + c #.ravel()

def parametric_ratio_growth_factor_model(k_redshift_tuple, a0, a1, b0, b1, c0, c1, redshift_pivot = 2.00):
    (k_z_mod, redshift) = k_redshift_tuple
    a = a0 * (((1. + redshift) / (1. + redshift_pivot)) ** a1)
    b = b0 * (((1. + redshift) / (1. + redshift_pivot)) ** b1)
    c = c0 * (((1. + redshift) / (1. + redshift_pivot)) ** c1)
    '''box_length = 75.
    omega_m = 0.27
    hubble_z = np.sqrt(omega_m * (1 + redshift) ** 3 + 1. - omega_m)
    c = 1. - (c0 / box_length / hubble_z)'''
    return 1. * (((1. + redshift) / (1. + redshift_pivot)) ** -3.55) / ((a * np.exp(b * k_z_mod) - 1.) ** 2) + c

def parametric_ratio_growth_factor_model_final(k_redshift_tuple, a0, a1, b0, b1, redshift_pivot = 2.00):
    (k_z_mod, redshift) = k_redshift_tuple
    a = a0 * (((1. + redshift) / (1. + redshift_pivot)) ** a1)
    b = b0 * (((1. + redshift) / (1. + redshift_pivot)) ** b1)
    return (((1. + redshift) / (1. + redshift_pivot)) ** -3.55) / ((a * np.exp(b * k_z_mod) - 1.) ** 2)

def fit_parametric_ratio_models(x, y):
    return spo.curve_fit(parametric_ratio_model, x, y)[0]

def fit_two_independent_variable_model(x0, x1, y, model_function, initial_param_values = None, param_bounds = (-np.inf, np.inf)):
    return spo.curve_fit(model_function, (x0, x1), y, p0 = initial_param_values, bounds = param_bounds, method = None) #'lm')

def forest_linear_bias_model(k_mu_tuple, b_F, beta_F):
    (k, mu) = k_mu_tuple
    return (b_F * (1. + (beta_F * (mu ** 2)))) ** 2

def forest_HCD_linear_bias_and_wings_model(k_mu_tuple, b_HCD, beta_HCD, L_HCD):
    b_F = -0.09764619
    beta_F = 1.72410826

    (k, mu) = k_mu_tuple

    F_HCD = np.sinc(k * mu * L_HCD)
    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    forest_auto_bias = forest_linear_bias ** 2
    HCD_linear_bias_and_wings = b_HCD * (1. + (beta_HCD * (mu ** 2))) * F_HCD
    HCD_auto_bias = HCD_linear_bias_and_wings ** 2
    forest_HCD_cross_bias = 2. * forest_linear_bias * HCD_linear_bias_and_wings

    return forest_auto_bias + HCD_auto_bias + forest_HCD_cross_bias

def forest_non_linear_function(k, mu): #k in h / Mpc
    k_NL = 6.40
    alpha_NL = 0.569
    k_P = 15.3
    alpha_P = 2.01
    k_V0 = 1.220
    alpha_V = 1.50
    k_V_prime = 0.923
    alpha_V_prime = 0.451

    k_V = k_V0 * ((1. + (k / k_V_prime)) ** alpha_V_prime)
    return np.exp(((k / k_NL) ** alpha_NL) - ((k / k_P) ** alpha_P) - ((k * mu / k_V) ** alpha_V))

#COURTESY OF BORIS LEISTEDT
def lngaussian(x, mu, sig):
    return - 0.5*((x - mu)/sig)**2 - 0.5*np.log(2*np.pi) - np.log(sig)

def fun(params):
    a_params, b_params, c_params = np.split(params, 3)
    z_powers = np.vstack((0*z+1, z, z**2, 1/z, np.log(z)))  # 5 ** Nz
    a_z = np.dot(a_params, z_powers)[:, None] # Nz * 1
    b_z = np.dot(b_params, z_powers)[:, None] # Nz * 1
    c_z = np.dot(c_params, z_powers)[:, None] # Nz * 1
    Pk_ratio_model = c_z + 1./(a_z * np.exp(b_z * kpar[None, :]) - 1)**2 # Nz * Nk
    #  define some sort of likelihood function with some fictitious error
    Pk_ratio_error = 0.01 * Pk_ratio_data  # can be adjusted
    lnlike = np.sum(lngaussian(Pk_ratio_data, Pk_ratio_model, Pk_ratio_error))
    # define some sort of L1 regularization term
    lnprior = np.log(np.sum(np.abs(a_params)) + np.log(np.abs(b_params)) + np.log(np.abs(c_params)))
    return - (lnlike + lnprior)

def get_optimal_model_parameter_values(initial_param_values):
    return spo.minimize(fun, x0 = initial_param_values)

if __name__ == "__main__":
    #power_file_name = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_LLS_dodged_64_750_10_raw.npz'
    #power_file_name = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_undodged_64_750_10_raw.npz'
    power_file_name = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_undodged_64_750_10_4_6_kMax1.npz'
    #power_linear = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_44_snap64_750_10_k_raw_max_1.npy') #(Mpc/h)^3 ? h
    power_linear = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_44_snap64_750_10_4_6_kMax1.npy')
    #fitting_model = forest_linear_bias_model
    fitting_model = forest_HCD_linear_bias_and_wings_model
    initial_param_values = None
    #initial_param_values = np.array([-0.0288, 0.681, 24.3410])
    param_bounds = (-np.inf, np.inf)
    param_bounds = (np.array([-np.inf, 0.3, 0.]), np.array([0., 0.7, np.inf]))
    k_max = 1. #h / Mpc
    power_file = np.load(power_file_name)

    '''power_box = power_file['arr_0'] * (75. ** 3) #(Mpc/h)^3
    k_box = power_file['arr_1'] / 0.704 #h/Mpc
    mu_box = np.absolute(power_file['arr_2']) #|mu|

    power_large_scales = power_box[k_box < k_max][1:] #Remove k = 0
    k_large_scales = k_box[k_box < k_max][1:]
    mu_large_scales = mu_box[k_box < k_max][1:]'''

    counts_binned = power_file['arr_2'].flatten()
    power_large_scales = power_file['arr_0'].flatten()[counts_binned > 0.] * (75. ** 3) #(Mpc/h)^3
    k_large_scales = power_file['arr_1'].flatten()[counts_binned > 0.] / 0.704 #h/Mpc
    mu_large_scales = np.absolute(power_file['arr_3'].flatten()[counts_binned > 0.]) #|mu|

    power_ratio = power_large_scales / (power_linear * forest_non_linear_function(k_large_scales, mu_large_scales))

    param_array, param_covar = fit_two_independent_variable_model(k_large_scales, mu_large_scales, power_ratio, fitting_model, initial_param_values=initial_param_values, param_bounds=param_bounds)
    print(param_array)
    print(param_covar)

    '''z = np.array([2.0, 2.44, 3.49, 4.43])  # shape Nz
    kpar = ? np.linspace(?)  # shape Nk
    #  the data you want to fit, Pk_ratio_data, is a 2D array, function of (z, k)

    Pk_ratio_data ='''