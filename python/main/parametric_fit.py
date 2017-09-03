import numpy as np
import math as mh
import copy as cp
import emcee as mc
import corner as co
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import numpy.random as npr
import scipy.optimize as spo
import scipy.interpolate as spp

#plt.rcParams['figure.figsize'] = 8, 6
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

def fit_two_independent_variable_model(x0, x1, y, model_function, initial_param_values = None, y_sigma = None, param_bounds = (-np.inf, np.inf)):
    return spo.curve_fit(model_function, (x0, x1), y, p0 = initial_param_values, sigma = y_sigma, bounds = param_bounds, method = None) #'lm')


#Bayesian inference
#Likelihoods
def lnlike_forest_linear_bias_model(param_array, x, y, yerr, power_linear=None):
    model_evaluation = forest_linear_bias_model(x, param_array[0], param_array[1], param_array[2], param_array[3], param_array[4], power_linear=power_linear)
    return -0.5 * np.sum(((y - model_evaluation)**2) / ((yerr * model_evaluation * mh.sqrt(2.))**2))

def lnlike_forest_HCD_linear_bias_and_wings_model(param_array, x, y, yerr):
    model_evaluation = forest_HCD_linear_bias_and_wings_model(x, param_array[0], param_array[1], param_array[2])
    return -0.5 * np.sum(((y - model_evaluation)**2) / (np.mean(yerr * model_evaluation * mh.sqrt(2.))**2)) #model_evaluation

def lnlike_forest_HCD_linear_bias_and_wings_model_fully_floated(param_array, x, y, yerr):
    model_evaluation = forest_HCD_linear_bias_and_wings_model_fully_floated(x, param_array[0], param_array[1], param_array[2], param_array[3], param_array[4])
    return -0.5 * np.sum(((y - model_evaluation)**2) / ((yerr * model_evaluation * mh.sqrt(2.))**2)) #model_evaluation

def lnlike_forest_HCD_linear_bias_and_Voigt_wings_model(param_array, x, y, yerr):
    model_evaluation = forest_HCD_linear_bias_and_Voigt_wings_model_errorbars(x, param_array[0], param_array[1], param_array[2], param_array[3]) #, param_array[4], param_array[5], param_array[6])
    errorbar_amp_evaluation = forest_HCD_linear_bias_and_Voigt_wings_model_errorbars(x, param_array[0], param_array[1], param_array[2], param_array[3]) #, param_array[4], param_array[5], param_array[6])
    return -0.5 * np.sum(((y - model_evaluation)**2) / ((yerr * errorbar_amp_evaluation * mh.sqrt(2.))**2)) #model_evaluation

def lnlike_forest_HCD_linear_bias_and_sinc_model(param_array, x, y, yerr):
    model_evaluation = forest_HCD_linear_bias_and_sinc_model_full(x, param_array[0], param_array[1], param_array[2], param_array[3], param_array[4])
    errorbar_amp_evaluation = forest_HCD_linear_bias_and_sinc_model_full(x, param_array[0], param_array[1], param_array[2], param_array[3], param_array[4])
    return -0.5 * np.sum(((y - model_evaluation)**2) / ((yerr * errorbar_amp_evaluation * mh.sqrt(2.))**2))

def lnlike_forest_HCD_linear_bias_and_parametric_wings_model(param_array, x, y, yerr, power_linear=None):
    k_max = 1.
    mu_max = 0.
    slice_array = np.logical_not((x[0] > k_max) * (x[1] > mu_max))
    #print(slice_array.shape)
    '''x[0] = x[0][slice_array]
    x[1] = x[1][slice_array]
    y = y[slice_array]
    yerr = yerr[slice_array]'''

    model_evaluation = forest_HCD_linear_bias_and_parametric_wings_model_full(x, param_array[0], param_array[1], param_array[2], param_array[3], param_array[4], param_array[5], power_linear=power_linear, slice_array=slice_array)
    errorbar_amp_evaluation = forest_HCD_linear_bias_and_parametric_wings_model_full(x, param_array[0], param_array[1], param_array[2], param_array[3], param_array[4], param_array[5], power_linear=power_linear, slice_array=slice_array)
    return -0.5 * np.sum(((y[slice_array] - model_evaluation[slice_array])**2) / ((yerr[slice_array] * errorbar_amp_evaluation[slice_array] * mh.sqrt(2.))**2))

def lnlike_joint(param_array, x, y, yerr, power_linear=None):
    return lnlike_forest_HCD_linear_bias_and_parametric_wings_model(param_array, x, y[1], yerr, power_linear=power_linear) + lnlike_forest_linear_bias_model(np.concatenate((param_array[:2],param_array[3:])), x, y[0], yerr, power_linear=power_linear)

#Priors
def lnprior_forest_linear_bias_model(param_array):
    if -10. < param_array[0] < 0. and 0. < param_array[1] < 10.: #b_F (1 + beta_F); beta_F
        return 0.
    else:
        return -np.inf

def lnprior_forest_HCD_linear_bias_and_wings_model(param_array):
    if -0.2 < param_array[0] < 0. and 0. < param_array[1] < 1.6 and 0. < param_array[2] < 70.: #b_HCD; beta_HCD; L_HCD
        return -0.5 * (((param_array[1] - 0.5) / 0.2)**2) #0.
    else:
        return -np.inf

def lnprior_forest_HCD_linear_bias_and_wings_model_fully_floated(param_array):
    if -0.2 < param_array[0] < 0. and 0. < param_array[1] < 1.6 and 0. < param_array[2] < 70. and -10. < param_array[3] < 0. and 0. < param_array[4] < 10.: #b_HCD; beta_HCD; L_HCD
        return (-0.5 * (((param_array[1] - 0.5) / 0.2)**2)) + (-0.5 * (((param_array[3] - -0.267) / 0.004)**2)) + (-0.5 * (((param_array[4] - 1.617) / 0.068)**2)) #-0.267; 1.617; 0.004; 0.068
    else:
        return -np.inf

def lnprior_forest_HCD_linear_bias_and_Voigt_wings_model(param_array):
    if -0.2 < param_array[0] < 0. and 0. < param_array[1] < 10. and -10. < param_array[2] < 0. and 0. < param_array[3] < 10.: #and -1.e-30 < param_array[5] < 1.e-30 and -1.e-30 < param_array[6] < 1.e-30: #b_HCD; beta_HCD; L_HCD
        return (-0.5 * (((param_array[1] - 0.7) / 0.2)**2)) + (-0.5 * (((param_array[2] - -0.267) / 0.004)**2)) + (-0.5 * (((param_array[3] - 1.617) / 0.068)**2)) #-0.267; 1.617; 0.004; 0.068 #-0.2696; 0.0044; 1.7205; 0.0732
    else:
        return -np.inf

def lnprior_forest_HCD_linear_bias_and_sinc_model(param_array):
    if -0.2 < param_array[0] < 0. and 0. < param_array[1] < 10. and -10. < param_array[2] < 0. and 0. < param_array[3] < 10. and 0. < param_array[4] < 50.:
        return (-0.5 * (((param_array[1] - 0.7) / 0.2)**2)) + (-0.5 * (((param_array[2] - -0.267) / 0.004)**2)) + (-0.5 * (((param_array[3] - 1.617) / 0.068)**2))
    else:
        return -np.inf

def lnprior_forest_HCD_linear_bias_and_parametric_wings_model(param_array): #Also for joint analysis
    if -10. < param_array[0] < 0. and 0. < param_array[1] < 10. and -0.02 < param_array[2] < 0. and 0. < param_array[3] < 2. and -1.e-30 < param_array[4] < 1.e-30 and -1.e-30 < param_array[5] < 1.e-30: #and -10. < param_array[2] < 0. and 0. < param_array[3] < 10. and -1.e-30 < param_array[4] < 1.e-30
        return (-0.5 * (((param_array[3] - 0.7) / 0.2)**2)) + (-0.5 * (((param_array[2] - -0.003) / 0.001)**2)) #+ (-0.5 * (((param_array[4] - 0.4) / 0.2)**2)) + (-0.5 * (((param_array[5] - 0.25) / 0.15)**2)) #+ (-0.5 * (((param_array[2] - -0.267) / 0.004)**2)) + (-0.5 * (((param_array[3] - 1.617) / 0.068)**2))
    else:
        return -np.inf

def get_starting_positions_in_uniform_prior(prior_limits, n_walkers):
    n_params = prior_limits.shape[0]
    return npr.uniform(low = prior_limits[:,0], high = prior_limits[:,1], size = (n_walkers, n_params))

#Posteriors
def lnprob(param_array, x, y, yerr, lnlike, lnprior, power_linear=None):
    lnprior_evaluation = lnprior(param_array)
    if not np.isfinite(lnprior_evaluation):
        return -np.inf
    else:
        return lnprior_evaluation + lnlike(param_array, x, y, yerr, power_linear=power_linear)

#Sampling
def gelman_rubin_convergence_statistic(mcmc_chains): #dims: Walkers * Steps * Parameters
    n_walkers = mcmc_chains.shape[0]
    n_steps = mcmc_chains.shape[1]

    within_chain_variance = np.mean(np.var(mcmc_chains, axis = 1, ddof = 1), axis = 0) #dims: Parameters

    chain_means = np.mean(mcmc_chains, axis = 1)
    between_chain_variance = np.var(chain_means, axis = 0, ddof = 1) * n_steps

    posterior_marginal_variance = ((n_steps - 1) * within_chain_variance / n_steps) + ((n_walkers + 1) * between_chain_variance / n_steps / n_walkers)
    return np.sqrt(posterior_marginal_variance / within_chain_variance)

def get_posterior_samples(lnlike, lnprior, x, y, yerr, n_params, n_walkers, n_steps, n_burn_in_steps, starting_positions, power_linear=None):
    sampler = mc.EnsembleSampler(n_walkers, n_params, lnprob, args = (x, y, yerr, lnlike, lnprior, power_linear))
    sampler.run_mcmc(starting_positions, n_steps)
    return sampler.chain[:, n_burn_in_steps:, :].reshape((-1, n_params)), sampler.chain[:, n_burn_in_steps:, :], sampler #, mc.autocorr.integrated_time(sampler.chain,axis=1), sampler.chain

#Models
def forest_linear_bias_model(k_mu_tuple, b_F_weighted, beta_F, a, b, c, power_linear=None):
    '''b_F_weighted = -0.274
    beta_F = 1.671'''

    (k, mu) = k_mu_tuple
    b_F = b_F_weighted / (1. + beta_F)
    return ((b_F * (1. + (beta_F * (mu ** 2)))) ** 2) * (forest_non_linear_function(k, mu, power_linear=power_linear) ** 1.)

def forest_HCD_linear_bias_and_wings_model(k_mu_tuple, b_HCD, beta_HCD, L_HCD):
    b_F = -0.102 #-0.122 #-0.09764619
    beta_F = 1.617 #1.663 #1.72410826

    (k, mu) = k_mu_tuple

    F_HCD = np.sinc(k * mu * L_HCD / mh.pi)
    #F_HCD = np.sin(k * mu * L_HCD) / (k * mu * L_HCD)
    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    forest_auto_bias = forest_linear_bias ** 2
    HCD_linear_bias_and_wings = b_HCD * (1. + (beta_HCD * (mu ** 2))) * F_HCD
    HCD_auto_bias = HCD_linear_bias_and_wings ** 2
    forest_HCD_cross_bias = 2. * forest_linear_bias * HCD_linear_bias_and_wings

    return HCD_auto_bias + forest_HCD_cross_bias #+ forest_auto_bias

def forest_HCD_linear_bias_and_wings_model_fully_floated(k_mu_tuple, b_HCD, beta_HCD, L_HCD, b_F_weighted, beta_F):
    #b_F = -0.102 #-0.122 #-0.09764619
    #beta_F = 1.617 #1.663 #1.72410826
    b_F = b_F_weighted / (1. + beta_F)

    (k, mu) = k_mu_tuple

    F_HCD = np.sinc(k * mu * L_HCD / mh.pi) #** 2
    #F_HCD = np.sin(k * mu * L_HCD) / (k * mu * L_HCD)
    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    forest_auto_bias = forest_linear_bias ** 2
    HCD_linear_bias_and_wings = b_HCD * (1. + (beta_HCD * (mu ** 2))) * F_HCD
    HCD_auto_bias = HCD_linear_bias_and_wings ** 2
    forest_HCD_cross_bias = 2. * forest_linear_bias * HCD_linear_bias_and_wings

    return HCD_auto_bias + forest_HCD_cross_bias #+ forest_auto_bias

def forest_HCD_linear_bias_and_Voigt_wings_model(k_mu_tuple, b_HCD, beta_HCD, b_F_weighted, beta_F, plot=False): #, L, a, b, plot=False):
    b_F_weighted = -0.267
    beta_F = 1.617
    '''b_F_weighted = -0.2696
    beta_F = 1.7205'''

    b_F = b_F_weighted / (1. + beta_F)

    (k, mu) = k_mu_tuple

    if plot == False:
        F_HCD = np.loadtxt('/Users/kwame/Simulations/Illustris/snapdir_064/k_h_Mpc_F_HCD_Voigt_small_DLAs_interpolated_CDDF_mean.txt')[:,1]
    elif plot == True:
        k_F_HCD_Voigt = np.loadtxt('/Users/kwame/Simulations/Illustris/snapdir_064/k_h_Mpc_F_HCD_Voigt_small_DLAs_CDDF_mean.txt') #[:,1]
        Voigt_interpolating_function = spp.interp1d(k_F_HCD_Voigt[:1500,0],k_F_HCD_Voigt[:1500,1],kind='cubic')
        F_HCD = np.ones_like(k)
        F_HCD[(k * mu) > k_F_HCD_Voigt[0,0]] = Voigt_interpolating_function((k * mu)[k * mu > k_F_HCD_Voigt[0,0]])

    #F_HCD = 1. / (np.exp((k - 0.25) * L / 0.25) + 1)
    #F_HCD = np.sinc(k * mu * L / mh.pi) #* np.sinc(k * mu * a / mh.pi)) / 2.
    #F_HCD = np.sinc(k * mu * L / mh.pi) #* np.exp((((k * 1.) ** 1.) * (mu ** 1.)) / (-1. * a)) #* np.sinc(k * mu * a / mh.pi) #L = 11.
    ##F_HCD = np.sin(k * mu * L_HCD) / (k * mu * L_HCD)
    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    forest_auto_bias = forest_linear_bias ** 2
    HCD_linear_bias_and_wings = b_HCD * (1. + (beta_HCD * (mu ** 2))) * F_HCD
    HCD_auto_bias = HCD_linear_bias_and_wings ** 2
    forest_HCD_cross_bias = 2. * forest_linear_bias * HCD_linear_bias_and_wings

    return forest_HCD_cross_bias + HCD_auto_bias #- (b * (mu ** 2.) * (k ** 0.2)) #- (b * (k ** 0.5) * (mu ** 2)) #forest_auto_bias +

def forest_HCD_linear_bias_and_Voigt_wings_model_errorbars(k_mu_tuple, b_HCD, beta_HCD, b_F_weighted, beta_F, plot=False):
    b_F_weighted = -0.267
    beta_F = 1.617
    '''b_F_weighted = -0.2696
    beta_F = 1.7205'''

    b_F = b_F_weighted / (1. + beta_F)

    (k, mu) = k_mu_tuple

    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    forest_auto_bias = forest_linear_bias ** 2

    return forest_auto_bias + forest_HCD_linear_bias_and_Voigt_wings_model(k_mu_tuple,b_HCD,beta_HCD,b_F_weighted,beta_F,plot=plot)

def forest_HCD_linear_bias_and_sinc_model(k_mu_tuple, b_HCD, beta_HCD, b_F_weighted, beta_F, L_HCD):
    '''b_F_weighted = -0.267
    beta_F = 1.617'''

    b_F = b_F_weighted / (1. + beta_F)

    (k, mu) = k_mu_tuple

    F_HCD = np.sinc(k * mu * L_HCD / mh.pi)
    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    HCD_linear_bias_and_wings = b_HCD * (1. + (beta_HCD * (mu ** 2))) * F_HCD
    HCD_auto_bias = HCD_linear_bias_and_wings ** 2
    forest_HCD_cross_bias = 2. * forest_linear_bias * HCD_linear_bias_and_wings

    return forest_HCD_cross_bias + HCD_auto_bias

def forest_HCD_linear_bias_and_sinc_model_full(k_mu_tuple, b_HCD, beta_HCD, b_F_weighted, beta_F, L_HCD):
    '''b_F_weighted = -0.267
    beta_F = 1.617'''

    b_F = b_F_weighted / (1. + beta_F)

    (k, mu) = k_mu_tuple

    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    forest_auto_bias = forest_linear_bias ** 2

    return forest_auto_bias + forest_HCD_linear_bias_and_sinc_model(k_mu_tuple,b_HCD,beta_HCD,b_F_weighted,beta_F,L_HCD)

def forest_HCD_linear_bias_and_parametric_wings_model(k_mu_tuple, b_F_weighted, beta_F, b_HCD, a, b, c, plot=False, F_Voigt=None, power_linear=None, slice_array=None):
    '''b_F_weighted = -0.274
    beta_F = 1.671'''
    #b_HCD = -0.008
    beta_HCD = a
    b = 1.e+30
    #c = 0.002

    b_F = b_F_weighted / (1. + beta_F)

    (k, mu) = k_mu_tuple

    if F_Voigt == None:
        if plot == False:
            F_Voigt = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/k_h_Mpc_F_HCD_Voigt_sub_DLAs_interpolated_sim_CDDF_bin_4_6_evenMu.txt')[:,1]
            #F_Voigt = F_Voigt[np.array([True, False, False, True, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False])]
            #F_Voigt = F_Voigt[np.array([True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False])]
            #F_Voigt = F_Voigt[np.array([True,  True, False,  True,  True,  True, False,  True,  True, True, False,  True,  True,  True, False,  True,  True,  True, False])]
            #F_Voigt[(k * mu) < 1.e-2] = 1.
        elif plot == True:
            k_F_HCD_Voigt = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/k_h_Mpc_F_HCD_Voigt_sub_DLAs_sim_CDDF_short.txt')
            Voigt_interpolating_function = spp.interp1d(k_F_HCD_Voigt[:8000:50,0],k_F_HCD_Voigt[:8000:50,1],kind='cubic')
            F_Voigt = np.ones_like(k)
            F_Voigt[(k * mu) > k_F_HCD_Voigt[0,0]] = Voigt_interpolating_function((k * mu)[k * mu > k_F_HCD_Voigt[0,0]])
            #F_Voigt[(k * mu) < 1.e-2] = 1.

    F_HCD = F_Voigt #[slice_array] #* np.exp(k * mu / a) - (b * k * mu)
    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    HCD_linear_bias_and_wings = b_HCD * (1. + (beta_HCD * (mu ** 2))) * F_HCD
    HCD_auto_bias = HCD_linear_bias_and_wings ** 2
    forest_HCD_cross_bias = 2. * forest_linear_bias * HCD_linear_bias_and_wings

    return ((forest_HCD_cross_bias + HCD_auto_bias) * np.exp((k * mu / b) ** 1.)) #- (c * (k ** 1.) * (mu ** 2.)) #* forest_non_linear_function(k, mu)

def forest_HCD_linear_bias_and_parametric_wings_model_full(k_mu_tuple, b_F_weighted, beta_F, b_HCD, a, b, c, plot=False, F_Voigt=None, power_linear=None, slice_array=None):
    '''b_F_weighted = -0.274
    beta_F = 1.671'''
    #b_HCD = -0.008
    beta_HCD = a
    #a = 1.e+30
    #b = 0.002

    b_F = b_F_weighted / (1. + beta_F)

    (k, mu) = k_mu_tuple

    forest_linear_bias = b_F * (1. + (beta_F * (mu ** 2)))
    forest_auto_bias = (forest_linear_bias ** 2) * (forest_non_linear_function(k, mu, power_linear=power_linear) ** 1.)

    return forest_auto_bias + forest_HCD_linear_bias_and_parametric_wings_model(k_mu_tuple,b_F_weighted,beta_F,b_HCD,a,b,c,plot=plot,F_Voigt=F_Voigt)

def forest_non_linear_function(k, mu, power_linear = None): #k in h / Mpc
    k_NL = 6.40 #6.77
    alpha_NL = 0.569 #0.550
    k_P = 15.3 #15.9
    alpha_P = 2.01 #2.12
    k_V0 = 1.220 #0.819
    alpha_V = 1.50
    k_V_prime = 0.923 #0.917
    alpha_V_prime = 0.451 #0.528

    k_V = k_V0 * ((1. + (k / k_V_prime)) ** alpha_V_prime)
    return np.exp(((k / k_NL) ** alpha_NL) - ((k / k_P) ** alpha_P) - ((k * mu / k_V) ** alpha_V))

'''q1 = 0.057
    q2 = 0.368
    a_v = 0.156
    k_v = 0.480 ** (1. / a_v)
    b_v = 1.57
    k_p = 9.2

    delta_squared = (k ** 3.) * power_linear / 2. / (mh.pi ** 2.)
    return np.exp((((q1 * delta_squared) + (q2 * (delta_squared ** 2.))) * (1. - (((k / k_v) ** a_v) * (mu ** b_v)))) - ((k / k_p) ** 2.))'''


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
    power_linear_file = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_3_49_snap57_750_10_4_6_evenMu_k_raw_max_1_pow_k_mu_binned.npz')
    #power_linear_file = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_44_snap64_750_10_4_6_kMax1.npy')
    power_linear = power_linear_file['arr_0']

    n_mu_bins = 4
    line_colours = ['blue','cyan','yellow','brown']
    #line_colours = ['indigo', 'blue', 'cyan', 'turquoise', 'green', 'yellow', 'orange', 'brown']

    k_max = 1. #h / Mpc
    k_min = 0.
    n_realisations = 1
    '''b_HCD_ensemble = [None] * n_realisations
    beta_HCD_ensemble = [None] * n_realisations
    L_HCD_ensemble = [None] * n_realisations
    b_F_weighted_ensemble = [None] * n_realisations
    beta_F_ensemble = [None] * n_realisations'''

    for i in range(n_realisations):
        #power_file_name = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_smallDLAs_forest_64_750_10_4_6_evenMu_kMax_1.00.npz'
        power_file_name = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/power_subDLAs_forest_57_750_10_4_6_evenMu_kMax_1.00.npz'
        #power_file_name_dodged = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_LLS_dodged_64_750_10_4_6_evenMu_kMax_1.00.npz'
        power_file_name_dodged = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_057/power_DLAs_LLS_dodged_57_750_10_4_6_evenMu_kMax_1.00.npz'

        power_file = np.load(power_file_name)
        power_file_dodged = np.load(power_file_name_dodged)

        counts_binned = power_file['arr_2'].flatten()
        power_large_scales = power_file['arr_0'].flatten()[counts_binned > 0.] * (75 ** 3)
        k_large_scales = power_file['arr_1'].flatten()[counts_binned > 0.] / 0.704 #h/Mpc
        mu_large_scales = np.absolute(power_file['arr_3'].flatten()[counts_binned > 0.]) #|mu|

        power_large_scales_dodged = power_file_dodged['arr_0'].flatten()[counts_binned > 0.][k_large_scales <= k_max] * (75 ** 3)
        power_theory_binned = power_linear.flatten()[counts_binned > 0.][k_large_scales <= k_max]

        counts_binned = counts_binned[counts_binned > 0.][k_large_scales <= k_max]
        power_large_scales = power_large_scales[k_large_scales <= k_max]
        mu_large_scales = mu_large_scales[k_large_scales <= k_max]
        k_large_scales = k_large_scales[k_large_scales <= k_max]

        full_mu = cp.deepcopy(mu_large_scales)

        mu_max = 2. #0.75
        counts_binned = counts_binned[mu_large_scales < mu_max]
        power_large_scales = power_large_scales[mu_large_scales < mu_max]
        power_large_scales_dodged = power_large_scales_dodged[mu_large_scales < mu_max]
        power_theory_binned = power_theory_binned[mu_large_scales < mu_max]
        k_large_scales = k_large_scales[mu_large_scales < mu_max]
        mu_large_scales = mu_large_scales[mu_large_scales < mu_max]

        power_ratio = (power_large_scales - 0.) / (power_theory_binned * 1.) #forest_non_linear_function(k_large_scales, mu_large_scales))
        power_ratio_dodged = power_large_scales_dodged / (power_theory_binned * 1.) #forest_non_linear_function(k_large_scales, mu_large_scales))
        power_difference = (power_large_scales - power_large_scales_dodged) / (power_theory_binned * 1.) #forest_non_linear_function(k_large_scales, mu_large_scales))
        power_ratio_errors = (1. / np.sqrt(counts_binned)) + 0.
        power_array = np.vstack((power_ratio_dodged,power_ratio))

        #Sampling
        n_params = 6
        n_walkers = 100
        n_steps = 500
        n_burn_in_steps = 100
        prior_limits = np.array([[-10., 0.], [0., 10.], [-0.02, 0.], [0., 2.], [-1.e-30, 1.e-30], [-1.e-30, 1.e-30]])

        starting_positions = get_starting_positions_in_uniform_prior(prior_limits, n_walkers)
        samples, chains_without_burn_in, sampler = get_posterior_samples(lnlike_joint, lnprior_forest_HCD_linear_bias_and_parametric_wings_model, [k_large_scales, mu_large_scales], power_array, power_ratio_errors, n_params, n_walkers, n_steps, n_burn_in_steps, starting_positions, power_linear=power_theory_binned)
        gelman_rubin_statistic = gelman_rubin_convergence_statistic(chains_without_burn_in)
        print(gelman_rubin_statistic)

        #Plotting
        fig, axes = plt.subplots(n_params, n_params, figsize = (9.5, 9.5)) # - 2
        fig = co.corner(samples, labels = ['b_F (1 + beta_F)', 'beta_F', 'b_HCD', 'beta_HCD', 'k_a (h / Mpc)', 'b (Mpc / h)'], fig=fig) #[:,:-2]

        b_F, beta_F, b_HCD, beta_HCD, k_a, b = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis = 0)))
        print(b_F, beta_F, b_HCD, beta_HCD, k_a, b)

        n_dof = (k_large_scales.size * 2.) - 4. #- 12. #n_params
        #print(-2. * lnlike_forest_HCD_linear_bias_and_parametric_wings_model([b_HCD[0], beta_HCD[0], k_a[0], b[0]], (k_large_scales, mu_large_scales), power_ratio, power_ratio_errors) / n_dof)
        print(-2. * lnlike_joint([b_F[0], beta_F[0], b_HCD[0], beta_HCD[0], k_a[0], b[0]],[k_large_scales, mu_large_scales], power_array, power_ratio_errors, power_linear=power_theory_binned) / n_dof)
        #print(-2. * lnlike_forest_linear_bias_model([b_F[0], beta_F[0]], (k_large_scales, mu_large_scales), power_ratio_dodged, power_ratio_errors) / n_dof)

        print(np.sum(counts_binned))
        plt.show()

    #Plot dodged data space
    '''plt.figure()
    cmap = plt.cm.jet
    #bounds = np.linspace(0,1,n_mu_bins+1)
    bounds = np.array([0., 0.5, 0.8, 0.95, 1.])
    norm = mpc.BoundaryNorm(bounds, cmap.N)
    k_plot = np.linspace(np.min(k_large_scales),np.max(k_large_scales),1000.)

    for i in range(bounds.shape[0] - 1):
        mu_plot = np.mean(mu_large_scales[(mu_large_scales >= bounds[i]) * (mu_large_scales < bounds[i+1])])
        if i == bounds.shape[0] - 2:
            mu_plot = np.mean(mu_large_scales[mu_large_scales >= bounds[i]])
        print(mu_plot)

        model_samples = np.zeros((samples.shape[0],k_plot.shape[0]))
        for j in range(samples.shape[0]):
            model_samples[j] = forest_linear_bias_model((k_plot, mu_plot),samples[j,0],samples[j,1])
        model_percentiles = np.percentile(model_samples, [16, 50, 84], axis=0)

        plt.plot(k_plot, forest_linear_bias_model((k_plot, mu_plot), b_F[0], beta_F[0]) * np.ones_like(k_plot), ls='--', color=line_colours[i], lw=2.)
        plt.plot(k_plot, model_percentiles[2], ls=':', color=line_colours[i], lw=0.5)
        plt.plot(k_plot, model_percentiles[0], ls='-.', color=line_colours[i], lw=0.5)

    plt.errorbar(k_large_scales, power_ratio_dodged, yerr=power_ratio_errors*power_ratio_dodged*mh.sqrt(2.), ecolor='gray', ls='')
    plt.scatter(k_large_scales, power_ratio_dodged, c=mu_large_scales, cmap=cmap, norm=norm, s=100.)
    plt.colorbar()

    plt.xscale('log')
    plt.xlim([8.e-2, 1.])
    plt.ylim([-0.01, 0.09])

    plt.show()

    #Plot undodged data space
    plt.figure()'''
    cmap = plt.cm.jet
    bounds = np.linspace(0,1,n_mu_bins+1)
    #bounds = np.array([0., 0.5, 0.8, 0.95, 1.])
    norm = mpc.BoundaryNorm(bounds, cmap.N)
    #line_colours = ['blue','cyan','yellow','brown']
    k_plot = np.linspace(np.min(k_large_scales),np.max(k_large_scales),1000.)

    #Generate F_Voigt
    F_Voigt = [None] * (bounds.shape[0] - 1)
    for i in range(bounds.shape[0] - 1):
        mu_plot = np.mean(mu_large_scales[(mu_large_scales >= bounds[i]) * (mu_large_scales < bounds[i+1])])
        if i == bounds.shape[0] - 2:
            mu_plot = np.mean(mu_large_scales[mu_large_scales >= bounds[i]])
        print(mu_plot)

        k_F_HCD_Voigt = np.loadtxt('/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/k_h_Mpc_F_HCD_Voigt_sub_DLAs_sim_CDDF_short.txt')
        Voigt_interpolating_function = spp.interp1d(k_F_HCD_Voigt[:8000:50, 0], k_F_HCD_Voigt[:8000:50, 1], kind='cubic')
        F_Voigt[i] = np.ones_like(k_plot)
        F_Voigt[i][(k_plot * mu_plot) > k_F_HCD_Voigt[0, 0]] = Voigt_interpolating_function((k_plot * mu_plot)[k_plot * mu_plot > k_F_HCD_Voigt[0, 0]])
        #F_Voigt[i][(k_plot * mu_plot) < 1.e-2] = 1.

    '''for i in range(bounds.shape[0] - 1):
        mu_plot = np.mean(mu_large_scales[(mu_large_scales >= bounds[i]) * (mu_large_scales < bounds[i+1])])
        if i == bounds.shape[0] - 2:
            mu_plot = np.mean(mu_large_scales[mu_large_scales >= bounds[i]])
        print(mu_plot)

        model_samples = np.zeros((samples.shape[0],k_plot.shape[0]))
        for j in range(samples.shape[0]):
            model_samples[j] = forest_HCD_linear_bias_and_parametric_wings_model_full((k_plot, mu_plot),samples[j,0],samples[j,1],samples[j,2],samples[j,3],samples[j,4],samples[j,5], plot=True, F_Voigt=F_Voigt[i])
        model_percentiles = np.percentile(model_samples, [16, 50, 84], axis=0)

        plt.plot(k_plot, forest_HCD_linear_bias_and_parametric_wings_model_full((k_plot, mu_plot), b_F[0], beta_F[0], b_HCD[0], beta_HCD[0], k_a[0], b[0], plot=True, F_Voigt=F_Voigt[i]) * np.ones_like(k_plot), ls='--', color=line_colours[i], lw=2.)
        plt.plot(k_plot, model_percentiles[2], ls=':', color=line_colours[i], lw=0.5)
        plt.plot(k_plot, model_percentiles[0], ls='-.', color=line_colours[i], lw=0.5)

    plt.errorbar(k_large_scales, power_ratio, yerr=power_ratio_errors*power_ratio*mh.sqrt(2.), ecolor='gray', ls='')
    plt.scatter(k_large_scales, power_ratio, c=mu_large_scales, cmap=cmap, norm=norm, s=100.)
    plt.colorbar()

    plt.xscale('log')
    plt.xlim([8.e-2, 1.])
    plt.ylim([-0.01, 0.09])

    plt.show()'''


    #Plot difference data space
    plt.figure()
    cmap = plt.cm.jet
    #bounds = np.linspace(0,1,n_mu_bins+1)
    #bounds = np.array([0., 0.5, 0.8, 0.95, 1.])
    norm = mpc.BoundaryNorm(bounds, cmap.N)
    #line_colours = ['blue','cyan','yellow','brown']
    k_plot = np.linspace(np.min(k_large_scales),np.max(k_large_scales),1000.)

    for i in range(bounds.shape[0] - 1):
        mu_plot = np.mean(mu_large_scales[(mu_large_scales >= bounds[i]) * (mu_large_scales < bounds[i+1])])
        #print(mu_large_scales[(mu_large_scales >= bounds[i]) * (mu_large_scales < bounds[i+1])])
        if i == bounds.shape[0] - 2:
            mu_plot = np.mean(mu_large_scales[mu_large_scales >= bounds[i]])
            print(mu_large_scales[mu_large_scales >= bounds[i]])
        print(mu_plot)
        '''plt.plot(k_plot, forest_linear_bias_model((k_plot,mu_plot),b_HCD[0],beta_HCD[0]) * np.ones_like(k_plot), ls='--', color=line_colours[i], lw=2.) #,b_F[0],beta_F[0],L_HCD[0],a[0],b[0] #,plot=True
        plt.plot(k_plot, forest_linear_bias_model((k_plot, mu_plot), b_HCD[0] + b_HCD[1], beta_HCD[0] + beta_HCD[1]) * np.ones_like(k_plot), ls=':', color=line_colours[i], lw=1.)  # ,b_F[0],beta_F[0],L_HCD[0],a[0],b[0] #,plot=True
        plt.plot(k_plot, forest_linear_bias_model((k_plot, mu_plot), b_HCD[0] - b_HCD[2], beta_HCD[0] - beta_HCD[2]) * np.ones_like(k_plot), ls='-.', color=line_colours[i], lw=1.)'''

        '''plt.plot(k_plot, forest_HCD_linear_bias_and_Voigt_wings_model((k_plot, mu_plot), b_HCD[0], beta_HCD[0], b_F[0], beta_F[0], plot=True) * np.ones_like(k_plot), ls='--', color=line_colours[i], lw=2.)  # ,b_F[0],beta_F[0],L_HCD[0],a[0],b[0] #,plot=True
        plt.plot(k_plot, forest_HCD_linear_bias_and_Voigt_wings_model((k_plot, mu_plot), b_HCD[0] + b_HCD[1], beta_HCD[0] + beta_HCD[1], b_F[0] + b_F[1], beta_F[0] + beta_F[1], plot=True) * np.ones_like(k_plot), ls=':', color=line_colours[i], lw=1.)  # ,b_F[0],beta_F[0],L_HCD[0],a[0],b[0] #,plot=True
        plt.plot(k_plot, forest_HCD_linear_bias_and_Voigt_wings_model((k_plot, mu_plot), b_HCD[0] - b_HCD[2], beta_HCD[0] - beta_HCD[2], b_F[0] - b_F[2], beta_F[0] - beta_F[2], plot=True) * np.ones_like(k_plot), ls='-.', color=line_colours[i], lw=1.)
        '''

        '''plt.plot(k_plot, forest_HCD_linear_bias_and_sinc_model((k_plot, mu_plot), b_HCD[0], beta_HCD[0], b_F[0], beta_F[0], L_HCD[0]) * np.ones_like(k_plot), ls='--', color=line_colours[i], lw=2.)
        plt.plot(k_plot, forest_HCD_linear_bias_and_sinc_model((k_plot, mu_plot), b_HCD[0] + b_HCD[1], beta_HCD[0] + beta_HCD[1], b_F[0] + b_F[1], beta_F[0] + beta_F[1], L_HCD[0] + L_HCD[1]) * np.ones_like(k_plot), ls=':', color=line_colours[i], lw=1.)
        plt.plot(k_plot, forest_HCD_linear_bias_and_sinc_model((k_plot, mu_plot), b_HCD[0] - b_HCD[2], beta_HCD[0] - beta_HCD[2], b_F[0] - b_F[2], beta_F[0] - beta_F[2], L_HCD[0] - L_HCD[2]) * np.ones_like(k_plot), ls='-.', color=line_colours[i], lw=1.)
        '''

        model_samples = np.zeros((samples.shape[0],k_plot.shape[0]))
        for j in range(samples.shape[0]):
            model_samples[j] = forest_HCD_linear_bias_and_parametric_wings_model((k_plot, mu_plot),samples[j,0],samples[j,1],samples[j,2],samples[j,3],samples[j,4],samples[j,5], plot=True, F_Voigt=F_Voigt[i])
        model_percentiles = np.percentile(model_samples, [16, 50, 84], axis=0)

        plt.plot(k_plot, forest_HCD_linear_bias_and_parametric_wings_model((k_plot, mu_plot), b_F[0], beta_F[0], b_HCD[0], beta_HCD[0], k_a[0], b[0], plot=True, F_Voigt=F_Voigt[i]) * np.ones_like(k_plot), ls='--', color=line_colours[i], lw=2.)
        #plt.scatter(k_large_scales,forest_HCD_linear_bias_and_parametric_wings_model((k_large_scales, mu_large_scales), b_F[0], beta_F[0], b_HCD[0],beta_HCD[0], k_a[0], b[0], plot=False) * np.ones_like(k_large_scales), color=line_colours[-1])

        plt.plot(k_plot, model_percentiles[2], ls=':', color=line_colours[i], lw=0.5)
        plt.plot(k_plot, model_percentiles[0], ls='-.', color=line_colours[i], lw=0.5)

        #plt.plot(k_plot, forest_HCD_linear_bias_and_sinc_model((k_plot, mu_plot), b_HCD[0], beta_HCD[0], b_F[0], beta_F[0], 24.341), ls='-', color=line_colours[i], lw=1.)

    plt.scatter(k_large_scales, power_difference, c=mu_large_scales, cmap=cmap, norm=norm, s=100.)
    plt.errorbar(k_large_scales, power_difference, yerr=power_ratio_errors * power_difference * mh.sqrt(2.), ecolor='gray', ls='')
    plt.colorbar()

    plt.xscale('log')
    plt.xlim([8.e-2, 1.])
    #plt.ylim([-0.0001, 0.0006])  #LLS
    #plt.ylim([-0.0005, 0.0025]) #Sub-DLAs
    #plt.ylim([-0.002, 0.005]) #Small DLAs
    #plt.ylim([-0.003, 0.005])  # Large DLAs
    #plt.ylim([-0.001, 0.006]) #Residual contamination
    #plt.ylim([-0.005, 0.011])  # Total contamination

    #z = 3.49
    plt.ylim([-0.003, 0.009]) #Sub-DLAs

    plt.axhline(y = 0., color = 'black', lw = 0.5, ls = ':')
    #plt.axvline(x = 0.25, color = 'black', lw = 0.5, ls = ':')

    plt.show()