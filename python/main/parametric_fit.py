import numpy as np
import math as mh
import emcee as mc
import corner as co
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import numpy.random as npr
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

def fit_two_independent_variable_model(x0, x1, y, model_function, initial_param_values = None, y_sigma = None, param_bounds = (-np.inf, np.inf)):
    return spo.curve_fit(model_function, (x0, x1), y, p0 = initial_param_values, sigma = y_sigma, bounds = param_bounds, method = None) #'lm')


#Bayesian inference
#Likelihoods
def lnlike_forest_linear_bias_model(param_array, x, y, yerr):
    model_evaluation = forest_linear_bias_model(x, param_array[0], param_array[1])
    return -0.5 * np.sum(((y - model_evaluation)**2) / ((yerr * model_evaluation * mh.sqrt(2.))**2)) #model_evaluation

def lnlike_forest_HCD_linear_bias_and_wings_model(param_array, x, y, yerr):
    model_evaluation = forest_HCD_linear_bias_and_wings_model(x, param_array[0], param_array[1], param_array[2])
    return -0.5 * np.sum(((y - model_evaluation)**2) / (np.mean(yerr * model_evaluation * mh.sqrt(2.))**2)) #model_evaluation

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

def get_starting_positions_in_uniform_prior(prior_limits, n_walkers):
    n_params = prior_limits.shape[0]
    return npr.uniform(low = prior_limits[:,0], high = prior_limits[:,1], size = (n_walkers, n_params))

#Posteriors
def lnprob(param_array, x, y, yerr, lnlike, lnprior):
    lnprior_evaluation = lnprior(param_array)
    if not np.isfinite(lnprior_evaluation):
        return -np.inf
    else:
        return lnprior_evaluation + lnlike(param_array, x, y, yerr)

#Sampling
def gelman_rubin_convergence_statistic(mcmc_chains): #dims: Walkers * Steps * Parameters
    n_walkers = mcmc_chains.shape[0]
    n_steps = mcmc_chains.shape[1]

    within_chain_variance = np.mean(np.var(mcmc_chains, axis = 1, ddof = 1), axis = 0) #dims: Parameters

    chain_means = np.mean(mcmc_chains, axis = 1)
    between_chain_variance = np.var(chain_means, axis = 0, ddof = 1) * n_steps

    posterior_marginal_variance = ((n_steps - 1) * within_chain_variance / n_steps) + ((n_walkers + 1) * between_chain_variance / n_steps / n_walkers)
    return np.sqrt(posterior_marginal_variance / within_chain_variance)

def get_posterior_samples(lnlike, lnprior, x, y, yerr, n_params, n_walkers, n_steps, n_burn_in_steps, starting_positions):
    #starting_positions = [[-0.325, 1.663] + 1.e-4 * np.random.randn(n_params) for i in range(n_walkers)]
    sampler = mc.EnsembleSampler(n_walkers, n_params, lnprob, args = (x, y, yerr, lnlike, lnprior))
    sampler.run_mcmc(starting_positions, n_steps)
    return sampler.chain[:, n_burn_in_steps:, :].reshape((-1, n_params)), sampler.chain[:, n_burn_in_steps:, :] #, mc.autocorr.integrated_time(sampler.chain,axis=1), sampler.chain

#Models
def forest_linear_bias_model(k_mu_tuple, b_F_weighted, beta_F):
    (k, mu) = k_mu_tuple
    b_F = b_F_weighted / (1. + beta_F)
    return (b_F * (1. + (beta_F * (mu ** 2)))) ** 2

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
    power_linear_file = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_44_snap64_750_10_8_6_k_raw_max_1_pow_k_mu_binned.npz')
    #power_linear_file = np.load('/Users/keir/Software/lyman-alpha/python/test/P_k_z_2_44_snap64_750_10_4_6_newMuK.npy')
    power_linear = power_linear_file['arr_0']

    k_max = 1. #h / Mpc
    k_min = 0.
    n_realisations = 1
    '''b_F_weighted_ensemble = [None] * n_realisations
    beta_F_ensemble = [None] * n_realisations'''

    for i in range(n_realisations):
        power_file_name = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_dodged_64_750_10_8_6_kMax_1.00.npz'
        #power_file_name = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_undodged_64_750_10_4_6_newMu_newK.npz'
        power_file_name_dodged = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_LLS_dodged_64_750_10_8_6_kMax_1.00.npz' #%(i+1)
        #power_file_name_dodged = '/Users/keir/Documents/lyman_alpha/simulations/illustris_big_box_spectra/snapdir_064/power_DLAs_LLS_dodged_64_750_10_4_6_newMu_newK.npz'

        power_file = np.load(power_file_name)
        power_file_dodged = np.load(power_file_name_dodged)

        '''power_box = power_file['arr_0'] * (75. ** 3) #(Mpc/h)^3
        k_box = power_file['arr_1'] / 0.704 #h/Mpc
        mu_box = np.absolute(power_file['arr_2']) #|mu|
    
        power_large_scales = power_box[k_box < k_max][1:] #Remove k = 0
        k_large_scales = k_box[k_box < k_max][1:]
        mu_large_scales = mu_box[k_box < k_max][1:]
    
        power_box_dodged = power_file_dodged['arr_0'] * (75. ** 3) #(Mpc/h)^3
        power_large_scales_dodged = power_box_dodged[k_box < k_max][1:] #Remove k = 0'''
        '''k_max_new = 0.63 #h / Mpc
        mu_large_scales = mu_large_scales[k_large_scales < k_max_new]
        power_large_scales_dodged = power_large_scales_dodged[k_large_scales < k_max_new]
        power_linear = power_linear[k_large_scales < k_max_new]
        k_large_scales = k_large_scales[k_large_scales < k_max_new]'''

        counts_binned = power_file['arr_2'].flatten()
        power_large_scales = power_file['arr_0'].flatten()[counts_binned > 0.] * (75 ** 3) #* (21. ** 6) #(Mpc/h)^3
        k_large_scales = power_file['arr_1'].flatten()[counts_binned > 0.] / 0.704 #h/Mpc
        mu_large_scales = np.absolute(power_file['arr_3'].flatten()[counts_binned > 0.]) #|mu|

        power_large_scales_dodged = power_file_dodged['arr_0'].flatten()[counts_binned > 0.][k_large_scales <= k_max] * (75 ** 3) #* (21. ** 6)  # (Mpc/h)^3
        power_theory_binned = power_linear.flatten()[counts_binned > 0.][k_large_scales <= k_max]
        #power_theory_binned = power_linear[k_large_scales <= k_max]

        counts_binned = counts_binned[counts_binned > 0.][k_large_scales <= k_max]
        power_large_scales = power_large_scales[k_large_scales <= k_max]
        mu_large_scales = mu_large_scales[k_large_scales <= k_max]
        k_large_scales = k_large_scales[k_large_scales <= k_max]

        '''counts_binned = counts_binned[k_large_scales >= k_min]
        power_large_scales = power_large_scales[k_large_scales >= k_min]
        power_large_scales_dodged = power_large_scales_dodged[k_large_scales >= k_min]
        power_theory_binned = power_theory_binned[k_large_scales >= k_min]
        mu_large_scales = mu_large_scales[k_large_scales >= k_min]
        #power_linear = power_linear[k_large_scales >= k_min]
        k_large_scales = k_large_scales[k_large_scales >= k_min]'''
        '''k_max2 = 0.7
        counts_binned = counts_binned[k_large_scales <= k_max2]
        power_large_scales = power_large_scales[k_large_scales <= k_max2]
        power_large_scales_dodged = power_large_scales_dodged[k_large_scales <= k_max2]
        mu_large_scales = mu_large_scales[k_large_scales <= k_max2]
        power_linear = power_linear[k_large_scales <= k_max2]
        k_large_scales = k_large_scales[k_large_scales <= k_max2]'''

        power_ratio = (power_large_scales - power_large_scales_dodged) / (power_theory_binned * forest_non_linear_function(k_large_scales, mu_large_scales)) #((k_large_scales / (0.5 / 0.704)) ** -3) #1. #(power_linear * 1.) #forest_non_linear_function(k_large_scales, mu_large_scales))
        power_ratio_errors = 1. / np.sqrt(counts_binned) #+ 0.013
        '''power_ratio = power_file_dodged['arr_0'].flatten()[1:] * (21**6) #/ 2.
        k_large_scales = power_file_dodged['arr_1'].flatten()[1:] / 0.704
        mu_large_scales = power_file_dodged['arr_2'].flatten()[1:]
        counts_binned = np.ones_like(k_large_scales)
        power_ratio_errors = np.ones_like(k_large_scales)'''

        '''param_array, param_covar = fit_two_independent_variable_model(k_large_scales, mu_large_scales, power_ratio, fitting_model, initial_param_values=initial_param_values, y_sigma=None, param_bounds=param_bounds)
        print(param_array)
        print(param_covar)
        print(np.sqrt(np.diag(param_covar)))'''

        #Sampling
        n_params = 3
        n_walkers = 100
        n_steps = 1000
        n_burn_in_steps = 200
        prior_limits = np.array([[-0.2, 0.], [0., 1.6], [0., 70.]])
        '''b_F_weighted_true = -0.325
        beta_F_true = 1.663'''

        #starting_positions = [[-0.0195, 0.31, 24.341] + 1.e-4 * np.random.randn(n_params) for i in range(n_walkers)]
        starting_positions = get_starting_positions_in_uniform_prior(prior_limits, n_walkers)
        samples, chains_without_burn_in = get_posterior_samples(lnlike_forest_HCD_linear_bias_and_wings_model, lnprior_forest_HCD_linear_bias_and_wings_model, (k_large_scales, mu_large_scales), power_ratio, power_ratio_errors, n_params, n_walkers, n_steps, n_burn_in_steps, starting_positions)
        gelman_rubin_statistic = gelman_rubin_convergence_statistic(chains_without_burn_in)
        print(gelman_rubin_statistic)

        #Plotting
        fig = co.corner(samples, labels = ['b_HCD', 'beta_HCD', 'L_HCD (Mpc / h)']) #, truths = [b_F_weighted_true,beta_F_true]) #[-0.325,1.663]

        b_HCD, beta_HCD, L_HCD = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis = 0)))
        print(b_HCD, beta_HCD, L_HCD)

        n_dof = k_large_scales.size - n_params
        print(-2. * lnlike_forest_HCD_linear_bias_and_wings_model([b_HCD[0], beta_HCD[0], L_HCD[0]], (k_large_scales, mu_large_scales), power_ratio, power_ratio_errors) / n_dof)

        print(np.sum(counts_binned))
        plt.show()
        '''b_F_weighted_ensemble = b_HCD[0]
        beta_F_ensemble = beta_HCD[0]'''

    #Data space plot
    '''plt.figure()
    for b_HCD_samples, beta_HCD_samples, L_HCD_samples in samples[npr.randint(len(samples), size = 100)]:
        plt.scatter(k_large_scales, forest_HCD_linear_bias_and_wings_model((k_large_scales,mu_large_scales), b_HCD_samples, beta_HCD_samples, L_HCD_samples), c=mu_large_scales, cmap='Reds', edgecolors='red') #alpha=0.1)
    #plt.scatter(k_large_scales, forest_linear_bias_model((k_large_scales,mu_large_scales),b_F_weighted_true,beta_F_true), color='blue', alpha=0.8)
    plt.errorbar(k_large_scales, power_ratio, yerr=power_ratio_errors*power_ratio, ecolor='gray', ls='')
    plt.scatter(k_large_scales, power_ratio, c=mu_large_scales, cmap='gray', edgecolors='black')
    plt.colorbar()
    plt.show()'''

    plt.figure()
    cmap = plt.cm.jet
    '''cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)'''
    bounds = np.linspace(0,1,9)
    #bounds = np.array([0.,0.5,0.8,0.95,1.])
    norm = mpc.BoundaryNorm(bounds, cmap.N)
    #line_colours = ['blue','cyan','yellow','brown']
    k_plot = np.linspace(0.08,1.,1000)
    #mu_plot = np.array([0.25,0.65,0.875,0.975])

    for i in range(bounds.shape[0] - 1):
        mu_plot = np.mean(mu_large_scales[(mu_large_scales >= bounds[i]) * (mu_large_scales < bounds[i+1])])
        if i == bounds.shape[0] - 2:
            mu_plot = np.mean(mu_large_scales[mu_large_scales >= bounds[i]])
        print(mu_plot)
        plt.plot(k_plot, forest_HCD_linear_bias_and_wings_model((k_plot,mu_plot),b_HCD[0],beta_HCD[0],L_HCD[0]), ls='--') #c=line_colours[i]
        plt.plot(k_plot, forest_HCD_linear_bias_and_wings_model((k_plot, mu_plot), b_HCD[0] + b_HCD[1], beta_HCD[0] + beta_HCD[1], L_HCD[0] + L_HCD[1]), ls=':', lw=0.5) #c=line_colours[i]
        plt.plot(k_plot, forest_HCD_linear_bias_and_wings_model((k_plot, mu_plot), b_HCD[0] - b_HCD[2], beta_HCD[0] - beta_HCD[2], L_HCD[0] - L_HCD[2]), ls='-.', lw=0.5) #c=line_colours[i]
    #plt.errorbar(k_large_scales, power_ratio, yerr=power_ratio_errors*power_ratio*mh.sqrt(2), c=mu_large_scales, cmap=cmap, norm=norm, ls='')
    plt.scatter(k_large_scales, power_ratio, c=mu_large_scales, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.show()

    '''z = np.array([2.0, 2.44, 3.49, 4.43])  # shape Nz
    kpar = ? np.linspace(?)  # shape Nk
    #  the data you want to fit, Pk_ratio_data, is a 2D array, function of (z, k)

    Pk_ratio_data ='''