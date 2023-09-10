import numpy as np 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from LISA_utils import FFT, waveform

def llike(data_f, signal_f, variance_noise_f):
    """
    Computes log likelihood 
    Assumption: Known PSD otherwise need additional term
    Inputs:
    data in frequency domain 
    Proposed signal in frequency domain
    Variance of noise
    """
    inn_prod = sum((abs(data_f - signal_f)**2) / variance_noise_f)
    # print(inn_prod)
    return(-0.5 * inn_prod)


def lprior_uniform(param,param_low_val,param_high_val):
    """
    Set uniform priors on parameters with select ranges.
    """
    if param < param_low_val or param > param_high_val:
        return -np.inf
    else:
        return 0

def lpost(data_f,signal_f, variance_noise_f,param, param_low_range = -10,param_high_range = 10):
    '''
    Compute log posterior - require log likelihood and log prior.
    '''
    return(lprior_uniform(param,param_low_range,param_high_range) + llike(data_f,signal_f,variance_noise_f))


def accept_reject(lp_prop, lp_prev):
    '''
    Compute log acceptance probability (minimum of 0 and log acceptance rate)
    Decide whether to accept (1) or reject (0)
    '''
    u = np.random.uniform(size = 1)  # U[0, 1]
    logalpha = np.minimum(0, lp_prop - lp_prev)  # log acceptance probability
    if np.log(u) < logalpha:
        return(1)  # Accept
    else:
        return(0)  # Reject

def MCMC_run(data_f,t,eps, variance_noise_f,
                   Ntotal, burnin, fdot_start, printerval, save_interval,fdot_var_prop):
    '''
    Metropolis MCMC sampler
    '''
    
    # plot_direc = os.getcwd() + "/plots"
    # Set starting values

    fdot_chain = [fdot_start]

    # Initial signal
    
    signal_init_t = waveform(fdot_chain[0],t,eps)   # Initial time domain signal
    signal_init_f = FFT(signal_init_t)  # Intial frequency domain signal


                                            
    # Initial value for log posterior
    lp = []
    lp.append(lpost(data_f, signal_init_f, variance_noise_f,fdot_chain[0]))  # Append first value of log posterior
    
    lp_store = lp[0]  # Create log posterior storage to be overwritten
                 
    #####                                                  
    # Run MCMC
    #####
    accept_reject_count = [1]

    for i in range(1, Ntotal):
        
        # if i % printerval == 0: # Print accept/reject ratio.
        #     accept_reject_ratio = sum(accept_reject_count)/len(accept_reject_count)
        #     # tqdm.write("Iteration {0}, accept_reject = {1}".format(i,accept_reject_ratio))

        lp_prev = lp_store  # Call previous stored log posterior
        
        # Propose new points according to a normal proposal distribution of fixed variance 
        fdot_prop = fdot_chain[i - 1] + np.random.normal(0, np.sqrt(fdot_var_prop))

        # Propose a new signal      
        signal_prop_t = waveform(fdot_prop,t,eps)
        signal_prop_f = FFT(signal_prop_t)

        
        # Compute log posterior
        lp_prop = lpost(data_f,signal_prop_f, variance_noise_f,fdot_prop)
        
        ####
        # Perform accept_reject call
        ####
        # breakpoint()
        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            fdot_chain.append(fdot_prop)    # accept \dot{f}_{prop} as new sample
            accept_reject_count.append(1)
            lp_store = lp_prop  # Overwrite lp_store
        else:  # Reject, if this is the case we use previously accepted values
            fdot_chain.append(fdot_chain[i - 1])
            accept_reject_count.append(0)

        lp.append(lp_store)
    
    # Recast as .nparrays
    fdot_chain = np.array(fdot_chain)

    
    return fdot_chain,lp  # Return chains and log posterior.