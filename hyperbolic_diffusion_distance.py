import numpy as np 
from diffusion_operator_util import *
from sklearn.metrics import pairwise_distances


TOL = 1e-6


def hyperbolic_diffusion(ev, left_evv, right_evv, K):

    X_hat, hdd = [], [], []
    single_mat = []
    time_step = 1/np.power(2, np.arange(K))
    weight    = 2/np.power(2, np.arange(K)/2)

    for ii in range(len(time_step)):
        evv_power = np.power(ev, time_step[ii])

        x_hat = (left_evv @ np.diag(evv_power) @ right_evv)
        x_hat = np.sqrt(np.where(x_hat>TOL, x_hat, TOL))
   
        single_mat.append((2 * np.arcsinh( weight[ii] * pairwise_distances(x_hat))))
        
        tmp = np.concatenate((x_hat, (1/(2*weight[ii])) * np.ones((left_evv.shape[0], 1))), axis = 1)
        X_hat.append(tmp)

    hdd = np.sum(single_mat, axis = 0)
    
    return X_hat, hdd
