import numpy as np 
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy import linalg
import networkx as nx

np.random.seed(1234)
TOL = 1e-6

def diffusion_operator_normalized_data(data, dis_mat, affinity = None, sigma = 2, 
                                    num_ev = 20, if_full_spec = False):
    
    if dis_mat is None:
        dis_mat = pairwise_distances(data)

    if affinity is None:
        Gaussian_kernel = np.exp(-dis_mat / sigma)
    else:
        print('there is affinity matrix')
        Gaussian_kernel = affinity

    d = 1 / np.sum(Gaussian_kernel, axis=1)
    K = np.diag(d) @ Gaussian_kernel @ np.diag(d)

    d_ = np.diag(1 / np.sqrt(np.sum(K, axis=1)))
    d__ = np.diag(np.sqrt(np.sum(K, axis=1)))
    M = d_ @ K @ d_
    
    if if_full_spec:
        ev, evv = linalg.eigh(M)
    else:
        ev, evv = eigsh(M, k=num_ev, which='LM')

    evv = evv[:, np.argsort(ev)[::-1]]
    ev = (np.sort(ev)[::-1])
    ev = np.where(ev>TOL, ev, TOL)

    left_evv  = d_ @ evv
    right_evv = evv.T @ d__

    return ev, left_evv, right_evv


def diffusion_operator_graph(affinity, num_ev = 20, if_normalized = False, if_full_spec = False):
    
    D = np.diag(affinity.sum(axis = 1))
    L = D - affinity

    if if_normalized:
        d_ = np.diag(1 / np.sqrt(np.sum(affinity, axis=1)))
        L  = d_ @ L @ d_
    
    if if_full_spec:
        ev, evv = linalg.eigh(L)
    else:
        ev, evv = eigsh(L, k=num_ev, which='SM')

    ev = np.exp(-ev)        
    left_evv  = evv 
    right_evv = evv 

    return ev, left_evv, right_evv