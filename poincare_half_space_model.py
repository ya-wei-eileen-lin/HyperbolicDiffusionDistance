import numpy as np 
from numpy import linalg as LA

def hyp_dis(x,y):
    return 2 * np.arcsinh(LA.norm(x-y)/(2*np.sqrt(x[-1]*y[-1])))

