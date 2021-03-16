#MPC formulation

import numpy as np
from numba import njit
import random
from scipy.integrate import solve_ivp, simps
from scipy.optimize import differential_evolution, Bounds, minimize
from scipy.stats import truncnorm
from matplotlib import pyplot as plt

def get_new_U_step():
    #Perform optimization for t=[t; TF], obtaining new U
    #-> call for optimize_U function
    #-> return U_step for implementation
    pass

def optimize_U():
    #Perform optimization for U, for t=[t; TF] obtaining new U
    #-> return U
    pass

def implement_U_step():
    #Implement U_step received from get_new_U_step
    #-> iterate t
    pass

def deterministic_MPC():
    #Main function for MPC
    pass