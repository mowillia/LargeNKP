import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, fmin
from largeN_algo.potential_functions import GN_zero_one
import math


# defining function to produce X_i 
def X_thresh(x_vec, threshold):
    
    """
    Converts a list of floating point values to the next highest
    integer if they are less than `threshold` away from that integer
    
    Parameters
    ----------
    xvec : array
        Vector of floating point values

    Returns
    ----------
    x_int : array
        Vector of integer values
    """    
    
    nvals = len(x_vec)
    
    x_int = np.zeros(nvals) # discrete values
    
    for k in range(nvals):
        
        x_int[k] = int(math.ceil(x_vec[k]-threshold))
        
    return x_int

# defining function to produce average x
def X_avg_zero_one(z, weights, values, T):
    
    """
    For the zero-one KP
    Computes the average occupancy for each object given 
    the solution to the steepest descent condition, model parameters
    (weights, values) and model hyperparameter (T)
    
    Parameters
    ----------
    z : float
        Solution to steepest descent condition
        
    weights : array
        Vector of weights for objects
        
    values : array
        Vector of values for objects
        
    T : float
        Temperature for statistical physics system

    Returns
    ----------
    xvec : array
        Vector of average occupancies
    """    
    
    nvals = len(weights)
    
    # empty vector 
    xvec = np.zeros(nvals) 
    
    # filling in averages
    for k in range(nvals):
        
        xvec[k] = np.exp(values[k]/T)/(z**(-weights[k]) + np.exp(values[k]/T))
        
    return xvec

# consolidating algorithm
def zero_one_algorithm(weights, values, limit, T = 1.0, threshold = 0.6):
    
    """
    Full algorithm for the zero-one KP
    
    Parameters
    ----------

    weights : array
        Vector of weights for objects
        
    values : array
        Vector of values for objects
        
    limit : int
        Weight limit
        
    override : bool
        Defines whether to override the default parameters

    T : float
        Temperature for statistical physics system
        
    threshold : float
        Limit for rounding to next highest integer


    Returns
    ----------
    x_soln : array
        Vector of occupancies
    
    """    
    
    
    # number of weights must be the same as the number of values
    assert len(values) == len(weights)
    
    # solving for z0
    z0 = fsolve(GN_zero_one, x0=0.095, args = (weights, values, limit, T))[0]
    
    # solving for averages
    x_avgs = X_avg_zero_one(z = z0, weights = weights, values = values, T= T)
    
    # thresholding averages
    x_soln = X_thresh(x_avgs, threshold = threshold)
    
    return x_soln


# consolidating algorithm
def limit_ratio(weights, values, limit, T = 1.0, threshold = 0.6):
    
    """
    Computing limit ratio for weights and values
    
    Parameters
    ----------

    weights : array
        Vector of weights for objects
        
    values : array
        Vector of values for objects
        
    limit : int
        Weight limit
        
    override : bool
        Defines whether to override the default parameters

    T : float
        Temperature for statistical physics system
        
    threshold : float
        Limit for rounding to next highest integer


    Returns
    ----------
    ratio : float
        Computed minimum value/weight ratio for item to be included 
        in knapsack
    
    """    
    
    
    # number of weights must be the same as the number of values
    assert len(values) == len(weights)
    
    # solving for z0
    z0 = fsolve(GN_zero_one, x0=0.095, args = (weights, values, limit, T))[0]
        
    # ratios when threshold isn't 1/2
    ratios = - T*np.log(z0)+ T* weights*np.log((1-threshold)/threshold)

    return ratios