import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, fmin
from largeN_algo.potential_functions import GN_bounded
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

def X_avg_bounded(z, weights, values, bounds, T):
    
    """
    For the bounded KP
    Computes the average occupancy for each object given 
    the solution to the steepest descent condition, model parameters
    (weights, values, C) and model hyperparameter (T)
    
    Parameters
    ----------
    z : float
        Solution to steepest descent condition
        
    weights : array
        Vector of weights for objects
        
    values : array
        Vector of values for objects
        
    bounds : array
        Max number of each item in collection
        
    T : float
        Temperature for statistical physics system

    Returns
    ----------
    xvec : array
        Vector of average occupancies
    """    
    
    # empty vector 
    xvec = np.zeros(len(weights)) 
    
    # filling in averages
    for k in range(len(weights)):
        
        xvec[k] = np.exp(values[k]/T)/(z**(-weights[k]) - np.exp(values[k]/T)) - (bounds[k]+1)*np.exp((bounds[k]+1)*values[k]/T)/(z**(-(bounds[k]+1)*weights[k]) - np.exp((bounds[k]+1)*values[k]/T))
        
    return xvec


def bounded_algorithm(weights, values, bounds, limit, override = False, T = 1.0, threshold = 0.6):
    
    """
    Full algorithm for the bounded KP
    
    Parameters
    ----------

    weights : array
        Vector of weights for objects
        
    values : array
        Vector of values for objects
        
    bounds : array
        Max number of each item in collection     
        
    limit : int
        Weight limit       

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
    z0 = fsolve(GN_bounded, x0=0.5, args = (weights, values,  bounds, limit,  T))[0]
    
    # solving for averages
    x_avgs = X_avg_bounded(z = z0, weights = weights, values = values, bounds = bounds, T= T)
    
    # thresholding averages
    x_soln = X_thresh(x_avgs, threshold = threshold)
    
    return x_soln
    

    
def unbounded_algorithm(weights, values, limit, T = 1.0, threshold = 0.6):
    
    """
    Full algorithm for the unbounded KP
    
    Parameters
    ----------

    weights : array
        Vector of weights for objects
        
    values : array
        Vector of values for objects
        
    W : int
        Weight limit       

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
    z0 = fsolve(GN_bounded, x0=0.5, args = (weights, values,  bounds, limit,  T))[0]
    
    # defining bounds 
    bounds = np.array([int(math.floor(W/w_elem)) for w_elem in weights])
    
    # solving for averages
    x_avgs = X_avg_bounded(z = z0, weights = weights, values = values, bounds = bounds, T= T)
    
    # thresholding averages
    x_soln = X_thresh(x_avgs, threshold = threshold)
    
    return x_soln
        
