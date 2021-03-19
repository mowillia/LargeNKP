from classic_algos.dp import knapsack01_dpW
import numpy as np
import math


def fptas(weights, values, limit, epsilon = 0.5):
    """
    Based on DP knapsack problem algorithm where matrix elements are weights
    and values define the elements
    """
    
    nvals = len(weights)
    
    max_value = max(values)
    
    K = epsilon*max_value/nvals
    
    new_values = np.zeros(nvals)
    for i in range(nvals):
        new_values[i] = math.floor(values[i]/K)
        
    new_values = [int(elem) for elem in new_values]
    
    return knapsack01_dpW(weights, new_values, limit)
