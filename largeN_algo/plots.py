import numpy as np
import matplotlib.pyplot as plt
from largeN_algo.potential_functions import FN_zero_one
from matplotlib import lines
from largeN_algo.largeN import zero_one_algorithm

def plot_potential_zero_one(weights, values, limit, T): 
    
    """
    Plotting zero one potential for a 
    temperature or multiple temperatures
    
    Parameters
    ----------

    weights : array
        Vector of weights for objects
        
    values : array
        Vector of values for objects
        
    limit : int
        Weight limit
        
    T : float or narray or list
        Temperature for statistical physics system


    Returns
    ----------
    plot : figure
    
    """      
    

    # plotting "Potential landscape"

    mvals = 300 # number of points to plot
    ztest = np.linspace(0.001, 0.99, mvals) # values to evaluate potential

    # figure
    plt.figure(figsize = (6, 5))
    ax = plt.subplot(111)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    linestyle_list = list(lines.lineStyles.keys())

    if type(T) == float:
        Tlist = [T]
    else:
        Tlist = T

    # Plotting function for three temperature values
    Fvals_dict = dict() # empty dictionary of F values
    for i in range(len(Tlist)):
        Fvals_dict[i] = np.zeros(mvals)
        for k in range(mvals):
            Fvals_dict[i][k] = FN_zero_one(ztest[k], weights = weights, values = values, limit = limit, T= Tlist[i])
        ax.plot(ztest, Fvals_dict[i], label = r'$T = %.1f$' % Tlist[i], linestyle = linestyle_list[i], linewidth = 2.5)


    # plot sides
    plt.xlabel(r'$z$', fontsize = 20, x = 1.0)
    plt.xticks(fontsize=13 )
    plt.ylabel(r'$F_N(z)$', fontsize = 18, rotation = 0, labelpad = 30, y = 0.9)
    plt.yticks(fontsize= 13)
    plt.grid(alpha = 0.5)
    plt.legend(loc = 'best', fontsize = 15)
    
    plt.show()
    
    
def plot_value_vs_temp(weights, values, limit, temp_low= 0.05, temp_high = 5.0): 
    
    """
    Plotting total computed value vs tempearture
    
    Parameters
    ----------

    weights : array
        Vector of weights for objects
        
    values : array
        Vector of values for objects
        
    limit : int
        Weight limit
        
    temp_low : float
        Lowest temperature to solve

    temp_high : float
        Highest temperature to solve

    Returns
    ----------
    plot : figure
    
    """      
    
    # temperature range and empty values
    Tvals = np.linspace(temp_low, temp_high, 300)
    total_value_list = list()

    for Tval in Tvals: 
        soln = zero_one_algorithm(weights = weights, values = values, limit = limit, T = Tval)
        total_value_list.append(np.dot(soln, values))    

    # figure
    plt.figure(figsize = (6, 5))
    ax = plt.subplot(111)

    #plots
    plt.plot(Tvals, total_value_list, label = r'Large $N$ Value', linewidth = 2.5)

    # plot formatting
    plt.xlabel(r'$T$', fontsize = 20, x = 1.0, labelpad = 20)
    plt.ylabel(r'$V$', fontsize = 18, rotation = 0, labelpad = 30, y = 0.9)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc = 'best', fontsize = 15)
    plt.grid(alpha = 0.5)
    
    plt.show()    

    
    