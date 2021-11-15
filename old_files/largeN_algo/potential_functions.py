import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines

# potential function for zero-one problem
FN_zero_one = lambda z, weights, values, limit, T: - limit*np.log(z)-np.log(1-z) + np.sum(np.log(1+z**(weights)*np.exp(values/T)))

# constraint function for zero-one problem
GN_zero_one = lambda z, weights, values, limit, T: -limit+ z/(1-z) + np.sum(weights/(z**(-weights)*np.exp(-values/T) +1))

# potential function for bounded problem
FN_bounded = lambda z, weights, values, bounds, limit, T: - limit*np.log(z)-np.log(1-z)+np.sum(np.log(1-z**((bounds+1)*weights)*np.exp((bounds+1)*values/T))) - np.sum(np.log(1-z**(weights)*np.exp(values/T)))

# constraint function for bounded problem
GN_bounded = lambda z, weights, values, bounds, limit, T: -limit+ z/(1-z) + np.sum(weights/(z**(-weights)*np.exp(-values/T) -1 )) - np.sum((bounds+1)*weights/(z**(-(bounds+1)*weights)*np.exp(-(bounds+1)*values/T) - 1))

###
# Note: Unbounded problem functions and potential are not included because we use the bounded problem as the basis for a solution
###
    
    