import numpy as np
from matplotlib import lines
import matplotlib.pyplot as plt
import copy
from scipy.optimize import fsolve, fmin
import math
from itertools import combinations

import mpmath as mp
mp.dps = 30; mp.pretty = True

# defining exponential for high precision calcs
mp_exp_array = np.frompyfunc(mp.exp, 1, 1)

def anycomb(items):
    ' return combinations of any length from the items '
    
    nvals = len(items)
    return ( comb
             for r in range(1, nvals+1)
             for comb in combinations(items, r)
             )
 
def totalvalue(comb, limit):
    ' Totalise a particular combination of items'
    totwt = totval = 0
    for item, wt, val in comb:
        totwt  += wt
        totval += val
    return (totval, -totwt) if totwt <= limit else (0, 0)


def x_selected(x_vals, w_vals, W):
    
    '''
    Selects the objects in decreasing order of 
    probability of occupancy until the weight limit is satisfied
    '''
    
    final_object_list = np.zeros_like(x_vals)
    for _ in range(len(x_vals)):
        new_final_object_list = copy.deepcopy(final_object_list)
        idx1 = np.argmax(x_vals)
        new_final_object_list[idx1] = 1
        x_vals[idx1] = 0
        if np.dot(w_vals, new_final_object_list) <= W:
            final_object_list = copy.deepcopy(new_final_object_list)
        else:
            break
            
    return final_object_list

def find_root(func, num_powers = 20, **args):
    
    """
    Finds the root of a function in the domain (0, 1) exclusive

    Parameters
    ----------
    func : function
        Function for which we are seeking the root
        
    num_powers : int
        Number of powers of 10 to include in final answer

    Returns
    ----------
    fin_z : float
        Value of critical point

    error : float 
        How far function is from zero at this critical point

    """    
    
    powers_10 = np.array([10**(-k) for k in range(num_powers+1)])
    coeffs = np.zeros_like(powers_10)
    coeffs[0] = 0

    # determining initial order of current z
    for i in range(1, num_powers+1):
        coeffs[i] = 1
        current_z =  np.dot(coeffs, powers_10)
        current_sign = np.sign(func(current_z, **args).real)
        # reset coeffs if sign doesn't change
        if current_sign > 0:
            coeffs[i] = 0
            l_start = i
        else:
            l_start = i
            break
            
    # determine subsequent coefficients
    for k in range(l_start,num_powers+1):

        # marching up
        for j in range(10):

            # try a value of the coefficient
            coeffs[k] = j
            new_z = np.dot(coeffs, powers_10)
            new_sign = np.sign(func(new_z,**args).real)  

            # if the sign has changed from the current sign, go to previous value
            if new_sign > 0:
                coeffs[k] = j-1 # set to the j value before the change
                current_z = np.dot(coeffs, powers_10)
                break

    # final z value
    fin_z = np.dot(coeffs, powers_10)    
    
    # error 
    error = func(fin_z, **args).real
    
    return fin_z, float(error)

def find_root_bisection(func, xmax = 100, num_bisections = 50, **args):

    """
    https://en.wikipedia.org/wiki/Bisection_method
    
    Finds the root of a function in the domain (0, 1) by 
    the bisection method

    Parameters
    ----------
    func : function
        Function for which we are seeking the root

    xmax : float
        Top value of interval for bisection
        
    num_bisections : int
        Number of times to bisect interval

    Returns
    ----------
    fin_z : float
        Value of critical point

    error : float 
        How far function is from zero at this critical point

    """        
    
    a = 0
    b = xmax
    c = (a+b)/2

    for k in range(num_bisections):
        f_a = func(a, **args).real
        f_b = func(b, **args).real
        f_c = func(c, **args).real

        if np.sign(f_a) == -np.sign(f_c):
            b = c
            c = (a+b)/2

        elif np.sign(f_b) == -np.sign(f_c):  
            a = c
            c = (a+b)/2

    fin_z = c
    error = func(c, **args).real

    return fin_z, error      

def trans_knapsack(x, weights, limit): 

    '''
    Randomly selects an item from our possible list of items and removes 
    it or adds contingent on whether it is already in the knapsack. 
    If adding the item pushes the knapsack above the weight limit, 
    another item is randomly selected. 

    Parameters
    ----------
    x : array
        Vector representing the items in the pack

    weights : array
        Vector of weights for objects

    limit : float
        Maximum weight of knapsack

    Returns
    ----------
    x_new : array
        New vector respresenting items in knapsack
    '''
    
    # define new vector
    x_new = np.copy(x)
    
    # selects random integer between 0 and len(X)-1
    idx = np.random.randint(len(x))
    
    # removes or adds item idx
    x_new[idx] = np.abs(x_new[idx]-1)
    
    k = 0
    # repeats selection if we have exceeded weight limit
    while np.dot(x_new, weights) > limit: 
        # define new vector
        x_new = np.copy(x)

        # selects random integer between 0 and len(X)-1
        idx = np.random.randint(len(x))

        # removes or adds item idx
        x_new[idx] = np.abs(x_new[idx]-1)
        
        #increment k by 1
        k +=1 
        
        if k == 100:
            
            raise Exception('Too many selection attempts; Maximum weight appears to be too low to be satisfiable.')
    
    return x_new

class KnapsackProblem:
    
    def __init__(self, weights, values, limit):
        self.weights = weights
        self.values = values
        self.limit = limit
        
        
        # number of weights must be the same as the number of values
        assert len(self.values) == len(self.weights)    
        
    def __str__(self):
        return str(f'<Knapsack Instance, \nWeights:{self.weights},\nValues:{self.values},\nLimit:{self.limit}>')
        
    def __repr__(self):
        return str(f'<Knapsack Instance, \nWeights:{self.weights},\nValues:{self.values},\nLimit:{self.limit}>')
        
    #####
    # Exact Partition Function Calculation
    #####
    
    def exact_z_algorithm(self, T=1.0, rounded=True, threshold=0.75):
        """
        Computes the exact partition function for the KP 
        using dynamical programming and then uses this solution
        to write the solution to the KP.

        """

        # item number and weight limit
        N, W = len(self.weights), self.limit

        # defining empty partition function matrix
        Z = [[1 for x in range(W + 1)] for q in range(N + 1)]

        # implementing recursion identity 
        for i in range(1, N + 1):
            wt, val = self.weights[i-1], self.values[i-1]
            for w in range(1, W + 1):                
                if w < wt:
                    Z[i][w] = Z[i-1][w]                   
                else:                     
                    # partition function identity 
                    Z[i][w] = Z[i-1][w] + mp.exp(val/T)*Z[i-1][w-wt]         
            
        # computing occupancy vector
        occupancy = [0]*N
        w = W
        for j in range(N, 0, -1):
            # using X_j definition with simple half threshold
            was_added = 1 - Z[j-1][w]/Z[j][w] > threshold

            if was_added:
                occupancy[j-1] = 1
                w -= self.weights[j-1]

        return occupancy 
        
        
    #####
    # Large W Algorithms: Zero-One
    #####        
        
    # defining function to produce average x
    def X_avg_zero_one(self, z = 0.095, T=1.0):

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

        nvals = len(self.weights)

        # empty vector 
        xvec = np.zeros(nvals) 

        # filling in averages
        for k in range(nvals):

            xvec[k] = np.exp(self.values[k]/T)/(z**(-self.weights[k]) + np.exp(self.values[k]/T))

        return xvec
    
    
    # potential function for zero-one problem
    def potential(self, z, T):
        return - self.limit*np.log(z)-np.log(1-z) + np.sum(np.log(1+z**(self.weights)*np.exp(self.values/T)))

    
    # derivative of potential function (with z product)    
    def constraint(self, z, T):
        return -self.limit+ z/(1-z) + np.sum(self.weights/(mp_exp_array(-self.values/T)*mp_exp_array(-self.weights*mp.log(z)) + 1))   
    
    # constraint function in the limit of zero temperature
    def constraint_zero(self, gamma):
        return  self.limit - np.sum(self.weights*np.heaviside(self.values/self.weights - gamma, 1/2))

    # consolidating algorithm
    def largeW_algorithm(self, return_gamma=False, T = None, threshold = 0.75, weight_check = False, verbose = False):

        """
        Full algorithm for the zero-one KP

        Parameters
        ----------
        return_gamma : Bool
            Whether to return the value of gamma
            
        T : float
            If not none, we give thresholded values of the average
            
        threshold : float
            Threshold for computing x_soln when T is given

        weight_check : bool
            Checks if solution satisfies weight limit and modifies if it doesn't

        Returns
        ----------
        x_soln : array
            Binary vector of occupancies
            
        gamma_val: 
            Value of gamma (minimum value/weight ratio)

        """    
        
        if T:
            # solving for z0
            # z0  = fmin(self.potential, x0=0.09, args = (T,), disp=False)[0]
            z0, error = find_root(self.constraint, T=T)
            
            # solving for averages
#             x_avgs =  1/(np.exp(-self.weights*np.log(z0)-self.values/T) + 1)
            x_avgs = np.array([float(elem) for elem in 1/(mp_exp_array(-self.weights*np.log(z0)-self.values/T) + 1)])

            # including threshold 
            x_soln = np.heaviside(x_avgs - threshold, 0)
            
            # removes marginal values if weight limit is violated
            if weight_check: 
                diffs = x_avgs - threshold
                diffs_mask = diffs>0
                sorted_masked_diffs = np.sort(diffs*diffs_mask) # differences in ascending order
                for k in range(len(diffs)):
                    diff_val = sorted_masked_diffs[k]
                    if diff_val > 0 and self.limit < np.dot(self.weights, x_soln):
                        idx = np.where(diffs == diff_val)[0][0]
                        if verbose: 
                            print(f'Eliminating object {idx} due to weight violation')
                        x_soln[idx] = 0            
        
        else: 
            # starting temperature
            gamma_val = self.gamma_calc()

            # selecting values according to definition
            x_soln = np.heaviside(self.values/self.weights - gamma_val, 0)
            
            # removes marginal values if weight limit is violated
            if weight_check: 
                diffs = self.values - gamma_val*self.weights
                diffs_mask = diffs>0
                sorted_masked_diffs = np.sort(diffs*diffs_mask) # differences in ascending order            
                for k in range(len(diffs)):
                    diff_val = sorted_masked_diffs[k]
                    if diff_val > 0 and self.limit < np.dot(self.weights, x_soln):
                        idx = np.where(diffs == diff_val)[0][0]
                        if verbose: 
                            print(f'Eliminating object {idx} due to weight violation')
                        x_soln[idx] = 0
                           
            
        return x_soln


    def gamma_calc(self):

        """
        Computing limit ratio for weights and values

        Parameters
        ----------

        Returns
        ----------
        gamma_zero : float
            Value of gamma at T=0

        """           

        # computing gamma_zero value
        gamma_zero, _ = find_root_bisection(self.constraint_zero, num_bisections=70)

        return gamma_zero
 
    
    def plot_potential(self, T=1.0): 

        """
        Plotting zero one potential for a 
        temperature or multiple temperatures

        Parameters
        ----------

        T : float or narray or list
            Temperature for statistical physics system


        Returns
        ----------
        plot : figure

        """      


        # plotting "Potential landscape"

        mvals = 1000 # number of points to plot
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
                Fvals_dict[i][k] = self.potential(z= ztest[k], T = Tlist[i])
            ax.plot(ztest, Fvals_dict[i], label = r'$T = %.1f$' % Tlist[i], linestyle = linestyle_list[i], linewidth = 2.5)


        # plot sides
        plt.xlabel(r'$z$', fontsize = 20, x = 1.0)
        plt.xticks(fontsize=13 )
        plt.ylabel(r'$F_N(z)$', fontsize = 18, rotation = 0, labelpad = 30, y = 0.9)
        plt.yticks(fontsize= 13)
        plt.grid(alpha = 0.5)
        plt.legend(loc = 'best', fontsize = 15)

        plt.show()
    
    def plot_constraint(self, T=1.0): 

        """
        Plotting zero one constraint for a 
        temperature or multiple temperatures

        Parameters
        ----------

        T : float or narray or list
            Temperature for statistical physics system


        Returns
        ----------
        plot : figure

        """      


        # plotting "Constraint"

        mvals = 1000 # number of points to plot
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
        partialFvals_dict = dict() # empty dictionary of F values
        for i in range(len(Tlist)):
            partialFvals_dict[i] = np.zeros(mvals)
            for k in range(mvals):
                partialFvals_dict[i][k] = self.constraint(z= ztest[k], T = Tlist[i])
            ax.plot(ztest, partialFvals_dict[i], label = r'$T = %.1f$' % Tlist[i], linestyle = linestyle_list[i], linewidth = 2.5)


        # plot sides
        plt.xlabel(r'$z$', fontsize = 20, x = 1.0)
        plt.xticks(fontsize=13 )
        plt.ylabel(r'$partial_F_N(z)$', fontsize = 18, rotation = 0, labelpad = 30, y = 0.9)
        plt.yticks(fontsize= 13)
        plt.grid(alpha = 0.5)
        plt.legend(loc = 'best', fontsize = 15)

        plt.show()    
    
    def plot_value_vs_temp(self, x0 = 0.095, temp_low= 0.05, temp_high = 5.0): 

        """
        Plotting total computed value vs tempearture

        Parameters
        ----------

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
            soln = self.largeW_algorithm(T = Tval)
            total_value_list.append(np.dot(soln, self.values))    

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
        
    #####
    # Standard Algorithms
    #####

        
    def brute_force(self):

        '''
        Brute force solution to knapsack problem
        '''        
        
        nvals = len(self.weights)

        comb =  np.array([np.arange(nvals), self.weights, self.values]).T

        bagged = max( anycomb(comb), key= lambda i: totalvalue(i, self.limit) )

        result = [0]*nvals

        for item_idx, _, _ in bagged: 

            result[item_idx] = int(1)

        return result        
        
    def greedy(self):

        '''
        Greedy solution to knapsack problem
        '''
        
        nelems = len(self.weights)
        idxs = list(range(nelems))

        ratios = [(index, self.values[index] / float(self.weights[index])) for index in range(nelems)]
        ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
        best_combination = [0] * nelems
        best_value = 0
        tot_weight = 0
        for index, ratio in ratios:
            if self.weights[index] + tot_weight <= self.limit:
                tot_weight += self.weights[index]
                best_value += self.values[index]
                best_combination[index] = 1
        return best_combination
    
    
    def greedy_ratio(self):
        '''
        Computes the minimum value/weight ratio for 
        including objects in greedy solution to knapsack problem
        '''

        nelems = len(self.weights)
        idxs = list(range(nelems))

        ratios = [(index, self.values[index] / float(self.weights[index])) for index in range(nelems)]
        ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
        best_combination = [0] * nelems
        best_value = 0
        tot_weight = 0

        num_ratio = 0
        for index, ratio in ratios:
            if self.weights[index] + tot_weight <= self.limit:
                tot_weight += self.weights[index]
                best_value += self.values[index]
                best_combination[index] = 1
            else:
                return(ratio)      


    def knapsack01_dpV(self):
        '''
        Dynamic programming solution to knapsack problem
        Table elements are values
        '''

        N, W = len(self.values), self.limit

        V = [[0 for w in range(W + 1)] for j in range(N + 1)]
        
        # computing value matrix
        for j in range(1, N + 1):
            wt, val = self.weights[j-1], self.values[j-1]
            for w in range(1, W + 1):
                if w< wt:
                    V[j][w] = V[j-1][w]
                else:
                    V[j][w] = max(V[j-1][w], V[j-1][w-wt] + val)
        
        # computing occupancy vector
        occupancy = [0]*N
        w = W
        for j in range(N, 0, -1):
            was_added = V[j][w] != V[j-1][w]

            if was_added:
                occupancy[j-1] = 1
                w -= self.weights[j-1]

        return occupancy



    def knapsack01_dpW(self, weights=None, values=None, limit=None):
        '''
        Dynamic programming solution to knapsack problem
        Table elements are weights
        '''

        if weights is None:
            weights = self.weights
        if values is None: 
            values = self.values
        if limit is None:
            limit = self.limit

        Vmax = np.sum(values)
        n = len(values)

        W = [[0 for v in range(Vmax+1)] for j in range(n+1)]

        W[0][0] = 0

        for v in range(1,Vmax+1):
            W[0][v] = np.sum(weights)

        for i in range(1, n+1):
            for v in range(1, Vmax+1):
                if values[i-1] <= v:
                    W[i][v] = min(W[i-1][v], weights[i-1]+W[i-1][v-values[i-1]])
                else:
                    W[i][v] = W[i-1][v]

        OPT = max([v for v in range(Vmax) if W[n][v] <= limit])

        occupancy = [0]*n
        v = OPT
        for i in range(n, 0, -1):
            wt, val = weights[i-1], values[i-1]
            if val <= v:
                if wt + W[i-1][v-val]< W[i-1][v]:
                    occupancy[i-1] = 1
                    v = v - val

        return occupancy


    def fptas(self, epsilon = 0.5):
        '''
        Fully polynomial time approximation scheme solution to knapsack problem
        
        Based on DP knapsack problem algorithm where matrix elements are weights
        and values define the elements
        '''

        nvals = len(self.weights)

        max_value = max(self.values)

        K = epsilon*max_value/nvals

        new_values = np.zeros(nvals)
        for i in range(nvals):
            new_values[i] = math.floor(self.values[i]/K)

        new_values = [int(elem) for elem in new_values]

        return self.knapsack01_dpW(weights = self.weights, values = new_values, limit = self.limit)

    
    def simann_knapsack(self,n_iterations=8000, 
                        init_temp = 3.0, 
                        hide = True):

        '''
        Simulated annealing solultion to knapsack problem
        '''
        
        # number of elements
        nelems = len(self.weights)

        # initialized vector which will contain states
        X = np.zeros((n_iterations+1,nelems))

        # defining initial vector
        X[0] = np.random.randint(0,2,nelems)

        # choose again if we have exceeded weight
        while np.dot(X[0], self.weights) > self.limit:
            X[0] = np.random.randint(0,2,nelems)

        # sets temperature, accepted number, initial iteration, and reannealing
        temp, accepted, i, step_reannl = init_temp, 0, 0, 0

        # sets current probability
        current_logprob = np.dot(X[0],self.values)/temp

        # setting reannealing
        reannealing = n_iterations//20

        for i in range(n_iterations):

            # get current state and new state
            current_state = X[i]

            # proposed new permutation; generated from random integer sampling
            new_state = trans_knapsack(current_state, self.weights, self.limit) 

            # Calculate posterior with proposed value
            proposed_logprob = np.dot(new_state, self.values)/temp

            # Log-acceptance rate
            log_alpha = proposed_logprob - current_logprob

            # Sample a uniform random variate
            log_u = np.log(np.random.random())

            # Test proposed value
            if log_u < log_alpha:
                # Accept and make next state the new state
                X[i+1] = new_state

                # make proposed_log probability the new log_probability
                current_logprob = proposed_logprob

                # increment accepted by 1
                accepted += 1

            else:
                # Stay put
                X[i+1] = X[i]

            # increment i by 1
            i = i+1

            if accepted % reannealing == 0:
                # increment step for annealing by 1
                step_reannl +=1 

                if hide == False:
                    # print current temperature and reannealing iteration
                    print('Current Temperature: %f, Reannealing # %f' % (temp, step_reannl))

                ## annealing schedule
                temp = temp/(np.log(1+step_reannl)) 

                # reheating
                if temp == 0.5:
                    print('Reheating')
                    temp = init_temp

            # stop if we reach a certain error tolerance
            if i > 100 and np.dot(X[-1], self.values)>0:
                var_obj = np.std([np.dot(elem, self.values) for elem in X[-100:]])

                if var_obj<1e-3:
                    print('Stopping annealing because error tolerance was reached')

        # return our final sample
        return(X[-1])        
        