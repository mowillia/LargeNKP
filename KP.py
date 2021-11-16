import numpy as np
from matplotlib import lines
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, fmin
import math
from itertools import combinations


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
            
            raise Exception('Maximum weight is too low to be satisfiable.')
    
    return x_new

class KnapsackProblem:
    
    def __init__(self, weights, values, limit, bounds = None):
        self.weights = weights
        self.values = values
        self.limit = limit
        self.bounds = bounds
        
        '''
        Note: For the unbounded knapsack problem we need to insert an 
        explicit definition of bounds as
        
        bounds = np.array([int(math.floor(W/w_elem)) for w_elem in weights])
        '''
        # number of weights must be the same as the number of values
        assert len(self.values) == len(self.weights)    
        
        # number of bounds must be same as number of values (if there are bounds)
        if self.bounds is not None:
            assert len(self.bounds) == len(self.values)   
        
    def __str__(self):
        if self.bounds is None:
            return str(f'<Knapsack Instance, \nWeights:{self.weights},\nValues:{self.values},\nLimit:{self.limit}>')
        else:
            return str(f'<Knapsack Instance, \nWeights:{self.weights},\nValues:{self.values},\nLimit:{self.limit},\nBounds:{self.bounds}>')
        
    def __repr__(self):
        if self.bounds is None:
            return str(f'<Knapsack Instance, \nWeights:{self.weights},\nValues:{self.values},\nLimit:{self.limit}>')
        else:
            return str(f'<Knapsack Instance, \nWeights:{self.weights},\nValues:{self.values},\nLimit:{self.limit},\nBounds:{self.bounds}>')
        
        
    #####
    # Large N Algorithms: Zero-One and Bounded
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
    
    # defining function to produce average x
    def X_avg_bounded(self, z, T):

        """
        For the bounded KP
        Computes the average occupancy for each object given 
        the solution to the steepest descent condition, model parameters
        (weights, values, C) and model hyperparameter (T)

        Parameters
        ----------
        z : float
            Solution to steepest descent condition

        T : float
            Temperature for statistical physics system

        Returns
        ----------
        xvec : array
            Vector of average occupancies
        """    
        
        if self.bounds is None:
            return 'Not a Bounded Knapsack Problem'

        # empty vector 
        xvec = np.zeros(len(self.weights)) 

        # filling in averages
        for k in range(len(self.weights)):

            xvec[k] = np.exp(self.values[k]/T)/(z**(-self.weights[k]) - np.exp(self.values[k]/T)) - (self.bounds[k]+1)*np.exp((self.bounds[k]+1)*self.values[k]/T)/(z**(-(self.bounds[k]+1)*self.weights[k]) - np.exp((self.bounds[k]+1)*self.values[k]/T))

        return xvec    
    
    # potential function for zero-one problem
    def potential(self, z, T):
        return - self.limit*np.log(z)-np.log(1-z) + np.sum(np.log(1+z**(self.weights)*np.exp(self.values/T)))

    # derivative of potential function (with z product)    
    def constraint(self, z, T):
        return -self.limit+ z/(1-z) + np.sum(self.weights/(z**(-self.weights)*np.exp(-self.values/T) +1))   

 
    # potential function for bounded problem
    def potential_bounded(self, z, T):
        if self.bounds is None:
            return 'Not a Bounded Knapsack Problem'        
        
        return - self.limit*np.log(z)-np.log(1-z)+np.sum(np.log(1-z**((self.bounds+1)*self.weights)*np.exp((self.bounds+1)*self.values/T))) - np.sum(np.log(1-z**(self.weights)*np.exp(self.values/T)))

    # derivative of potential function for bounded problem (with z product)    
    def constraint_bounded(self, z, T):
        if self.bounds is None:
            return 'Not a Bounded Knapsack Problem'        
        
        return -self.limit+ z/(1-z) + np.sum(self.weights/(z**(-self.weights)*np.exp(-self.values/T) -1 )) - np.sum((self.bounds+1)*self.weights/(z**(-(self.bounds+1)*self.weights)*np.exp(-(self.bounds+1)*self.values/T) - 1))
        
    
    # consolidating algorithm
    def largeN_algorithm(self, x0 = 0.105, T = 1.0, threshold = 0.6):

        """
        Full algorithm for the zero-one KP

        Parameters
        ----------
        
        x0 : float
            Starting value for solving constrain equation

        T : float
            Temperature for statistical physics system

        threshold : float
            Limit for rounding to next highest integer


        Returns
        ----------
        x_soln : array
            Vector of occupancies

        """    
        
        # defaulting to ounbounded problem
        if self.bounds is None:
            # solving for z0
            z0 = fsolve(self.constraint, x0=x0, args = (T))[0]
            # solving for averages
            x_avgs = self.X_avg_zero_one(z = z0, T= T)
        else:
            # solving for z0
            z0 = fsolve(self.constraint_bounded, x0=x0, args = (T))[0]
            # solving for averages
            x_avgs = self.X_avg_bounded(z = z0, T= T)

        # thresholding averages
        x_soln = np.abs(np.ceil(x_avgs - threshold))

        return x_soln      
    
    # consolidating algorithm
    def limit_ratio(self, x0 = 0.095, T = 1.0, threshold = 0.6):

        """
        Computing limit ratio for weights and values

        Parameters
        ----------
        
        x0 : float
            Starting value for solving constrain equation                

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
        
        # solving for z0
        z0 = fsolve(self.constraint, x0=x0, args = (T))[0]

        # ratios when threshold isn't 1/2
        ratios = - T*np.log(z0)+ T*self.weights*np.log((1-threshold)/threshold)

        return ratios    
    
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
            soln = self.largeN_algorithm(x0 = x0, T = Tval)
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
                num_ratio +=1
                if num_ratio ==1:
                    return(ratio)      


    def knapsack01_dpV(self):
        '''
        Dynamical programming solution to knapsack problem
        Table elements are values
        '''

        n = len(self.values)

        V = [[0 for w in range(self.limit + 1)] for j in range(n + 1)]

        for j in range(1, n + 1):
            wt, val = self.weights[j-1], self.values[j-1]
            for w in range(1, self.limit + 1):
                if w< wt:
                    V[j][w] = V[j-1][w]
                else:
                    V[j][w] = max(V[j-1][w],
                                      V[j-1][w-wt] + val)

        results = [0]*n
        w = self.limit
        for j in range(n, 0, -1):
            was_added = V[j][w] != V[j-1][w]

            if was_added:
                results[j-1] = 1
                w -= self.weights[j-1]

        return results



    def knapsack01_dpW(self, weights=None, values=None, limit=None):
        '''
        Dynamical programming solution to knapsack problem
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

        results = [0]*n
        v = OPT
        for i in range(n, 0, -1):
            wt, val = weights[i-1], values[i-1]
            if val <= v:
                if wt + W[i-1][v-val]< W[i-1][v]:
                    results[i-1] = 1
                    v = v - val

        return results


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
        