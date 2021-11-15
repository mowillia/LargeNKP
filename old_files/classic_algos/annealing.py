import numpy as np


def trans_knapsack(x, w, Wlimit): 

    '''
    Randomly selects an item from our possible list of items and removes 
    it or adds contingent on whether it is already in the knapsack. 
    If adding the item pushes the knapsack above the weight limit, 
    another item is randomly selected. 
    
    x: vector representing the items in the pack
    w: vector representing the weight of each item
    Wlimit: maximum weight of the knapsack
    '''
    # define new vector
    x_new = np.copy(x)
    
    # selects random integer between 0 and len(X)-1
    idx = np.random.randint(len(x))
    
    # removes or adds item idx
    x_new[idx] = np.abs(x_new[idx]-1)
    
    k = 0
    # repeats selection if we have exceeded weight limit
    while np.dot(x_new, w) > Wlimit: 
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

# definition of log probability in terms of temperature

def log_prob_knapsack(x, v, T):
    '''
    Computes probability to have a particular set of objects 
    as a function of T
    
    x: vector representing the items in the pack
    v: vector representing the value of each item 
    T: "temperature"
    '''    
    return np.dot(x, v)/T    

### Simulated Annealing of Knapsack Problem

def simann_knapsack(weights, 
                    values, 
                    limit,
                    n_iterations=8000, 
                    init_temp = 3.0, 
                    hide = True):
    
    # number of elements
    nelems = len(weights)
    
    # initialized vector which will contain states
    X = np.zeros((n_iterations+1,nelems))
    
    # defining initial vector
    X[0] = np.random.randint(0,2,nelems)

    # choose again if we have exceeded weight
    while np.dot(X[0], weights) > limit:
        X[0] = np.random.randint(0,2,nelems)
    
    # sets temperature, accepted number, initial iteration, and reannealing
    temp, accepted, i, step_reannl = init_temp, 0, 0, 0
    
    # sets current probability
    current_logprob = log_prob_knapsack(X[0],values, temp) 
    
    # setting reannealing
    reannealing = n_iterations//20
    
    for i in range(n_iterations):
    
        # get current state and new state
        current_state = X[i]
        
        # proposed new permutation; generated from random integer sampling
        new_state = trans_knapsack(current_state, weights, limit) 
        
        # Calculate posterior with proposed value
        proposed_logprob = log_prob_knapsack(new_state, values, temp)

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
        if i > 100 and np.dot(X[-1], values)>0:
            var_obj = np.std([np.dot(elem, values) for elem in X[-100:]])

            if var_obj<1e-3:
                print('Stopping annealing because error tolerance was reached')
    
    # return our final sample
    return(X[-1])



