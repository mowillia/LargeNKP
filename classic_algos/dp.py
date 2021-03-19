import numpy as np

def knapsack01_dpV(weights, values, limit):
    """
    Table elements are values
    """
    
    n = len(values)
    
    V = [[0 for w in range(limit + 1)] for j in range(n + 1)]
 
    for j in range(1, n + 1):
        wt, val = weights[j-1], values[j-1]
        for w in range(1, limit + 1):
            if w< wt:
                V[j][w] = V[j-1][w]
            else:
                V[j][w] = max(V[j-1][w],
                                  V[j-1][w-wt] + val)
 
    results = [0]*n
    w = limit
    for j in range(n, 0, -1):
        was_added = V[j][w] != V[j-1][w]
 
        if was_added:
            results[j-1] = 1
            w -= weights[j-1]
 
    return results



def knapsack01_dpW(weights, values, limit):
    """
    Table elements are weights
    """
    
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