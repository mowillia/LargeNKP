from itertools import combinations
import numpy as np

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

def brute_force(weights, values, limit):
    
    nvals = len(weights)
    
    comb =  np.array([np.arange(nvals), weights, values]).T
    
    bagged = max( anycomb(comb), key= lambda i: totalvalue(i, limit) )
    
    result = [0]*nvals
    
    for item_idx, _, _ in bagged: 
        
        result[item_idx] = int(1)
    
    return result
