def greedy(weights, values, limit):
    
    nelems = len(weights)
    idxs = list(range(nelems))
    
    ratios = [(index, values[index] / float(weights[index])) for index in range(nelems)]
    ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
    best_combination = [0] * nelems
    best_value = 0
    tot_weight = 0
    for index, ratio in ratios:
        if weights[index] + tot_weight <= limit:
            tot_weight += weights[index]
            best_value += values[index]
            best_combination[index] = 1
    return best_combination


def greedy_ratio(weights, values, limit):
    
    nelems = len(weights)
    idxs = list(range(nelems))
    
    ratios = [(index, values[index] / float(weights[index])) for index in range(nelems)]
    ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
    best_combination = [0] * nelems
    best_value = 0
    tot_weight = 0
    
    num_ratio = 0
    for index, ratio in ratios:
        if weights[index] + tot_weight <= limit:
            tot_weight += weights[index]
            best_value += values[index]
            best_combination[index] = 1
        else:
            num_ratio +=1
            if num_ratio ==1:
                return(ratio)