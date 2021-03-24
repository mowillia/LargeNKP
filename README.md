*Work completed in [Jellyfish Research](https://jellyfish.co/).*

# Large N Limit of Knapsack Problem

<p align="center">
<img align = "center" src = "https://user-images.githubusercontent.com/8810308/111638380-a53b3400-87d0-11eb-9407-78a613cdd922.png"  onmouseover= "Motivation for statistical physics based algorithm" width = "75%">
</p>

The Knapsack Problem is a classic problem from combinatorial optimization. In the "0-1" version of the problem, we are given N objects each of which has a value and a weight, and our objective is to find the collection of objects that maximizes the total value of the collection while ensuring that the weight remain under a given maximum. 

This repository provides algorithms for solving various incarnations of the  Knapsack Problem in the limit of where the total number of elements is large. Currently the libary supports approximate solutions to the "0-1", "bounded", and "unbounded" versions of the problem. 

There are exact algorithms for the knapsack problem [(RossettaCode Knapsack)](https://rosettacode.org/wiki/Knapsack_problem), but these take longer as the number of items increases. The algorithms in this repository provide approximate solutions in much less time. 


## Knapsack Instance

The following examples are taken from the [`examples.ipynb`](https://github.com/mowillia/largeNKP/blob/main/examples.ipynb) file. Run the entire file to reproduce all of the results below. 

In the following examples (up to Knapsack Problem Variation), we will use the item list, weights, values, and weight limits given as follows.
```
import numpy as np

items = (
    ("map", 9, 150), ("compass", 13, 35), ("water", 153, 200), ("sandwich", 50, 160),
    ("glucose", 15, 60), ("tin", 68, 45), ("banana", 27, 60), ("apple", 39, 40),
    ("cheese", 23, 30), ("beer", 52, 10), ("suntan cream", 11, 70), ("camera", 32, 30),
    ("t-shirt", 24, 15), ("trousers", 48, 10), ("umbrella", 73, 40),
    ("waterproof trousers", 42, 70), ("waterproof overclothes", 43, 75),
    ("note-case", 22, 80), ("sunglasses", 7, 20), ("towel", 18, 12),
    ("socks", 4, 50), ("book", 30, 10),
    )

# defining weight and value vectors and weight limit
weight_vec = np.array([item[1] for item in items])
value_vec = np.array([item[2] for item in items])
Wlimit = 400
```

These values are taken from the problem statement in [RossettaCode Knapsack: 0-1](https://rosettacode.org/wiki/Knapsack_problem/0-1)

## Running Large N algorithm

Given weights, values, and a limit, the large N algorithm outputs a list of 1s and 0 correspon algorithm corresponding to putting the respective item in the list in the knapsack or leaving it out. To quickly run the algorithm, execute the following code after defining the item list above.

```
from largeN_algo import zero_one_algorithm

soln = zero_one_algorithm(weights = weight_vec, values = value_vec, limit = Wlimit)
for k in range(len(soln)):
    if soln[k] == 1:
        print(items[k][0])
```

```
>>>
map
compass
water
sandwich
glucose
banana
suntan cream
waterproof trousers
waterproof overclothes
note-case
sunglasses
socks
```

To apply the problem to other instances of items, values, and weights, just replace the values and weight lists in the quick start up with your chosen lists. 

## Plotting potential function

The potential function for the zero-one problem is 
```
FN_zero_one = lambda z, weights, values, limit, T: - limit*np.log(z)-np.log(1-z) + np.sum(np.log(1+z**(weights)*np.exp(values/T)))
```
This function gives a continuous representation of the standard discrete optimization objective. If the function has a local minimum, then the large N algorithm can solve the knapsack problem. This minimum depends on temperature, and as the temperature is lowered the minimum better defines an optimal solution for the knapsack problem. To plot the potential function for the above instance, execute the following code. 

```
from largeN_algo import plot_potential_zero_one

plot_potential_zero_one(weights = weight_vec, values = value_vec, limit = Wlimit, T= 1.5)
>>>
```
<p align="center">
<img align = "middle" src = "https://user-images.githubusercontent.com/8810308/111629285-84221580-87c7-11eb-9486-6828c446040d.png" width = "40%">
</p>

## Plotting total value as a function of temperature

To plot the calculated total value as a function of temperature, execute the following code

```
from largeN_algo import plot_value_vs_temp

plot_value_vs_temp(weights = weight_vec, values = value_vec, limit = Wlimit, temp_low=1.0, temp_high = 60.0)
>>>
```
<p align="center">
<img align = "middle" src = "https://user-images.githubusercontent.com/8810308/111698215-f5d08280-880c-11eb-8361-330a35755881.png" width = "40%">
</p>

## Algorithm comparison plots

In the original paper, we compare the performance of various classic knapsack problem algorithms to the proposed algorithm. The algorithms we compare are

- **Brute Force**(`brute_force`): Involves listing all possible combinations of items, computing the total weights and total values of each combination and selecting the combination with the highest value with a weight below the stated limit. 

- **Dynamical Programming Solution**(`knapsack_dpV`):  Standard iterative solution to the problem which involves storing sub-problem solutions in matrix elements

- **Fully Polynomial Time Approximate Solution (FPTAS)**(`fptas`):  Algorithm that is polynomial time in the number of elements and which has a tunable accuracy

- **Greedy Algorithm**(`greedy`):  Involves computing the ratio of weights to volumes for each object and filling in the collection until the max weight is reached. 

- **Simulated Annealing**(`simannl_knapsack`):   Involves representing the system computationally as a statistical physics one and then "annealing" the system to low temperatures. 

- **Large N Algorithm**(`zero_one_algorithm`):  Algorithm based on statistical physics representation of the system

A quick comparison of these algorithms for the problem instance shown above is given by the following code. 

Assembling needed algorithms and modules
```
from classic_algos import (brute_force, 
                           knapsack01_dpV, 
                           fptas, 
                           greedy, 
                           simann_knapsack)
from largeN_algo import zero_one_algorithm

from tabulate import tabulate
from collections import defaultdict

import time
```
Defining dictionary of algorithms and empty dictionary for results
```
# dictionary of algorithm names and functions
algo_name_dict = {'Brute': brute_force,
                  'DP': knapsack01_dpV,
                  'FPTAS': fptas,
                  'Greedy': greedy,
                  'Annealing': simann_knapsack,
                  'Large N': zero_one_algorithm}

# dictionary of algorithm names and results
results_name_dict = dict()
```
Running algorithm and creating table of results
```
for name, func in algo_name_dict.items():
    start_clock = time.time()
    soln  = func(weights = weight_vec, values = value_vec, limit = Wlimit)
    
    # calculating values
    tot_value = str(round(np.dot(value_vec, soln), 0))
    tot_weight = str(round(np.dot(weight_vec, soln), 0))
    time_calc = str(round(time.time()-start_clock, 5)) 
    
    # assembling results
    results_name_dict[name] = [name, tot_value, tot_weight, time_calc]
    
# creating table of results
tabular_results = []
for k, v in results_name_dict.items():
    tabular_results.append(v) 
```
Printing Table
```
print(tabulate(tabular_results, ["Algorithm", "Value", "Weight", "Time (sec)"], tablefmt="grid"))
>>>
+-------------+---------+-----------+--------------+
| Algorithm   |   Value |   Weight  |   Time (sec) |
+=============+=========+===========+==============+
| Brute       |    1030 |       396 |     33.0739  |
+-------------+---------+-----------+--------------+
| DP          |    1030 |       396 |      0.00488 |
+-------------+---------+-----------+--------------+
| FPTAS       |    1030 |       396 |      0.00396 |
+-------------+---------+-----------+--------------+
| Greedy      |    1030 |       396 |      8e-05   |
+-------------+---------+-----------+--------------+
| Annealing   |     920 |       387 |      0.13638 |
+-------------+---------+-----------+--------------+
| Large N     |    1030 |       396 |      0.00107 |
+-------------+---------+-----------+--------------+

```


## Knapsack Variations: Bounded

Repository includes code for two variations of the knapsack problem: The unbounded and the bounded knapsack problems. Their implementations are identical to the implementation for the `zero_one_algorithm` except the bounded knapsack problem takes the additional argument of "bounds". The following problem instance is from [RossettaCode Knapsack: Bounded](https://rosettacode.org/wiki/Knapsack_problem/Bounded#Dynamic_programming_solution_2)

```
from largeN_algo import bounded_algorithm
import numpy as np

items = (("map", 9, 150, 1),("compass", 13, 35, 1), ("water", 153, 200, 3),("sandwich", 50, 60, 2),
         ("glucose", 15, 60, 2),("tin", 68, 45, 3), ("banana", 27, 60, 3),("apple", 39, 40, 3),
            ("cheese", 23, 30, 1),("beer", 52, 10, 3),("suntan cream", 11, 70, 1),("camera", 32, 30, 1),
            ("t-shirt", 24, 15, 2),("trousers", 48, 10, 2),("umbrella", 73, 40, 1),("waterproof trousers", 42, 70, 1),
            ("waterproof overclothes", 43, 75, 1),("note-case", 22, 80, 1),("sunglasses", 7, 20, 1),("towel", 18, 12, 2),
            ("socks", 4, 50, 1),("book", 30, 10, 2),
           )

# defining weight and value vectors and weight limit
weight_vec = np.array([item[1] for item in items])
value_vec = np.array([item[2] for item in items])
bound_vec = np.array([item[3] for item in items])
Wlimit = 400

soln = bounded_algorithm(weights = weight_vec, values = value_vec, bounds=bound_vec, limit = Wlimit, T = 8.10, threshold = 0.51)
print('Item: Item #')
print('-----------')
for k in range(len(soln)):
    if soln[k] == 1:
        print('%s : %i ' % (items[k][0], items[k][3]))
print()        
print('Total Value: %i' % (np.dot(soln, value_vec)))
print('Total Weight: %i' % (np.dot(soln, weight_vec)))
```
With the result

```
>>>
Item: Item #
-----------
map : 1 
compass : 1 
water : 3 
suntan cream : 1 
waterproof trousers : 1 
waterproof overclothes : 1 
note-case : 1 
sunglasses : 1 
socks : 1 

Total Value: 1050
Total Weight: 415
```

This solution over-estimates the actual optimum because it did not respect the weight limit. Problems with larger values of N should fare better.

Notes: Although these algorithms are analytically based on a large N approximation, Python has bounds on the size of integers it can process. Thus often using the bounded algorithm (with variables raised to an extra power of the bound), results in an overflow eror.


## Reproducing figures and tables

The notebooks that reproduce the figures and tables in the paper are as follows

- [`potential_landscape.ipynb`](https://github.com/mowillia/largeNKP/blob/main/potential_landscape.ipynb): Reproduces Figure 2(a); Runs in < 1 minute
- [`total_value_vs_temperature.ipynb`](https://github.com/mowillia/largeNKP/blob/main/total_value_vs_temperature.ipynb): Reproduces Figure 2(b); Runs in < 1 minute
- [`algorithm_comparisons.ipynb`](https://github.com/mowillia/largeNKP/blob/main/algorithm_comparisons.ipynb): Reproduces Figure 3; Runs in 15 minutes
- [`limit_ratio_vs_temperature.ipynb`](https://github.com/mowillia/largeNKP/blob/main/limit_ratio_vs_temperature.ipynb): Reproduces Table 1; Runs in < 1 minute
- [`failure_modes.ipynb`](https://github.com/mowillia/largeNKP/blob/main/failure_modes.ipynb): Gives examples of "Failure Modes" discussed in Appendix


<!---
## References
[1] Mobolaji Williams. "Large N Limit of the Knapsack Problem." *Journal Name.* 2021. [[arxiv]](https://arxiv.org/abs/XXXX)
--->
---
<!---
If you found this repository useful in your research, please consider citing
```
@article{williams2021knapsack,
  title={Large N Limit of the Knapsack Problem},
  author={Williams, Mobolaji},
  journal={arXiv preprint arXiv:CCC},
  year={2021}
}
```
--->
