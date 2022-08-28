# Large W Limit of Knapsack Problem

<p align="center">
<img align = "center" src = "https://user-images.githubusercontent.com/8810308/111638380-a53b3400-87d0-11eb-9407-78a613cdd922.png"  onmouseover= "Motivation for statistical physics based algorithm" width = "75%">
</p>

The Knapsack Problem is a classic problem from combinatorial optimization. In the "0-1" version of the problem, we are given N objects each of which has a value and a weight, and our objective is to find the collection of objects that maximizes the total value of the collection while ensuring that the weight remain under a given maximum. 

This repository contains code for an algorithm that solves this "0-1" problem in the limit where the weight W is large.

We compare this algorithm with other exact algorithms for the knapsack problem [(RossettaCode Knapsack)](https://rosettacode.org/wiki/Knapsack_problem), which generally proceed in more time. The code and figures in this repository were used to produce the results in the [associated paper](https://arxiv.org/abs/2107.14080).


## Knapsack Instance

The following examples are taken from the [`example.ipynb`](https://github.com/mowillia/largeNKP/blob/main/example.ipynb) file. Run the entire file to reproduce all of the results below. 

In the following examples, we will use the item list, weights, values, and weight limits given as follows.
```
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

# defining instance of problem
KP_camping = KnapsackProblem(weights = weight_vec, values = value_vec, limit = Wlimit)
```

These values are taken from the problem statement in [RossettaCode Knapsack: 0-1](https://rosettacode.org/wiki/Knapsack_problem/0-1)

## Running Large W algorithm

Given weights, values, and a limit, the large W algorithm outputs a list of 1s and 0 corresponding to putting the respective item in the list in the knapsack (output of 1) or leaving said item out (output of 0). From such a list, we can output the final collection of items in the knapsack. To run the algorithm, execute the following code after defining the item list above.

```
soln = KP_camping.largeN_algorithm()
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
This result reproduces the solution given in [RossettaCode Knapsack: 0-1](https://rosettacode.org/wiki/Knapsack_problem/0-1). To apply the algorithm to other knapsack problem instances, replace the values, weights, and limit in the [`example.ipynb`](https://github.com/mowillia/largeNKP/blob/main/example.ipynb) file with the respective quantities for your chosen instance. 

## Plotting potential function

The potential function for the zero-one knapsack problem is 
```
FN_zero_one = lambda z, weights, values, limit, T: - limit*np.log(z)-np.log(1-z) + np.sum(np.log(1+z**(weights)*np.exp(values/T)))
```
This function gives a continuous representation of the standard discrete optimization objective. If the function has a local minimum, then the large N algorithm can solve the knapsack problem. This minimum depends on temperature, and as the temperature is lowered the minimum better defines an optimal solution for the knapsack problem. To plot the potential function for the above instance, execute the following code. 

```
KP_camping.plot_potential(T = 1.5)
>>>
```
<p align="center">
<img align = "middle" src = "https://user-images.githubusercontent.com/8810308/111629285-84221580-87c7-11eb-9486-6828c446040d.png" width = "40%">
</p>

## Plotting total value as a function of temperature

To plot the calculated total value as a function of temperature, execute the following code

```
KP_camping.plot_value_vs_temp(temp_low=1.0, temp_high = 60.0)
>>>
```
<p align="center">
<img align = "middle" src = "https://user-images.githubusercontent.com/8810308/111698215-f5d08280-880c-11eb-8361-330a35755881.png" width = "40%">
</p>

This plot uses the nonzero-temperature algorithm to compute the object composition for each temperature between `temp_low` and `temp_high` and then computes the total value of that composition. We see that as temperature is lowered, the total value of the collection increases until it presumably has reached its maximum value. 

## Algorithm comparison plots

In the original paper, we compare the performance of various classic knapsack problem algorithms to the proposed algorithm. The algorithms we compare are

- **Brute Force**(`brute_force`): Involves listing all possible combinations of items, computing the total weights and total values of each combination and selecting the combination with the highest value with a weight below the stated limit. 

- **Dynamical Programming Solution**(`knapsack_dpV`):  Standard recursive solution to the problem which involves storing sub-problem solutions in matrix elements

- **Fully Polynomial Time Approximate Solution (FPTAS)**(`fptas`):  Algorithm that is polynomial time in the number of elements and which has a tunable accuracy

- **Greedy Algorithm**(`greedy`):  Involves computing the ratio of weights to volumes for each object and filling in the collection until the max weight is reached. 

- **Simulated Annealing**(`simannl_knapsack`):   Involves representing the system computationally as a statistical physics one and then "annealing" the system to low temperatures. 

- **Large W Algorithm (T=0):**(`largeN_algorithm`)  Algorithm proposed in paper; based on statistical physics representation of the system at $T=0$

- **Large W Algorithm (T\=0):**(`largeN_algorithm`)  Algorithm proposed in paper; on statistical physics representation of the system at $T\neq0$

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
algo_name_dict = {'Brute': KP_camping.brute_force,
                  'DP': KP_camping.knapsack01_dpV,
                  'FPTAS': KP_camping.fptas,
                  'Greedy': KP_camping.greedy,
                  'Annealing': KP_camping.simann_knapsack,
                  'Large W (T=0)': KP_camping.largeW_algorithm,
                  'Large W (T/=0)': KP_camping.largeW_algorithm}

# dictionary of algorithm names and results
results_name_dict = defaultdict(lambda: list())
```
Running algorithm and creating table of results
```
for name, func in algo_name_dict.items():
    start_clock = time.time()
    if name == 'Large W (T/=0)':
        soln = func(T=1.0)
    else:    
        soln = func()    
    
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
Stopping annealing because error tolerance was reached
+----------------+---------+-----------+--------------+
| Algorithm      |   Value |   Weight  |   Time (sec) |
+================+=========+===========+==============+
| Brute          |    1030 |       396 |     18.2886  |
+----------------+---------+-----------+--------------+
| DP             |    1030 |       396 |      0.00418 |
+----------------+---------+-----------+--------------+
| FPTAS          |    1030 |       396 |      0.00532 |
+----------------+---------+-----------+--------------+
| Greedy         |    1030 |       396 |      8e-05   |
+----------------+---------+-----------+--------------+
| Annealing      |     857 |       396 |      0.07015 |
+----------------+---------+-----------+--------------+
| Large W (T=0)  |    1030 |       396 |      0.0603  |
+----------------+---------+-----------+--------------+
| Large W (T/=0) |    1030 |       396 |      0.00362 |
+----------------+---------+-----------+--------------+

```
We see that both large W algorithms yield the correct result, though they are not the fastest algorithms for this instance. 

## Jupyter notebooks associated with paper

The notebooks that reproduce the figures and tables in the paper are as follows


**Main Notebooks**
- [`potential_landscape.ipynb`](https://github.com/mowillia/largeNKP/blob/main/potential_landscape.ipynb): Reproduces Figure 3(a); Runs in < 1 minute
- [`total_value_vs_temperature.ipynb`](https://github.com/mowillia/largeNKP/blob/main/total_value_vs_temperature.ipynb): Reproduces Figure 3(b); Runs in < 1 minute
- [`algorithm_comparisons.ipynb`](https://github.com/mowillia/largeNKP/blob/main/algorithm_comparisons.ipynb): Reproduces Figure 4; Runs in 15 minutes
- [`linear_gamma.ipynb`](https://github.com/mowillia/largeNKP/blob/main/linear_gamma.ipynb): Reproduces Figure 2; Runs in < 1 minute

**Additional Notebooks**
- [`example.ipynb`](https://github.com/mowillia/largeNKP/blob/main/example.ipynb): Not referenced in paper; Example file for the current readme.
- [`root_finding_algos.ipynb`](https://github.com/mowillia/largeNKP/blob/main/root_finding_algos.ipynb): Not referenced in paper; Notebook for testing various root-finding algorithms
- [`dynamic_partition_function.ipynb`](https://github.com/mowillia/largeNKP/blob/main/dynamic_partition_function.ipynb): Referenced in Section 2; Shows how to use dynamic programming to compute the partition function and solution of KP
- [`dynamic_partition_comparisons.ipynb`](https://github.com/mowillia/largeNKP/blob/main/dynamic_partition_comparisons.ipynb): Referenced in Section 2; Compares the accuracy and computation time of the exact partition function solution to the KP and the dynamic programing solution to the KP


## Acknowledgments
Work completed in [Jellyfish Research](https://jellyfish.co/).



