# Implementation: Decomposing and Tracing Mutual Information by Quantifying Reachable Decision Regions

This repository provides an implementation of the method and examples in:
-  	Mages, T.; Rohner, C. Decomposing and Tracing Mutual Information by Quantifying Reachable Decision Regions. Entropy 2023

## Overview
- `pid_implementation.py`: provides the implementation of the presented method and flow analysis
- `pid_examples.py`: provides the decompositions examples of Appendix E
- `flow_example.py`: provides the flow analysis example of Section 4.2

## Example usage
```
import numpy as np
import pandas as pd
from pid_implementation import pid

'''
 PID based on a joint distribution
'''
# define joint distribution with column 'Pr'
jointDistribution = pd.DataFrame(
    #          V1 V2 V3  T   Pr 
    np.array([[ 0, 0, 0, 0, 1/4],
              [ 0, 1, 1, 1, 1/4],
              [ 1, 0, 1, 2, 1/4],
              [ 1, 1, 0, 3, 1/4]]), 
              columns=['V1', 'V2', 'V3', 'T', 'Pr'])
# compute pid
pid(['V1', 'V2', 'V3'], 'T', jointDistribution, normalize=False)
```
## Running examples
- `python pid_examples.py`: computes typical examples and prints the results 
- `python flow_example.py`: computes a flow analysis and prints the results
 
## Note
- The redundancy lattice grows quickly with the number of variables. For larger numbers of visible variables, it is beneficial to search the lattice rather than fully computing it (e.g. repeatedly splitting the variables into 2-3 sets for searching the desired interactions).
