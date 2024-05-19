# Implementation: Decomposing and Tracing Mutual Information by Quantifying Reachable Decision Regions

This repository provides ...
1. an implementation of the method and all examples/analyses in:

	_Mages, T.; Rohner, C. Decomposing and Tracing Mutual Information by Quantifying Reachable Decision Regions. Entropy 2023. [https://doi.org/10.3390/e25071014](https://doi.org/10.3390/e25071014)_
    
2. a non-negative decomposition of mutual information on the redundancy lattice and its corresponding information-flow analysis for Markov chains.

## Update 2: New implementation
**_A new implementation is available:_** [uu-core/pid-blackwell-specific-information](https://github.com/uu-core/pid-blackwell-specific-information)

It provides an implementation for the partial information decomposition and information flow-analysis of any f-information measure on both a redundancy and synergy lattice. The decomposition and tracing of Rényi-information can be obtained as transformation as described in the corresponding publication: 
- _Mages, T.; Anastasiadi, E.; Rohner, C. Non-Negative Decomposition of Multivariate Information: From Minimum to Blackwell-Specific Information. Entropy 2024. [https://doi.org/10.3390/e26050424](https://doi.org/10.3390/e26050424)_

## Update 1: Available in `dit`
The decomposition is now also available in the [dit Python package for discrete information theory](https://github.com/dit/dit) under the name `PID_RDR` (Partial Information Decomposition - Reachable Decision Regions). After installing the package from its git-repository, it can be used as shown below:

```
import dit

# Example XOR
d = dit.Distribution(['000', '011', '101', '110'], [1/4]*4)
print(dit.pid.PID_RDR(d))

# Example XOR CAT
d = dit.pid.distributions.trivariates['xor cat']
print(dit.pid.PID_RDR(d))
```

## Overview
- `pid_implementation.py`: provides the implementation of the presented method and flow analysis
- `pid_examples.py`: provides the decompositions examples of Appendix E
- `flow_example.py`: provides the flow analysis example of Section 4.2

_**Requirements:**_ [NumPy](https://numpy.org/install/), [SciPy](https://scipy.org/install/) and [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html).

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
