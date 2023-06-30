import numpy as np
import pandas as pd
from pid_implementation import *

'''
An implementation of the Partial Information Decomposition (PID) as presented by Mages and Rohner (2023)

The decomposition comparison of Appendix E (Table A1-A8) in Mages and Rohner (2023).
We use the examples of Finn and Lizier (2018) since they provided an extensive discussion for their motivation and interpretation.

References:
	Mages, T.; Rohner, C. Decomposing and Tracing Mutual Information by Quantifying Reachable Decision Regions. Entropy 2023
	Williams, P.L., Beer, R.D., 2010. Nonnegative decomposition of multivariate information. arXiv:1004.2515.
	Finn, C., Lizier, J.T., 2018. Pointwise partial information decomposition using the specificity and ambiguity lattices. Entropy 20. doi:10.3390/e20040297.
	J. Rauh, N. Bertschinger, E. Olbrich and J. Jost, Reconsidering unique information: Towards a multivariate information decomposition, 2014 IEEE International Symposium on Information Theory, 2014, pp. 2232-2236, doi: 10.1109/ISIT.2014.6875230.
'''

predictors = ['V1', 'V2']
target = 'T'

# Example of Section 3.1 of Mages and Rohner (2023) to highlight difference from Williams and Beer (2010)
pdProbabilitiesOrig = pd.DataFrame(np.array([[0,0,0,0.0625],[0,0,1,0.3],[1,0,0,0.0375],[1,0,1,0.05],[0,1,0,0.1875],[0,1,1,0.15],[1,1,0,0.2125]]), columns=['V1', 'V2', 'T', 'Pr'])
pid(predictors, target, pdProbabilitiesOrig)
print(f'Target entropy: {entropy(target, pdProbabilitiesOrig)}')

# Example XOR Figure 4 of Finn and Lizier (2018)
pdProbabilitiesOrig = pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,1,1/4],[1,1,0,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])
pid(predictors, target, pdProbabilitiesOrig)
print(f'Target entropy: {entropy(target, pdProbabilitiesOrig)}')


# Example PwUnq Figure 5 of Finn and Lizier (2018)
pdProbabilitiesOrig = pd.DataFrame(np.array([[0,1,1,1/4],[1,0,1,1/4],[0,2,2,1/4],[2,0,2,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])
pid(predictors, target, pdProbabilitiesOrig)
print(f'Target entropy: {entropy(target, pdProbabilitiesOrig)}')


# Example RdnErr Figure 6 of Finn and Lizier (2018)
pdProbabilitiesOrig = pd.DataFrame(np.array([[0,0,0,3/8],[1,1,1,3/8],[0,1,0,1/8],[1,0,1,1/8]]), columns=['V1', 'V2', 'T', 'Pr'])
pid(predictors, target, pdProbabilitiesOrig)
print(f'Target entropy: {entropy(target, pdProbabilitiesOrig)}')


# Example Tbc Figure 7 of Finn and Lizier (2018)
pdProbabilitiesOrig = pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,2,1/4],[1,1,3,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])
pid(predictors, target, pdProbabilitiesOrig)
print(f'Target entropy: {entropy(target, pdProbabilitiesOrig)}')


# Example Unq Figure A2 of Finn and Lizier (2018)
pdProbabilitiesOrig = pd.DataFrame(np.array([[0,0,0,1/4],[0,1,0,1/4],[1,0,1,1/4],[1,1,1,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])
pid(predictors, target, pdProbabilitiesOrig)
print(f'Target entropy: {entropy(target, pdProbabilitiesOrig)}')


# Example AND Figure A3 of Finn and Lizier (2018)
pdProbabilitiesOrig = pd.DataFrame(np.array([[0,0,0,1/4],[0,1,0,1/4],[1,0,0,1/4],[1,1,1,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])
pid(predictors, target, pdProbabilitiesOrig)
print(f'Target entropy: {entropy(target, pdProbabilitiesOrig)}')


predictors = ['V1', 'V2', 'V3']
target = 'T'

# Example IdentityCounterExample Theorem 2: Rauh et al. (2014)
# Example Tbep Figure A1 of Finn and Lizier (2018)
pdProbabilitiesOrig = pd.DataFrame(np.array([[0,0,0,0,1/4],[0,1,1,1,1/4],[1,0,1,2,1/4],[1,1,0,3,1/4]]), columns=['V1', 'V2', 'V3', 'T', 'Pr'])
pid(predictors, target, pdProbabilitiesOrig)
print(f'Target entropy: {entropy(target, pdProbabilitiesOrig)}')


'''
# Comparing methods:

import dit

# Highglight difference from Williams and Beer (Example of Motivation 3.1)
d = dit.Distribution(['000','001','100','101','010','011','110'], [0.0625,0.3,0.0375,0.05,0.1875,0.15,0.2125])
print(dit.pid.PID_BROJA(d))
print(dit.pid.PID_WB(d))
print(dit.pid.PID_PM(d))

# Example XOR
d = dit.Distribution(['000', '011', '101', '110'], [1/4]*4)
print(dit.pid.PID_BROJA(d))
print(dit.pid.PID_WB(d))
print(dit.pid.PID_PM(d))

# Example PwUnq
d = dit.Distribution(['010', '100', '021', '201'], [1/4]*4)
print(dit.pid.PID_BROJA(d))
print(dit.pid.PID_WB(d))
print(dit.pid.PID_PM(d))

# Example RdnErr
d = dit.Distribution(['000', '111', '010', '101'], [3/8,3/8,1/8,1/8])
print(dit.pid.PID_BROJA(d))
print(dit.pid.PID_WB(d))
print(dit.pid.PID_PM(d))


# Example Tbc
d = dit.Distribution(['000', '011', '102', '113'], [1/4]*4)
print(dit.pid.PID_BROJA(d))
print(dit.pid.PID_WB(d))
print(dit.pid.PID_PM(d))

# Example Unq
d = dit.Distribution(['000', '010', '101', '111'], [1/4]*4)
print(dit.pid.PID_BROJA(d))
print(dit.pid.PID_WB(d))
print(dit.pid.PID_PM(d))

# Example And
d = dit.Distribution(['000', '010', '100', '111'], [1/4]*4)
print(dit.pid.PID_BROJA(d))
print(dit.pid.PID_WB(d))
print(dit.pid.PID_PM(d))

# Example Tbep
d = dit.Distribution(['0000', '0111', '1012', '1103'], [1/4]*4)
print(dit.pid.PID_BROJA(d))
print(dit.pid.PID_WB(d))
print(dit.pid.PID_PM(d))

'''
