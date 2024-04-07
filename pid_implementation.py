'''
An implementation of the Partial Information Decomposition (PID) as presented in Mages and Rohner (2023)

References:
	Mages, T.; Rohner, C. Decomposing and Tracing Mutual Information by Quantifying Reachable Decision Regions. Entropy 2023
	Finn, C., 2019. A New Framework for Decomposing Multivariate Information. Ph.D. thesis. University of Sydney.
	Williams, P.L., Beer, R.D., 2010. Nonnegative decomposition of multivariate information. arXiv:1004.2515.

Structure:
	1. The PID Framework: 
			definitions of the redundancy lattice (implements Section 2.2)
	2. The Partial Information Decomposition:
			implements the Blackwell meet/join on binary channels and the valuation function "f_p" (implements Section 3.2)
	3. The Partial Information Decomposition - given joint distributions: 
			implements the PID given the joint distribution of variables (implements Section 4.1)
	4. The Partial Information Decomposition - given channel and input distributions: 
			implements the PID given a channel and input distribution (implements Section 4.1)
	5. The Information Flow Analysis: 
			implementation for tracing partial information components (implements Section 4.2)

	- The example decompositions of Appendix E are shown in "pid_examples.py"
	- The flow analysis example of Section 4.2 is implemented in "flow_example.py"
	- For the computation of larger examples, the implementation should be optimized. 
	  For example, the provided recursive implementation computes the same atom multiple times.

Main functions:
	- `pid(predictors, predictors, target, normalize=True,printPrecision=4)`, see example usage below or in 'pid_example.py'. It performs the decomposition for a given joint distribution. It takes a pandas DataFrame containing the joint distribution `pdProbabilities`. The `predictors` are the visible variables (list of column names), `target` corresponds to the column name of the target variable. The probabilities shall be stored in a column named `'Pr'`.

	- `channel_pid(predictors, targetDistr, channelDict, normalize=True,printPrecision=4)`, see example usage in 'flow_example.py'. It performs the decomposition given a channel and input distribution. It takes a list of predictor names which will be used as keys in the `channelDict`, a numpy array of the input distribution (`targetDistr`) and a dictionary `channelDict` that returns a channel (numpy array) given list of predictor variables (power-set of `predictors` converted to string are the assumed keys). For example: `predictors = ['V1','V2']`, then the dictionary should contain the keys `str(['V1','V2'])`, `str(['V1'])` and `str(['V2'])`.
	
	- `flow_analysis(predictorsA, predictorsB, channelSetA channelSetB, targetDistr, normalize=True, printPrecision=5)`, see example usage in 'flow_example.py'. It assumes the same format as 'channel_pid' for the Markov chain T -> A -> B.
'''

from itertools import combinations, product
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

logBase = 2
precision = 14

###############################################################
##### 1. The PID Framework
###############################################################

''' predictors: list of predictor variables e.g. ['V1', 'V2', 'C3'] '''
def P1(predictors): # Sources: power set without the empty set
    return [list(source) for r in range(1,len(predictors)+1) for source in combinations(predictors, r)]

#--> Subset for axiom 2: Monotonicity             <--#
''' A, B: sources (list of predictors) e.g. A=[['V1', 'V2'],['C3']] '''
def subset(A,B): # True if A subset of B 
	return all([a in B for a in A])

#--> PI-atom queality:                            <--#
''' alpha, beta: collections of sources e.g. alpha=[['V1', 'V2'],['C3']] '''
def equalAtom(alpha, beta):
	return all([any([set(a)==set(b) for b in beta]) for a in alpha] + [any([set(a)==set(b) for a in alpha]) for b in beta])

#--> Equation 4      of Williams and Beer (2010)  <--#
#--> Equation 2.17   of Thesis Finn (2019)        <--#
''' predictors: list of predictor variables e.g. ['V1', 'V2', 'C3'] '''
def A(predictors): # all relevant collection of sources from the predictors
	return [x for x in P1(P1(predictors)) if not any([subset(A1,A2) for (A1, A2) in combinations(x,2)])]

#--> Equation 5      of Williams and Beer (2010)  <--#
#--> Equation 2.18   of Thesis Finn (2019)        <--#
''' alpha,beta: collections of sources e.g. alpha=[['V1', 'V2'],['C3']] '''
def leq(alpha,beta): # less or equal for two collections of sources
	return all([any([subset(A,B) for A in alpha]) for B in beta])



###############################################################
##### 2. The Partial Information Decomposition
###############################################################


''' helper functions for Blackwell meet/joint '''
def channelToPoints(k):
	points = sorted(k.T.tolist(),key=lambda x: x[1]/x[0] if x[0] != 0 else 10**15, reverse=True) # sort entries by likelihood ratio
	# accumulate components to get the curve
	curve = [np.array([0,0])]
	for p in points:
		curve.append((np.array(p) + curve[-1]))
	return np.array(curve)

def hullToChannel(hull_points):
	cornerPoints = sorted(hull_points,key=lambda x: tuple(x)) 
	channel, init = [], np.array([0,0])
	for x in cornerPoints:
		if np.any(x != init):
			channel.append(np.array(x)-init)
			init = np.array(x)
	return np.array(channel).T

def myConvexHull(points):
	points_unq = np.unique(np.round(points,precision-3),axis=0)
	if all([abs(a-b) < 10**(-precision) for a,b in points_unq.tolist()]):
		return np.array([[0,0],[1,1]])
	else:
		return np.unique(np.vstack([points_unq[s] for s in ConvexHull(points_unq).simplices]),axis=0)

'''
	returns the Blackwell-joint element of 'k1' and 'k2' (return: k1 u k2)
'''
def BlackwellJoint(k1,k2):
	k1,k2=np.round(k1,precision),np.round(k2,precision)
	assert np.all(k1 >= 0.0) and np.all(k2 >= 0.0), f'invalid channel {k1} or {k2}'
	return hullToChannel(myConvexHull(np.vstack((channelToPoints(k1),channelToPoints(k2)))).tolist())

''' 
	meetList: list of channels (numpy arrays) to take the meet of: k1 ^ k2 ^ k3 is given as [k1, k2, k3]
	p: parameter of the valuation function
'''
def valuationf(meetList,p):
	# construct matrix
	def recursiveMeetExpansion(meetL):
		match meetL:
			case []: 
				print('WARNING: Empty PIatom has been interpreted as bottom element. Unless explicilely used, this should not be reached.')
				return np.array([[1],[1]])
			case [head]:
				return head
			case [head, *tail]:
				# sum-rule with distributivity: a^b^c = [a, b^c, -((a v b)^(a v c))]
				return np.hstack(( head , recursiveMeetExpansion(tail), -recursiveMeetExpansion([BlackwellJoint(head,x) for x in tail]) ))
	matrix = recursiveMeetExpansion(meetList)
	return sum([(x*np.log(x/(p*x+(1-p)*y))/np.log(logBase) if (x != 0.0 and x != y) else 0) for (x,y) in np.round(matrix.T,precision)])


####################################################################################
##### 3. The Partial Information Decomposition - given joint distributions
#####################################################################################

'''
example: 
> pdProbabilitiesOrig = pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,1,1/4],[1,1,0,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])
> toBinaryChannel(['V1','V1'],'T',1.0,pdProbabilitiesOrig)
return: numpy array of pointwise channel
'''
def toBinaryChannel(preds,target,t,pdProbabilities): 
	# generate binary input channel for state t
	pdPointwise = pdProbabilities[preds+[target,'Pr']].copy(deep=True)      # copy to leave original unchanged
	pdPointwise[target] = pdPointwise[target].apply(lambda x: x == t)       # convert T to binary variable (True: x==t or False: x!=t)
	pdPointwise = pdPointwise.groupby(preds+[target],as_index=False).sum()  # combine identical states
	p = pdPointwise[pdPointwise[target] == True]['Pr'].sum()                # compute p = P(T==t)
	pdPointwise.loc[pdPointwise[target] == True, 'Pr'] /= p                 # get conditional probability for T==t: x/p
	pdPointwise.loc[pdPointwise[target] == False, 'Pr'] /= (1-p)            # get conditional probability for T!=t: x/(1-p)
	return pdPointwise.pivot_table(index=[target],columns=set(preds),values='Pr',fill_value=0).sort_index(ascending=False).to_numpy()

''' PIatom: to be computed, collection of sources e.g. [['V1', 'V2'], ['V1']]
	target: target variable symbol e.g. 'T'
	pdProbabilities: dataframe of joint probabilities
		- columns: predictors+[target]+['Pr']  <- 'Pr' storing the probability 
		- e.g. pd.DataFrame(np.array([[0,0,0,0.1],[0,0,1,0.15],[0,1,0,0.05], ...]), columns=['V1', 'V2', 'T', 'Pr']) 
'''
def Ihat(PIatom,target,pdProbabilities):
	Pt = lambda state: pdProbabilities.groupby([target]).sum()['Pr'][state]
	# compute expected value of valuation results (combining pointwise lattices)
	return sum([Pt(t)*valuationf([toBinaryChannel(preds,target,t,pdProbabilities) for preds in PIatom], Pt(t)) for t in set(pdProbabilities[target])])

''' visibleVar: variable to compute the entropy of e.g. 'T'
	pdProbabilities: dataframe of joint distribution
		- columns: predictors+[target]+['Pr']  <- 'Pr' storing the probability 
		- e.g. pd.DataFrame(np.array([[0,0,0,0.1],[0,0,1,0.15],[0,1,0,0.05], ...]), columns=['V1', 'V2', 'T', 'Pr']) 
'''
def entropy(visibleVar,pdProbabilities):
	Pt = pdProbabilities.groupby(visibleVar).sum()['Pr'].to_numpy()
	return sum([ (-x*np.log(x)/np.log(logBase) if x != 0.0 else 0) for x in Pt])

### Perform PID analysis on DataFrame
''' predictors: list of predictor variables e.g. ['V1', 'V2'] 
	target: target variable symbol e.g. 'T'
	pdProbabilities: dataframe of joint distribution
		- columns: predictors+[target]+['Pr']  <- 'Pr' storing the probability 
		- e.g. pd.DataFrame(np.array([[0,0,0,0.1],[0,0,1,0.15],[0,1,0,0.05], ...]), columns=['V1', 'V2', 'T', 'Pr']) 
'''
def pid(predictors, target, pdProbabilities,normalize=True,printPrecision=4):
	print(f'\nWARNING: The results are {"" if normalize else "_not_ "}normalized.')
	# check that sources and target match the provided dataframe
	assert set(predictors+[target, 'Pr']) == set(pdProbabilities.columns)
	As = A(predictors)
	resultCumulative = {}
	resultPartial = {}
	## compute cumulative result
	for PIatom in As:
		resultCumulative[str(PIatom)] = Ihat(PIatom,target,pdProbabilities)/(entropy(target,pdProbabilities) if normalize else 1)
	## compute partial contributions
	def partialComputation(atom):
		strictDownset = lambda atom: [x for x in As if (leq(x,atom) and not leq(atom,x))]
		return resultCumulative[str(atom)] - sum([partialComputation(x) for x in strictDownset(atom)])
	for PIatom in As:
		resultPartial[str(PIatom)] = partialComputation(PIatom)
	for k in resultCumulative:
		print(f'{k}: {round(resultCumulative[k],printPrecision)} ({round(resultPartial[k],printPrecision)})') 
	#return resultCumulative, resultPartial


####################################################################################
##### 4. The Partial Information Decomposition - given channel and input distributions
#####################################################################################
def channel_toBinary(channel, index, inputDistr):
    jointDistr = (np.array([inputDistr]).T)*channel
    return np.vstack((channel[index,:],np.sum(np.vstack((jointDistr[:index,:],jointDistr[index+1:,:])),axis=0)/(1 - inputDistr[index]))) 

def channel_Ihat(PIatom,targetDistr,channelDict):
	return sum([pt*valuationf([channel_toBinary(channelDict[str(preds)],i,targetDistr) for preds in PIatom], pt) for i,pt in enumerate(targetDistr)])

def channel_pid(predictors, targetDistr, channelDict, normalize=True,printPrecision=4):
    print(f'\nWARNING: The results are {"" if normalize else "_not_ "}normalized.')
    As = A(predictors)
    resultCumulative = {}
    resultPartial = {}
    ## compute cumulative result
    for PIatom in As:
        resultCumulative[str(PIatom)] = channel_Ihat(PIatom,targetDistr,channelDict)/(sum([ (-x*np.log(x)/np.log(logBase) if x != 0.0 else 0) for x in targetDistr]) if normalize else 1)
    ## compute partial contributions
    def partialComputation(atom):
        strictDownset = lambda atom: [x for x in As if (leq(x,atom) and not leq(atom,x))]
        return resultCumulative[str(atom)] - sum([partialComputation(x) for x in strictDownset(atom)])
    for PIatom in As:
        resultPartial[str(PIatom)] = partialComputation(PIatom)
    for k in resultCumulative:
        print(f'{k}: {round(resultCumulative[k],printPrecision)} ({round(resultPartial[k],printPrecision)})') 


####################################################################################
##### 5. The Information Flow Analysis
####################################################################################
def flow_AtoB(predictorsA, predictorsB, channelDictA, channelDictB, atomA, atomB, targetDistr, normalize=True):
    strictDownset = lambda atom,predictors: [x for x in A(predictors) if (leq(x,atom) and not leq(atom,x))]
    ## compute cumulative result
    entropyT = sum([ (-x*np.log(x)/np.log(logBase) if x != 0.0 else 0) for x in targetDistr]) if normalize else 1
    channelDict = channelDictA | channelDictB
    def JdeltaA(_atomA,_atomB):
        downA = strictDownset(_atomA,predictorsA)
        return channel_Ihat(_atomA+_atomB,targetDistr,channelDict)/entropyT - sum([JdeltaA(x,_atomB) for x in downA])
    def JdeltaAB(_atomA,_atomB):
        return JdeltaA(_atomA,_atomB) - sum([JdeltaAB(_atomA,x) for x in strictDownset(_atomB,predictorsB)])
    return JdeltaAB(atomA, atomB)

def flow_analysis(predictorsA, predictorsB,channelSetA,channelSetB,targetDistr,normalize=True,printPrecision=5):
    print(f'\nWARNING: The results are {"" if normalize else "_not_ "}normalized.')
    As = A(predictorsA)
    Bs = A(predictorsB)
    for (x,y) in product(*[As,Bs]):
        print(f'{x} -> {y}: {round(flow_AtoB(predictorsA,predictorsB,channelSetA,channelSetB,x,y,targetDistr,normalize=normalize),printPrecision)}')


