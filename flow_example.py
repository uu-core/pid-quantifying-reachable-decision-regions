import numpy as np
from pid_implementation import *

'''
An implementation of the Partial Information Decomposition (PID) as presented by Mages and Rohner (2023)

Implementation of the Information Flow Analysis example shown in Section 4.2. (Figure 7+8)

References:
	Mages, T.; Rohner, C. Decomposing and Tracing Mutual Information by Quantifying Reachable Decision Regions. Entropy 2023
'''

'''
Construct setting: 
  - inDist: input distribution
  - Markov chain (k: initial channel T->V, gX: bitflips on wires, cX: computation of gates) 
        T -> V  = k
        T -> Q  = k@g1@c1
        T -> R  = k@g1@c1@g2@c2
        T -> Th = k@g1@c1@g2@c2@g3@c3@g4
  - origin distribution:
        A B C T Pr
        0 0 0 0 1/8
        0 0 1 1 1/8
        0 1 0 1 1/8
        0 1 1 2 1/8
        1 0 0 1 1/8
        1 0 1 2 1/8
        1 1 0 2 1/8
        1 1 1 3 1/8
'''
def bitflipMatrix(p1,p2,p3):
    bitflips = np.array([[-1.0]*8]*8)
    states = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    for i,input in enumerate(states):
        for o,output in enumerate(states):
            bitflips[i,o] = (p1 if input[0] != output[0] else (1-p1))*(p2 if input[1] != output[1] else (1-p2))*(p3 if input[2] != output[2] else (1-p3))
    return bitflips

# (0,0),(0,1),(1,0),(1,1), noted as (Sum,Carry)
inDist = np.array([1/8, 3/8, 3/8, 1/8])
# (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), ordered by index (V1,V2,V3)
k = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1/3, 0, 1/3, 1/3, 0],
              [0, 1/3, 1/3, 0, 1/3, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])
g1 = bitflipMatrix(0.005,0.007,0.007)
c1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0]])
g2 = bitflipMatrix(0.003,0.0,0.005)
c2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0]])
g3 = bitflipMatrix(0.003,0.003,0.0)
c3 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 1],
               [0, 0, 0, 1]])
pg4 = 0.001
g4 = np.array([[1-pg4,pg4,0,0],[pg4,1-pg4,0,0],[0,0,1-pg4,pg4],[0,0,pg4,1-pg4]])
#print(k@c1@c2@c3) <- identity matrix T=Th without bit-flips
''' defining channels '''
V = k
Q = V@g1@c1
R = Q@g2@c2
Th = R@g3@c3@g4

"extract variable 1"
extr1 = np.array([[1,0],
                  [1,0],
                  [1,0],
                  [1,0],
                  [0,1],
                  [0,1],
                  [0,1],
                  [0,1]])
"extract variable 3"
extr3 = np.array([[1,0],
                  [0,1],
                  [1,0],
                  [0,1],
                  [1,0],
                  [0,1],
                  [1,0],
                  [0,1]])
"extract variables 12"
extr12 = np.array([[1,0,0,0],
                   [1,0,0,0],
                   [0,1,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,1,0],
                   [0,0,0,1],
                   [0,0,0,1]])
"extract variables 23"
extr23 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1],
                   [1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
channelSets={'T': {str(['S','C']): np.identity(4),
                    str(['S']): np.array([[1,0],[1,0],[0,1],[0,1]]),
                    str(['C']): np.array([[1,0],[0,1],[1,0],[0,1]])},
            'V': 
                {str(['V12','V3']): V,
                 str(['V12']): V@extr12,
                 str(['V3']): V@extr3},
             'Q': 
                {str(['Q12','Q3']): Q,
                 str(['Q12']): Q@extr12,
                 str(['Q3']): Q@extr3},
             'R': 
                {str(['R1','R23']): R,
                 str(['R23']): R@extr23,
                 str(['R1']): R@extr1},
             'Th': {str(['Sh','Ch']): Th,
                    str(['Sh']): Th@np.array([[1,0],[1,0],[0,1],[0,1]]),
                    str(['Ch']): Th@np.array([[1,0],[0,1],[1,0],[0,1]])}
             }
Tvars = ['S','C']
Vvars = ['V12','V3']
Qvars = ['Q12','Q3']
Rvars = ['R1','R23']
Thvars = ['Sh','Ch']

'''
    Construct decomposition for each step
'''
channel_pid(Tvars,inDist,channelSets['T'])
print()
channel_pid(Vvars,inDist,channelSets['V'])
print()
channel_pid(Qvars,inDist,channelSets['Q'])
print()
channel_pid(Rvars,inDist,channelSets['R'])
print()
channel_pid(Thvars,inDist,channelSets['Th'])

'''
    Construct flow for each step
'''
flow_analysis(Tvars,Vvars,channelSets['T'],channelSets['V'],inDist)
flow_analysis(Vvars,Qvars,channelSets['V'],channelSets['Q'],inDist)
flow_analysis(Qvars,Rvars,channelSets['Q'],channelSets['R'],inDist)
flow_analysis(Rvars,Thvars,channelSets['R'],channelSets['Th'],inDist)