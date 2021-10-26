"""
File: problem/gridworld.py
Date: 10.07.2021 (created)
      10.15.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    A MDP model of grid-world problem.
"""

import common.simulator as sim

def coordPlus(p, q):
    assert isinstance(p, tuple)
    assert isinstance(q, tuple)
    assert len(p) == len(q)
    return tuple(p[i] + q[i] for i in range(len(p)))

class gwTP(sim.Transition):
    def __init__(self, S, A, p_e):
        super().__init__(S, A)
        self.pe = p_e # probability of error

    def __call__(self, s, a):

        def aToCoord(a):
            if a == 'up':
                return (0, 1)
            if a == 'down':
                return (0, -1)
            if a == 'left':
                return (-1, 0)
            if a == 'right':
                return (1, 0)
            if a == 'stay':
                return (0, 0)

        sn = coordPlus(s, aToCoord(a))
        nb = [ coordPlus(s, aToCoord(a_)) for a_ in self.A if a_ != a ] # neighbor states other than sn
        nb = [ i for i in nb if i in self.S ] # exclude forbidden states

        if sn in self.S:
            return [1.0 - len(nb) * self.pe if (i == sn) else
                   (self.pe if (i in nb) else 0.0)
                    for i in self.S ]
        else:
            return [ 1.0 if (i == s) else 0.0 for i in self.S ]

'''
class gwR(simulator.Transition):
    def __init__(self, S, A):
        super().__init__(S, A)
        self.rd = 11
        self.rs = 10
        self.rw = -5

    def __call__(self, sn, s, a):
        # reward R(s' | s, a)
        def sToR(s):
            if s == (2, 2):
                return self.rd
            if s == (2, 0):
                return self.rs
            if s[0] == 4:
                return self.rw

        return [ sToR(i) for i in self.S ]
'''

def gwR(sn, s, a):
    if sn == (2, 2):
        return 1
    if sn == (2, 0):
        return 10
    if sn[0] == 4:
        return -5
    else:
        return 0

# system parameters
m = 5 # range of x
n = 5 # range of y
B = [(1, 1), (2, 1), (1, 3), (2, 3)]
S = [(i, j) for j in range(n) for i in range (m) if (i, j) not in B] # coordinates in a m-by-n rectangle
A = ['left', 'up', 'right', 'down', 'stay']
#A = ['↑', '↓', '←', '→', '．']
P = gwTP(S, A, p_e = 0.1)

# task parameters
H = -1
gamma = 0.8
R = gwR

# bundle
env = {
    'S': S,
    'A': A,
    'P': P,
    'O': None,
    'R': R,
    'H': H,
    'gamma': gamma,
    'dim': (m, n),
    'B': B
}

#
