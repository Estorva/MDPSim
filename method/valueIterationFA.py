"""
File: valueIterationFA.py
Date: 10.14.2021 (created)
      10.15.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    Implementation of value iteration with function approximation.
"""

import random
import numpy as np
from math import sqrt as sqrt

FLOAT_MIN = -1.0e100

def solve(env: dict, H: int, gamma: float, thr: float = 0.2):
    # V-bar*(s) = 0 for s in S-bar
    # for i in 1..H
    #     D <- 0
    #     for each s in S-bar
    #         V-bar*(s) <- max_{a in A-bar} Sum_{s' in S-bar}
    #             P(s'|s, a) [r(s'|s, a) + gamma * V-hat*_i(s')]
    #         V-hat*_{i+1}(s) <- sum_j P(xi_j, s) V-bar_i(xi_j)
    #         D <- max(D, |v - V(s)|)
    #     if D < threshold: break

    S = env['S']
    A = env['A']
    P = env['P']
    O = env['O']
    R = env['R']
    Phi = env['Phi']
    nS = len(S)
    nA = len(A)
    SS = env['SS']
    AA = env['AA']
    PP = env['PP']
    RR = env['RR']
    NNS = env['NNS']
    NNA = env['NNA']
    nSS = len(SS)
    nAA = len(AA)
    nPhi = len(Phi(S[0])) # num of features
    V = [0.0 for i in range(nS)]
    Vhat = [0.0 for i in range(nSS)]
    Theta = [0.0 for i in range(nPhi)]
    Theta = np.array(Theta)
    pi = [[0.0 for i in range(nA)] for j in range(nS)]
    Vhat_ = Vhat.copy()
    i = H
    D = 0.0

    MtPhi = [Phi(ss) for ss in SS]
    MtPhi = np.array(MtPhi)
    PIPhi = np.linalg.pinv(MtPhi)

    while True:
        # sampled state space = S0, sample = s1, s2, .., sN
        # N = # of sample
        # M = # of feature
        # dim(Theta) = M x 1
        # dim(PHI) = N x M
        # dim(V) = N x 1, V = PHI . theta
        # Once Theta is fitted to the system, a general formula V(s) = PHI(s) . Theta
        # where s is in original state space S can be computed.

        # V^ <- Theta . Phi
        # N.N.: V^(s) = Vbar(xi) where xi = arg min_xi' ||s-xi'||
        Vhat = np.matmul(MtPhi, Theta).tolist()

        # iterate through SS to compute DP of V^
        Vhat_ = Vhat.copy()
        for ss in range(nSS):
            Vhat_[ss] = 0.0
            maxV = FLOAT_MIN
            for aa in range(nAA):
                tempV = 0.0
                for ss_ in range(nSS):
                    tempV += PP(SS[ss], AA[aa])[ss_] * ( RR(SS[ss_], SS[ss], AA[aa]) + gamma * Vhat[ss_] )
                maxV = max(maxV, tempV)
            Vhat_[ss] = maxV
            print("ss =", ss)
        Vhat = Vhat_.copy()

        # compute new Theta = PseudoInv(PHI) . V^ and check if Theta converges
        # PHI = matrix of all features of all sampled states
        oldTheta = Theta.copy()
        Theta = np.matmul(PIPhi, np.array(Vhat))
        D = np.linalg.norm(oldTheta - Theta)

        print(D)

        if D < thr: break
        D = 0.0

        if H != -1:
            i -= 1
            if i <= 0:
                break


    # pi(s) = arg max_a Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')] for s and s' in sample state
    # or use eps-greedy
    eps = 0.1
    piHat = [[0.0 for i in range(nAA)] for j in range(nSS)]
    for ss in range(nSS):
        aa_max = 0
        maxV = FLOAT_MIN
        for aa in range(nAA):
            tempV = 0.0
            for ss_ in range(nSS):
                tempV += PP(SS[ss], AA[aa])[ss_] * ( R(SS[ss_], SS[ss], AA[aa]) + gamma * Vhat[ss_] )
            if maxV < tempV:
                maxV = tempV
                aa_max = aa

        piHat[ss] = [1-eps if NNA(a) == aa_max else eps/(nAA - 1) for a in range(nA)]
        sum_ = sum(piHat[ss])
        piHat[ss] = [i/sum_ for i in piHat[ss]]

    # pi(s) = pi^(nearest neighbor of s)
    pi = [piHat[NNS(s)] for s in range(nS)]

    # V = PHI . THeta
    V = [np.dot(Phi(S[s]), Theta) for s in range(nS)]

    print(pi)
    print()
    print(V)
    print()
    print(Theta)


    return (pi, V)

# http://people.csail.mit.edu/agf/Files/11ICRA-LinearCMDP.pdf
# https://people.eecs.berkeley.edu/~pabbeel/cs287-fa11/slides/discretization-of-continuous-state-space-MDPs.pdf
# http://cs229.stanford.edu/notes2020spring/cs229-notes12.pdf
