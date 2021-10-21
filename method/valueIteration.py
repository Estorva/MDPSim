"""
File: simulator.py
Date: 10.13.2021
Author: Shih-Che Sun
Synopsis:
    Implementation of value iteration.
"""

import random

FLOAT_MIN = -1.0e100

def solve(env: dict, H: int, gamma: float, thr: float = 0.2):
    # V*_0(s) = 0 for s in S
    # for i in 1..H
    #     D <- 0
    #     for each s in S
    #         v <- V(s)
    #         V(s) <- max_a Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')]
    #         D <- max(D, |v - V(s)|)
    #     if D < threshold: break

    S = env['S']
    A = env['A']
    P = env['P']
    O = env['O']
    R = env['R']
    nS = len(S)
    nA = len(A)
    V = [0.0 for i in range(nS)]
    pi = [[random.random() for i in range(nA)] for j in range(nS)]
    for i in range(nS):
        s = sum(pi[i])
        pi[i] = [j/s for j in pi[i]]
    V_ = V.copy()
    i = H
    D = 0.0

    while True:
        for s in range(nS):
            V_[s] = 0.0
            maxV = FLOAT_MIN
            for a in range(nA):
                tempV = 0.0
                for s_ in range(nS):
                    tempV += P(S[s], A[a])[s_] * ( R(S[s_], S[s], A[a]) + gamma * V[s_] )
                maxV = max(maxV, tempV)
            V_[s] = maxV
            D = max(D, abs(V_[s] - V[s]))
        V = V_.copy()

        if D < thr: break
        D = 0.0

        if H != -1:
            i -= 1
            if i <= 0:
                break

    # pi(s) = arg max_a Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')]
    # or use eps-greedy
    eps = 0.1
    for s in range(nS):
        a_max = 0
        maxV = FLOAT_MIN
        for a in range(nA):
            tempV = 0.0
            for s_ in range(nS):
                tempV += P(S[s], A[a])[s_] * ( R(S[s_], S[s], A[a]) + gamma * V[s_] )
            if maxV < tempV:
                maxV = tempV
                a_max = a

        pi[s] = [1-eps if a == a_max else eps/(nA - 1) for a in range(nA)]

    return (pi, V)
