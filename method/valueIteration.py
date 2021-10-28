"""
File: simulator.py
Date: 10.13.2021 (created)
      10.25.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    Implementation of value iteration.
"""

import random
import numpy as np
import numpy.linalg as npla
import common.function as func


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
    nS = len(S)
    nA = len(A)
    V = np.zeros((nS,))
    V_ = V.copy()
    i = H
    D = 0.0

    while True:
        for s in range(nS):
            V_[s] = func.bellmanBackup(env, gamma, V, s)
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
    pi = []
    for s in range(nS):
        Q = func.Q(env, gamma, V, s)
        a_max = Q.argmax()
        pi.append([1-eps if a == a_max else eps/(nA - 1) for a in range(nA)])

    return (np.array(pi), V)
