"""
File: simulator.py
Date: 10.13.2021 (created)
      10.25.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    Implementation of policy iteration.
"""

import random
import numpy as np
import numpy.linalg as npla
import common.function as func

def solve(env: dict, H: int, gamma: float, thr: float):
    nS = len(env['S'])
    nA = len(env['A'])

    V = np.zeros((nS,))
    pi = func.normalizedRandomArray((nS, nA), axis=1)

    while True:
        V = policyEvaluation(env, pi, V, H, gamma, thr)
        pi, stable = policyImprovement(env, pi, V, H, gamma, thr)
        if stable: break

    pi = pi.tolist()
    return (pi, V)

def policyEvaluation(env, pi, V, H, gamma, thr):
    # repeat
    #     D <- 0
    #     for each s in S
    #         v <- V(s)
    #         V(s) <- Sum_s' P(s'|s, pi(s)) [r(s'|s, pi(s)) + gamma * V(s')]
    #         D <- max(D, |v - V(s)|)
    #     if D < threshold: break

    S = env['S']
    nS = len(S)

    V_ = V.copy()
    D = 0.0

    while True:
        for s in range(nS):
            V_[s] = func.bellmanBackup(env, gamma, V, s, pi=pi)
            D = max(D, abs(V_[s] - V[s]))
        V = V_.copy()

        if D < thr: break
        D = 0.0

    return V

def policyImprovement(env, pi, V, H, gamma, thr):
    # policyIsStable <- True
    # for each s in S:
    #     b <- pi(s)
    #     pi(s) <- arg max_a Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')]
    #     if b != pi(s), then policyIsStable <- False
    # if !policyIsStable:
    #     policyEvaluation()

    S = env['S']
    A = env['A']
    nS = len(S)
    nA = len(A)

    a1 = pi.argmax(axis=1)
    a2 = np.fromiter((func.Q(env, gamma, V, s).argmax() for s in range(nS)), dtype=int)
    stable = np.array_equal(a1, a2)

    eps = 0.1
    r = np.arange(0, nS) # r = [0 1 ... nS-1]
    pi = np.full(pi.shape, eps/(nA-1)) # initialize with all values=eps
    pi[r, a2] = 1-eps
    # M[[x1 x2 ... xn], [y1 y2 ... yn]] = [k1 k2 ... kn]
    # is equivalent to
    # M[x1, y1] = k1, M[x2, y2] = k2, ..., M[xn, yn] = kn
    # if only one constant is present at the right hand side of assignment sign,
    # all points specified by (x, y) is set to that constant

    return pi, stable
