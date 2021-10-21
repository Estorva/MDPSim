"""
File: simulator.py
Date: 10.13.2021
Author: Shih-Che Sun
Synopsis:
    Implementation of policy iteration.
"""

import random

def solve(env: dict, H: int, gamma: float, thr: float):
    nS = len(env['S'])
    nA = len(env['A'])

    V = [0.0 for i in range(nS)]
    pi = [[random.random() for i in range(nA)] for j in range(nS)]
    for i in range(nS):
        s = sum(pi[i])
        pi[i] = [j/s for j in pi[i]]

    while True:
        V = policyEvaluation(env, pi, V, H, gamma, thr)
        pi, stable = policyImprovement(env, pi, V, H, gamma, thr)
        if stable: break

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
    A = env['A']
    P = env['P']
    O = env['O']
    R = env['R']
    nS = len(S)
    nA = len(A)

    V_ = V.copy()
    D = 0.0

    while True:
        for s in range(nS):
            V_[s] = 0.0
            for s_ in range(nS):
                for a in range(nA):
                    V_[s] += pi[s][a] * P(S[s], A[a])[s_] * ( R(S[s_], S[s], A[a]) + gamma * V[s_] ) # Bellman backup
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
    P = env['P']
    O = env['O']
    R = env['R']
    nS = len(S)
    nA = len(A)

    stable = True
    for s in range(nS):
        max1 = 0.0
        max2 = 0.0
        a_max1 = 0
        a_max2 = 0
        for a in range(nA):
            if pi[s][a] > max1:
                a_max1 = a
                max1 = pi[s][a]

            # summ = Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')]
            summ = 0.0
            for s_ in range(nS):
                summ += P(S[s], A[a])[s_] * ( R(S[s_], S[s], A[a]) + gamma * V[s_] )
            if summ > max2:
                a_max2 = a
                max2 = summ

        # epsilon-greedy
        eps = 0.1
        pi[s] = [1-eps if a == a_max2 else eps/(nA - 1) for a in range(nA)]
        stable = stable and (a_max1 == a_max2)

    return pi, stable
