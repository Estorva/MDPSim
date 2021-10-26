"""
File: function.py
Date: 10.22.2021 (created)
      10.25.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    Functions that are frequently called in MDP-solving methods. They are
    abstracted for better maintenance.
"""

import numpy as np

def bellmanBackup(env, gamma, V, s: int, pi=None) -> float:
    # Deterministic policy:
    #   V(s) = max_a sum_s' P(s'|s, a) (R(s'|s, a) + gamma * V(s'))
    # Stochastic policy:
    #   V(s) = sum_a pi(s, a) sum_s' P(s'|s, a) (R(s'|s, a) + gamma * V(s'))
    if isinstance(pi, np.ndarray):
        # Q(...) returns [Q(s, a1) Q(s, a2) ... Q(s, an)]
        return np.dot(pi[s], Q(env, gamma, V, s))
    else:
        return np.max(Q(env, gamma, V, s))

def Q(env, gamma, V, s: int) -> np.array:
    # V(s) = max_a sum_s' P(s'|s, a) (R(s'|s, a) + gamma * V(s'))
    #      = max_a Q(s, a)
    l = []
    A = env['A']
    S = env['S']
    P = env['P']
    R = env['R']
    s = S[s]
    for i, a in enumerate(A):
        summ = 0.0
        for i, s_ in enumerate(S):
            summ += P(s, a)[i] * ( R(s_, s, a) + gamma * V[i] )
        l.append(summ)
    return np.array(l)

def normalizedRandomArray(shape, axis=0):
    a = np.random.random(shape)
    a /= a.sum(axis=axis, keepdims=True)
    '''
      Suppose that M is an n-dimensional array of dim (d1, d2, ..., dn),
      sum(M, axis=i, keepdims=True) returns an array of dim (d1, ..., d{i-1}, 1, d{i+1})
      if keepdims is set to false, sum(.) returns an (n-1)-dimensional array.

      Suppose that op(.,.) is a binary operator, M is a n-by-n matrix, and v
      is a vector in R^n, op(M, v) broadcasts v onto row vectors of M by default.
      op(M, v[:, np.newaxis]) broadcasts v onto column vectors of M. In other words,
      M' = op(M, v) <=> M'[i] = op(M[i], v[i])
      M' = op(M, v[:, np.newaxis]) <=> M'[:, i] = op(M[:, i], v[i])
    '''
    return a
