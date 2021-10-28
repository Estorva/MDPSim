"""
File: problem/numberlineLQR.py
Date: 10.26.2021 (created)
      10.26.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    A LQR model of number-line problem.
"""

# S and A are continuous

import numpy as np
import numpy.linalg as npla
import common.function as func

# s_{t+1} = [1 1] . [y_t] + [0 0] . [f_t] + [0 0] . [d_t]
#           [0 1]   [v_t]   [0 1]   [f_t]   [0 1]   [d_t]

# x' = Ax + Bu + Bw . w
A = np.array([[1., 1.], [0., 1.]])
B = np.array([[0., 0.], [0., 1.]])
# g = xT . Q . x + xT . N . u + uT . R . u
Q = np.array([[0.5, 0.1], [.3, .4]])
N = np.array([[0., 0.], [0., 0.]])
R = np.array([[.1, 0.], [0., .1]])

s0 = [20., 2.]

sigma_d = 0.2

env = {
    'A': A,
    'B': B,
    'Bw': B,
    'Q': Q,
    'N': N,
    'R': R,
    's0': s0,
    'sigma_d': sigma_d,
}
