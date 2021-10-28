"""
File: simulator.py
Date: 10.26.2021 (created)
      10.26.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    Implementation of LQR.
"""

import numpy as np
import numpy.linalg as npla
import common.function as func

def MTNM(M: np.ndarray, N: np.ndarray):
    return M.T @ N @ M

def solve(env: dict, H: int, gamma: float, thr: float):
    # P_0 = Q
    # repeat
    #     compute K_H
    #     u_{H+1} = u*_H = -K_H . x_H
    #     P_{H+1} = Q + (K_H)T . R . K_H + (A-B.K_H)T . P_H . (A-B.K_H)
    #     H += 1
    # until H reaches horizon or P converges

    A = env['A']
    B = env['B']
    Bw = env['Bw']
    Q = env['Q']
    R = env['R']
    s = env['s0']
    sigma_d = env['sigma_d']
    P = Q.copy()
    K = Q.copy()
    i = H
    u = np.zeros((B.shape[1],))

    history = [s]

    while True:
        w = np.random.normal(0, sigma_d**2, size=(2,)) # sample from gaussian
        K = npla.inv(R/gamma + MTNM(B, P)) @ B.T @ P @ A
        u = -np.matmul(K, s)
        oldP = P.copy()
        P = Q + MTNM(K, R) + gamma * MTNM(A-np.matmul(B, K), P)
        s = np.matmul(A, s) + np.matmul(B, u) + np.matmul(Bw, w)
        history.append(s)

        if npla.norm(oldP - P) < thr:
            break

        if H != -1:
            i -= 1
            if i <= 0:
                break

    return (np.array(history), 0)



















#
