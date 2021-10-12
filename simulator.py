"""
File: simulator.py
Date: 10.06.2021 (created)
      10.07.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    A fundamental framework for MDP's and reinforcement learning.
    A state-space system is a mathematical model for environments,
    defined as a 4-tuple: (S, A, P, O) where
        S: state space, the set of possible states of the system.
        A: action space, the set of possible actions the system takes.
        P: transition probability, the prob. of a next state s' given
           current state s and action a.
           (P: S x S x A -> R)
        O: observation probability, the prob. of an observation of system state
           based on the measured state and action.
           (O: S x A -> R)
           (this will be omitted as of now)
    An MDP involves finding the best policy for the agent to reach goals or
    get maximum rewards in an environment.
"""

import random
import sys
FLOAT_MIN = -1.0e100

class Environment:
    def __init__(self, S, A, P, O, R):
        self.S = S
        self.A = A
        self.P = P
        self.O = O
        self.R = R
        self.time = 0
        self.state = None
        self.H = None
        self.gamma = None
        self.pi = None
        self.V = None
        self.Q = None

        # computation on parameters
        self.nS = len(S)
        self.nA = len(A)

    def step(self, action):
        p = self.P(self.state, action)
        assert(len(p) == len(self.S))

        q = random.random()
        s = 0.0
        so = self.state # state_old

        for i in range(len(p)):
            s += p[i]
            if (q <= s):
                self.state = self.S[i]
                break

        self.time += 1
        return self.R(self.state, so, action)
        # return (self.O(sn), R(sn, s, a))

    def initialize(self, s):
        self.state = s
        self.time = 0

        if (s not in self.S):
            raise ValueError("Error: state {} is not in given state space!!".format(str(s)))

    def task(self, H, gamma):
        self.H = H # horizon
        self.gamma = gamma # discounting factor
        # stochastic policy: S x A -> R
        self.pi = [[random.random() for i in range(self.nA)] for j in range(self.nS)]
        for i in range(self.nS):
            s = sum(self.pi[i])
            self.pi[i] = [j/s for j in self.pi[i]]
        self.V = [0.0 for i in range(self.nS)]

    def valueIteration(self):
        # V*_0(s) = 0 for s in S
        # for i in 1..H
        #     D <- 0
        #     for each s in S
        #         v <- V(s)
        #         V(s) <- max_a Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')]
        #         D <- max(D, |v - V(s)|)
        #     if D < threshold: break

        D = 0.0
        thr = 0.3
        V_ = self.V.copy()
        i = self.H

        while True:
            for s in range(self.nS):
                V_[s] = 0.0
                maxV = FLOAT_MIN
                for a in range(self.nA):
                    tempV = 0.0
                    for s_ in range(self.nS):
                        tempV += self.P(self.S[s], self.A[a])[s_] * ( self.R(self.S[s_], self.S[s], self.A[a]) + self.gamma * self.V[s_] )
                        #if s == 1 and a == 2: print(s_, self.P(self.S[s], self.A[a])[s_], self.R(self.S[s_], self.S[s], self.A[a]))

                    maxV = max(maxV, tempV)
                V_[s] = maxV
                D = max(D, abs(V_[s] - self.V[s]))
            self.V = V_.copy()

            if D < thr: break
            D = 0.0

            if self.H != -1:
                if i <= 0:
                    break

        # pi(s) = arg max_a Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')]
        # or use eps-greedy
        eps = 0.1
        for s in range(self.nS):
            a_max = 0
            maxV = FLOAT_MIN
            for a in range(self.nA):
                tempV = 0.0
                for s_ in range(self.nS):
                    tempV += self.P(self.S[s], self.A[a])[s_] * ( self.R(self.S[s_], self.S[s], self.A[a]) + self.gamma * self.V[s_] )
                if maxV < tempV:
                    maxV = tempV
                    a_max = a

            self.pi[s] = [1-eps if a == a_max else eps/(self.nA - 1) for a in range(self.nA)]

    def policyEvaluation(self):
        # repeat
        #     D <- 0
        #     for each s in S
        #         v <- V(s)
        #         V(s) <- Sum_s' P(s'|s, pi(s)) [r(s'|s, pi(s)) + gamma * V(s')]
        #         D <- max(D, |v - V(s)|)
        #     if D < threshold: break

        D = 0.0
        thr = 0.3
        V_ = self.V.copy()

        while True:
            for s in range(self.nS):
                V_[s] = 0.0
                for s_ in range(self.nS):
                    for a in range(self.nA):
                        V_[s] += self.pi[s][a] * self.P(self.S[s], self.A[a])[s_] * ( self.R(self.S[s_], self.S[s], self.A[a]) + self.gamma * self.V[s_] ) # Bellman backup
                D = max(D, abs(V_[s] - self.V[s]))
            self.V = V_.copy()

            if D < thr: break
            D = 0.0

    def policyImprovement(self) -> bool:
        # policyIsStable <- True
        # for each s in S:
        #     b <- pi(s)
        #     pi(s) <- arg max_a Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')]
        #     if b != pi(s), then policyIsStable <- False
        # if !policyIsStable:
        #     policyEvaluation()

        stable = True
        for s in range(self.nS):
            max1 = 0.0
            max2 = 0.0
            a_max1 = self.A[0]
            a_max2 = self.A[0]
            for a in range(self.nA):
                if self.pi[s][a] > max1:
                    a_max1 = a
                    max1 = self.pi[s][a]

                # summ = Sum_s' P(s'|s, a) [r(s'|s, a) + gamma * V(s')]
                summ = 0.0
                for s_ in range(self.nS):
                    summ += self.P(self.S[s], self.A[a])[s_] * ( self.R(self.S[s_], self.S[s], self.A[a]) + self.gamma * self.V[s_] )
                if summ > max2:
                    a_max2 = a
                    max2 = summ

            # epsilon-greedy
            eps = 0.1
            self.pi[s] = [1-eps if a == a_max2 else eps/(self.nA - 1) for a in range(self.nA)]
            stable = stable and (a_max1 == a_max2)

        return stable

    def policyIteration(self):
        while True:
            self.policyEvaluation()
            if self.policyImprovement(): break


class Transition:
    def __init__(self, S, A):
        self.S = S
        self.A = A

    def __call__(self, s, a):
        pass











#
