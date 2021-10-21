"""
File: simulator.py
Date: 10.06.2021 (created)
      10.13.2021 (updated)
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


class Environment:
    def __init__(self, S, A, P, O, R):
        self.S = S
        self.A = A
        self.P = P
        self.O = O
        self.R = R
        self.time = 0
        self.state = None

        # computation on parameters
        self.nS = len(S)
        self.nA = len(A)


    @classmethod
    def fromDict(cls, d: dict):
        return cls(d['S'], d['A'], d['P'], d['O'], d['R'])

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



class Transition:
    def __init__(self, S, A):
        self.S = S
        self.A = A

    def __call__(self, s, a):
        pass











#
