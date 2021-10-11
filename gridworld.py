"""
File: simulator.py
Date: 10.07.2021
Author: Shih-Che Sun
Synopsis:
    A MDP model of grid-world problem.
"""

import simulator
import matplotlib.pyplot as plt
import numpy as np

################################################################################
###################            Problem Definition            ###################
################################################################################

def coordPlus(p, q):
    assert isinstance(p, tuple)
    assert isinstance(q, tuple)
    assert len(p) == len(q)
    return tuple(p[i] + q[i] for i in range(len(p)))

class gwTP(simulator.Transition):
    def __init__(self, S, A, p_e):
        super().__init__(S, A)
        self.pe = p_e # probability of error

    def __call__(self, s, a):
        # translating text to coordinates
        def aToCoord(a):
            if a == 'up':
                return (0, 1)
            if a == 'down':
                return (0, -1)
            if a == 'left':
                return (-1, 0)
            if a == 'right':
                return (1, 0)
            if a == 'stay':
                return (0, 0)

        sn = coordPlus(s, aToCoord(a))
        nb = [ coordPlus(s, aToCoord(a_)) for a_ in self.A if a_ != a ] # neighbor states other than sn
        nb = [ i for i in nb if i in self.S ] # exclude forbidden states

        if sn in self.S:
            return [1.0 - len(nb) * self.pe if (i == sn) else
                   (self.pe if (i in nb) else 0.0)
                    for i in self.S ]
        else:
            return [ 1.0 if (i == s) else 0.0 for i in self.S ]

'''
class gwR(simulator.Transition):
    def __init__(self, S, A):
        super().__init__(S, A)
        self.rd = 11
        self.rs = 10
        self.rw = -5

    def __call__(self, sn, s, a):
        # reward R(s' | s, a)
        def sToR(s):
            if s == (2, 2):
                return self.rd
            if s == (2, 0):
                return self.rs
            if s[0] == 4:
                return self.rw

        return [ sToR(i) for i in self.S ]
'''

def gwR(sn, s, a):
    if (sn == (2, 2)):
        return 1
    if (sn == (2, 0)):
        return 10
    if (sn[0] == 4):
        return -5
    else:
        return 0

# system parameters
m = 5 # range of x
n = 6 # range of y
B = [(1, 1), (2, 1), (1, 3), (2, 3)]
S = [(i, j) for j in range(n) for i in range (m) if (i, j) not in B] # coordinates in a m-by-n rectangle
A = ['left', 'up', 'right', 'down', 'stay']
#A = ['↑', '↓', '←', '→', '．']
P = gwTP(S, A, p_e = 0.0)

# task parameters
H = -1
gamma = 0.8
R = gwR

################################################################################
###################                Simluation                ###################
################################################################################

sim = simulator.Environment(S=S, A=A, P=P, O=None, R=R)
sim.task(H=H, gamma=gamma)
#sim.policyIteration()
sim.valueIteration()

'''
a snippet for evaluating the system from a initial state
sim.initialize((2, 4))
sim.step('right')
sim.step('down')
sim.step('stay')
sim.step('left')
sim.step('stay')
sim.step('down')
sim.step('left')
print(sim.state)
'''

################################################################################
###################              Visualization               ###################
################################################################################

fig = plt.figure(figsize=plt.figaspect(.4))
n_subfig = 3
current_fig = 1
# reward
ax = fig.add_subplot(1, n_subfig, current_fig)
current_fig += 1
matR = [[gwR((i, j), S[0], A[0]) for j in range(n)] for i in range(m)]
for i in range(m):
    for j in range(n):
        if (i, j) in B: continue
        ax.text(i, j, matR[i][j], ha="center", va="center", color="black")

ax.set_xlim([-0.5, m-0.5])
ax.set_ylim([-0.5, n-0.5])
ax.set_xlabel("Reward")


# value(color mesh)
ax = fig.add_subplot(1, n_subfig, current_fig)
current_fig += 1
x = [float(i) - 0.5 for i in range(m+1)]
y = [float(i) - 0.5 for i in range(n+1)]

# sim.V is a 1d vector of length nS <= m*n
V = [[-10.0 for j in range(n)] for i in range(m)]
for i, s in enumerate(S):
    V[s[0]][s[1]] = sim.V[i]
for i in B:
    V[i[0]][i[1]] = np.NaN

pcm = ax.pcolormesh(x, y, np.transpose(V))
fig.colorbar(pcm)

for i in range(m):
    for j in range(n):
        if (i, j) in B: continue
        ax.text(i, j, f"{V[i][j]:.2f}", ha="center", va="center", color="black")

ax.set_xlabel("Value")

# policy
ax = fig.add_subplot(1, n_subfig, current_fig)
current_fig += 1
for i, s in enumerate(S):
    a = A[sim.pi[i].index(max(sim.pi[i]))]
    ax.text(s[0], s[1], str(a), ha="center", va="center", color="black")
ax.set_xlabel("Policy")
ax.set_xlim([-0.5, m-0.5])
ax.set_ylim([-0.5, n-0.5])

plt.savefig('images/gridworld.jpg')




























#
