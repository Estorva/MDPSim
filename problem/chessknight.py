import common.simulator as simulator
import numpy as np

def coordPlus(p, q):
    assert isinstance(p, tuple)
    assert isinstance(q, tuple)
    assert len(p) == len(q)
    return tuple(p[i] + q[i] for i in range(len(p)))

class gwTP(simulator.Transition):
    # the system is deterministic for knights on chess
    def __init__(self, S, A):
        super().__init__(S, A)

    def __call__(self, s, a):
        def aToCoord(a):
            if a == 'nne':
                return (1, 2)
            if a == 'ene':
                return (2, 1)
            if a == 'ese':
                return (2, -1)
            if a == 'sse':
                return (1, -2)
            if a == 'ssw':
                return (-1, -2)
            if a == 'wsw':
                return (-2, -1)
            if a == 'wnw':
                return (-2, 1)
            if a == 'nnw':
                return (-1, 2)

        sn = coordPlus(s, aToCoord(a))

        if sn in self.S:
            return [ 1.0 if (i == sn) else 0.0 for i in self.S ]
        else:
            return [ 1.0 if (i == s) else 0.0 for i in self.S ]

dim = (8,8) # (x,y) = (8,8)

B = [(13,12),
(1,9),
(19,6),
(7,14),
(13,11),
(16,17),
(4,6),
(9,16),
(0,7),
(13,8),
(14,4),
(7,2),
(14,10),
(12,9),
(6,13),
(1,11),
(8,0),
(5,15),
(18,0),
(18,13),
(4,1),
(16,18),
(6,6),
(18,12),
(12,3),
(15,2),
(7,0),
(1,9),
(14,3),
(5,6),
(11,17),
(1,18),
(0,7),
(13,4),
(17,14),
(17,5),
(7,8),
(14,1),
(4,5),
(18,1),
(0,5),
(19,3),
(5,13),
(7,7),
(4,13),
(1,2),
(7,15),
(3,12),
(12,15),
(10,1),
(3,18),
(3,16),
(11,6),
(4,15),
(16,17),
(19,5),
(8,9),
(3,11),
(0,11),
(18,11),
(7,1),
(18,13),
(12,2),
(13,10),
(2,7),
(15,15),
(6,5),
(11,7),
(17,1),
(1,0),
(15,13),
(18,0),
(0,14),
(16,1),
(8,18),
(0,10),
(4,2),
(8,10),
(0,1),
(16,2),
(1,10),
(18,6),
(15,7),
(13,2),
(5,4),
(8,10),
(9,4),
(14,11),
(1,18),
(15,17),
(5,9),
(3,3),(8,8),
(8,9),
(8,10),
(8,11),
(8,12),
(8,13),
(8,14),
(8,15),
(9,8),
(9,9),
(9,10),
(9,11),
(9,12),
(9,13),
(9,14),
(9,15),
(10,8),
(10,9),
(10,10),
(10,11),
(10,12),
(10,13),
(10,14),
(10,15),
(11,8),
(11,9),
(11,10),
(11,11),
(11,12),
(11,13),
(11,14),
(11,15),
(12,8),
(12,9),
(12,10),
(12,11),
(12,12),
(12,13),
(12,14),
(12,15),
(13,8),
(13,9),
(13,10),
(13,11),
(13,12),
(13,13),
(13,14),
(13,15),
(14,8),
(14,9),
(14,10),
(14,11),
(14,12),
(14,13),
(14,14),
(14,15),
(15,8),
(15,9),
(15,10),
(15,11),
(15,12),
(15,13),
(15,14),
(15,15),
]

# state space is 8x8 chess grid
S = [(i,j) for i in range(dim[0]) for j in range(dim[1]) if (i,j) not in B]
# action space is subset of 16-wind compass points
A = ['nne', 'ene', 'ese', 'sse', 'ssw', 'wsw', 'wnw', 'nnw']

# estimate of distance between two points
def h(s1, s2):
    # heuristic that estimates the distance btw s and sf
    # !!! this is problem-specific !!!
    # L1-norm
    #return abs(S[s1][0] - S[s2][0]) + abs(S[s1][1] - S[s2][1])
    # L2-norm
    return (S[s1][0] - S[s2][0])**2 + (S[s1][1] - S[s2][1])**2

env = {
    'S': S,
    'A': A,
    'P': gwTP(S, A),
    'sisf': (0, len(S)-1),
    'dim': dim,
    'B': B,
    'h': h,
}
