import common.simulator as simulator

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

B = [(3,5), (1,6), (4,3), (3,2), (3,6), (4,5), (6,5), (7,5)]

# state space is 8x8 chess grid
S = [(i,j) for i in range(dim[0]) for j in range(dim[1]) if (i,j) not in B]
# action space is subset of 16-wind compass points
A = ['nne', 'ene', 'ese', 'sse', 'ssw', 'wsw', 'wnw', 'nnw']

env = {
    'S': S,
    'A': A,
    'P': gwTP(S, A),
    'sisf': (0, len(S)-1),
    'dim': dim,
    'B': B,
}
