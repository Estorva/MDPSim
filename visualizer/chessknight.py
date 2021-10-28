import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

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

def visualize(pi, V, env, H, gamma, thr, imageOutput="image"):
    A = env['A']
    B = env['B']
    S = env['S']
    dim = env['dim']
    si = env['sisf'][0]

    board = [[(i+j)%2 for i in range(dim[0])] for j in range(dim[1])]
    for b in B:
        if b[0] >= dim[0] or b[1] >= dim[1]:
            continue
        board[b[0]][b[1]] = 2
    cmap = ListedColormap([(0.91, 0.84, 0.77), (0.62, 0.5, 0.47), (0,0,0)])
    board = np.array(board).T # matshow() is i,j-indexed but text() and arror() are x,y-indexed
    plt.matshow(board, cmap=cmap)

    for i, parent in enumerate(pi):
        if parent == -1: continue
        p, a = parent
        x = S[p][0]
        y = S[p][1]

        if p == si:
            plt.text(x, y, "\u265E", fontsize=30, horizontalalignment='center', verticalalignment='center')

        a = aToCoord(A[a])
        dx = a[0]
        dy = a[1]
        plt.arrow(x, y, dx, dy, length_includes_head=True, width=0.1, head_length=0.2)

    plt.xlim([-0.5, dim[0]-0.5])
    plt.ylim([-0.5, dim[1]-0.5])

    plt.savefig(imageOutput + '.jpg')
