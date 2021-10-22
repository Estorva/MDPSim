import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import problem.chessknight as ck
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

def visualize(simPi, simV, imageOutput = "image"):
    board = [[(i+j)%2 for i in range(ck.dim[0])] for j in range(ck.dim[1])]
    for b in ck.B:
        if b[0] >= ck.dim[0] or b[1] >= ck.dim[1]:
            continue
        board[b[0]][b[1]] = 2
    cmap = ListedColormap([(0.91, 0.84, 0.77), (0.62, 0.5, 0.47), (0,0,0)])
    board = np.array(board).T # matshow() is i,j-indexed but text() and arror() are x,y-indexed
    plt.matshow(board, cmap=cmap)

    for i, parent in enumerate(simPi):
        if parent == -1: continue
        p, a = parent
        x = ck.S[p][0]
        y = ck.S[p][1]

        if p == ck.env['sisf'][0]:
            plt.text(x, y, "\u265E", fontsize=30, horizontalalignment='center', verticalalignment='center')

        a = aToCoord(ck.A[a])
        dx = a[0]
        dy = a[1]
        plt.arrow(x, y, dx, dy, length_includes_head=True, width=0.1, head_length=0.2)

    plt.xlim([-0.5, ck.dim[0]-0.5])
    plt.ylim([-0.5, ck.dim[1]-0.5])

    plt.savefig('images/' + imageOutput + '.jpg')

    # print how many steps the knight makes
