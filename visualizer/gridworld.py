import problem.gridworld as gw
import numpy as np
import matplotlib.pyplot as plt

def visualize(simPi, simV, imageOutput = None):
    fig = plt.figure(figsize=plt.figaspect(.4))
    n_subfig = 3
    current_fig = 1

    # reward
    ax = fig.add_subplot(1, n_subfig, current_fig)
    current_fig += 1
    matR = [[gw.gwR((i, j), gw.S[0], gw.A[0]) for j in range(gw.n)] for i in range(gw.m)]
    for i in range(gw.m):
        for j in range(gw.n):
            if (i, j) in gw.B: continue
            ax.text(i, j, matR[i][j], ha="center", va="center", color="black")

    ax.set_xlim([-0.5, gw.m-0.5])
    ax.set_ylim([-0.5, gw.n-0.5])
    ax.set_xlabel("Reward")

    # value(color mesh)
    ax = fig.add_subplot(1, n_subfig, current_fig)
    current_fig += 1
    x = [float(i) - 0.5 for i in range(gw.m+1)]
    y = [float(i) - 0.5 for i in range(gw.n+1)]

    # sim.V is a 1d vector of length nS <= m*n
    V = [[-10.0 for j in range(gw.n)] for i in range(gw.m)]
    for i, s in enumerate(gw.S):
        V[s[0]][s[1]] = simV[i]
    for i in gw.B:
        V[i[0]][i[1]] = np.NaN

    pcm = ax.pcolormesh(x, y, np.transpose(V))
    fig.colorbar(pcm)

    for i in range(gw.m):
        for j in range(gw.n):
            if (i, j) in gw.B: continue
            ax.text(i, j, f"{V[i][j]:.2f}", ha="center", va="center", color="black")

    ax.set_xlabel("Value")

    # policy
    ax = fig.add_subplot(1, n_subfig, current_fig)
    current_fig += 1
    for i, s in enumerate(gw.S):
        a = gw.A[simPi[i].index(max(simPi[i]))]
        ax.text(s[0], s[1], str(a), ha="center", va="center", color="black")
    ax.set_xlabel("Policy")
    ax.set_xlim([-0.5, gw.m-0.5])
    ax.set_ylim([-0.5, gw.n-0.5])

    plt.savefig(imageOutput + '.jpg')
