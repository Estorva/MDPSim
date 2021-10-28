import numpy as np
import matplotlib.pyplot as plt

def visualize(pi, V, env, H, gamma, thr, imageOutput="image"):
    R = env['R']
    m, n = env['dim']
    S = env['S']
    A = env['A']
    B = env['B']

    fig = plt.figure(figsize=plt.figaspect(.4))
    n_subfig = 3
    current_fig = 1

    # reward
    ax = fig.add_subplot(1, n_subfig, current_fig)
    current_fig += 1
    matR = [[R((i, j), S[0], A[0]) for j in range(n)] for i in range(m)]
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

    # V is a 1d vector of length nS <= m*n
    V_ = [[-10.0 for j in range(n)] for i in range(m)]
    for i, s in enumerate(S):
        V_[s[0]][s[1]] = V[i]
    for i in B:
        V_[i[0]][i[1]] = np.NaN

    pcm = ax.pcolormesh(x, y, np.transpose(V_))
    fig.colorbar(pcm)

    for i in range(m):
        for j in range(n):
            if (i, j) in B: continue
            ax.text(i, j, f"{V_[i][j]:.2f}", ha="center", va="center", color="black")

    ax.set_xlabel("Value")

    # policy
    ax = fig.add_subplot(1, n_subfig, current_fig)
    current_fig += 1
    for i, s in enumerate(S):
        a = A[pi.argmax(axis=1)[i]]
        ax.text(s[0], s[1], str(a), ha="center", va="center", color="black")
    ax.set_xlabel("Policy")
    ax.set_xlim([-0.5, m-0.5])
    ax.set_ylim([-0.5, n-0.5])

    plt.savefig(imageOutput + '.jpg')
