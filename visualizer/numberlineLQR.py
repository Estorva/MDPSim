import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def visualize(pi, V, env, H, gamma, thr, imageOutput="image"):
    ax = plt.gca()
    # Set bottom spine to zero position
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Show arror tip
    ax.plot(1, 0, '>k', transform=ax.get_yaxis_transform(), clip_on=False)

    # Add text to axis
    ax.text(1.02, -0.2, "time", ha='left', va='top', transform=ax.get_yaxis_transform())

    x = np.arange(0, pi.shape[0])
    plt.xticks(x)
    plt.xlim(x[0] - 0.5, x[-1] + 0.5)
    plt.plot(x, pi[:, 0], color='orange', label='position')
    plt.bar(x, pi[:, 1], label='velocity')

    extraString = '$\sigma_d$ = {}\nconvergence threshold = {}'.format(env['sigma_d'], thr)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=extraString))
    plt.legend(handles=handles)

    plt.savefig(imageOutput + '.jpg')
