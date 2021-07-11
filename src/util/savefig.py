import os

from matplotlib import pyplot

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fig_count = 0


def build_path():
    return os.path.join(root_path, 'build')


def savefig(plt: pyplot):
    global fig_count
    fig_count += 1
    fig_path = os.path.join(build_path(), f'weather_fig{fig_count}.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'saved: {fig_path}')
