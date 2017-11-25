from intervals import find_best_interval
from Utils import *
import matplotlib.pyplot as plt


def plot(esM, epM, K):

    plt.plot(K, epM, 'g', label='true error')
    plt.plot(K, esM, 'r', label='empirical error')
    plt.xlim([-5, 25])
    plt.xlabel('number of intervals(k)')
    plt.ylabel('error')
    title = 'section D - empirical and true errors'
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(title + ".png")
    plt.show()

m = 50
K = range(1, 21, 1)

xs, ys = sample(m)

esK = []
epK = []
for k in K:
    intervals, es = find_best_interval(xs, ys, k)
    esK.append(float(es)/m)
    epK.append(calculateTrueError(intervals))

plot(esK, epK, K)
