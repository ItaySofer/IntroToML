from intervals import find_best_interval
from Utils import *
import matplotlib.pyplot as plt


def plot(esK, esHoldoutK, K):

    plt.scatter(K, esHoldoutK, color='g', label='holdout error')
    plt.scatter(K, esK, color='r', label='empirical error')
    plt.xlim([-5, 25])
    plt.xlabel('number of intervals(k)')
    plt.ylabel('error')
    title = 'section E - empirical and holdout errors'
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(title + ".png")
    plt.show()

m = 50
K = range(1, 21, 1)

xs, ys = sample(m)
xsHoldout, ysHoldout = sample(m)

esK = []
esHoldoutK = []
for k in K:
    intervals, es = find_best_interval(xs, ys, k)
    esK.append(float(es)/m)
    esHoldoutK.append(calculateHoldoutError(intervals, xsHoldout, ysHoldout))

plot(esK, esHoldoutK, K)
