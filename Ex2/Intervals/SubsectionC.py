from intervals import find_best_interval
from Utils import *
import matplotlib.pyplot as plt


def plot(esM, epM, M):

    plt.plot(M, epM, 'g', label='true error')
    plt.plot(M, esM, 'r', label='empirical error')
    plt.xlim([-10, 110])
    plt.xlabel('number of samples(m)')
    plt.ylabel('error')
    title = 'section C - empirical and true errors'
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(title + ".png")
    plt.show()

M = range(10, 101, 5)
k = 2

esM = []
epM = []
for m in M:
    esTotal = 0
    epTotal = 0
    for T in range(0, 100):
        xs, ys = sample(m)
        intervals, es = find_best_interval(xs, ys, k)
        esTotal += float(es)/m
        epTotal += calculateTrueError(intervals)
    esM.append(esTotal/100)
    epM.append(epTotal/100)

plot(esM, epM, M)
