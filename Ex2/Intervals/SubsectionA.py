from intervals import find_best_interval
from Utils import *
import matplotlib.pyplot as plt


def plot(xs, ys, intervals, title):
    '''
    Plots the X & Y samples on the canvas, using intervals to highlight the hypothesis decision boundaries
    :param xs: X values
    :param ys: Y labels
    :param intervals: Hypothesis decision boundaries
    :param title: Title of the graph (also the filename)
    '''
    plt.scatter(xs, ys)
    plt.xlim([-0.1, 1.1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.axvline(x=0.25, color='r', linestyle='-')
    plt.axvline(x=0.5, color='r', linestyle='-')
    plt.axvline(x=0.75, color='r', linestyle='-')
    for interval in intervals:
        plt.axvspan(interval[0], interval[1], facecolor = 'g', alpha = 0.3)
    plt.grid(True)
    plt.savefig(title + ".png")
    plt.show()


m = 100
xs, ys = sample(m)
title = 'section A - 100 samples'
intervals, _ = find_best_interval(xs, ys, 2)
plot(xs, ys, intervals, title)
