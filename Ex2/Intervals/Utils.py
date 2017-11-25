import random
from operator import itemgetter
import numpy as np


def sample(m):
    '''
    Generates m samples according to the pre-defined distributions
    :param m: Number of samples to return
    :return: Samples of X, Y
    '''

    samples = []
    for i in range(0, m):
        x = random.uniform(0, 1)
        py = 0.8 if (x >= 0 and x <= 0.25) or (x >= 0.5 and x <= 0.75) else 0.1
        y = np.random.binomial(1, py)
        samples.append((x, y))
    samples = sorted(samples, key=itemgetter(0))
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]

    return xs, ys


def calculateTrueError(intervals):
    '''
    Calculates the true error for the hypothesis
    :param intervals: The interval boundaries determined by the hypothesis
    :return: The true error value of the hypothesis against the known probabilities
    '''
    anchors = set(sum(intervals, ()))  # Flatten points from all intervals

    # We create a sorted set of all intervals from the hypothesis and the ground truth,
    # as each intersection has different probability values.
    # Each anchor represents the range of values that span from the current anchor to previous one
    if 0 in anchors:
        anchors.remove(0)
    anchors.add(0.25)
    anchors.add(0.5)
    anchors.add(0.75)
    anchors.add(1)
    anchors = sorted(anchors)

    # We calculate the true error components for each intersection range, and then multiply and add them
    # to return the true error.
    # Following is the chance that Y=0 or Y=1 given x being in a known boundary
    pY0GivenX = [0.2 if (0 < x <= 0.25) or (0.5 < x <= 0.75) else 0.9 for x in anchors]
    pY1GivenX = [0.8 if (0 < x <= 0.25) or (0.5 < x <= 0.75) else 0.1 for x in anchors]

    # Calculate Zero-One Loss when the hypothesis intersection thinks wrong
    delta0 = [1 if np.any(filter(lambda interval:interval[0] < x <= interval[1], intervals)) else 0 for x in anchors]
    delta1 = [0 if np.any(filter(lambda interval:interval[0] < x <= interval[1], intervals)) else 1 for x in anchors]
    px = np.diff([0] + anchors)

    return np.dot(np.multiply(pY0GivenX, px), delta0) + np.dot(np.multiply(pY1GivenX, px), delta1)


def calculateHoldoutError(intervals, xsHoldout, ysHoldout):
    delta = 0
    for i in range(0, len(xsHoldout)):
        x = xsHoldout[i]
        y = ysHoldout[i]
        hOfX = np.any(filter(lambda interval: interval[0] <= x <= interval[1], intervals))
        if y != hOfX:
            delta += 1
    return float(delta)/len(xsHoldout)
