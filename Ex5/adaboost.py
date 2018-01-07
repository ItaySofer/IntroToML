import numpy as np
import hw5
import argparse
from math import log, exp
import matplotlib.pyplot as plt


class WeakLearner:
    def __init__(self):
        self.F_star = float("inf")
        self.j = None
        self.theta = None
        self.b = None

    def train(self, X, Y, D):
        '''
        @:param X - The training set data; X ~ R[m, d]
        @:param Y - The training set labels; Y ~ R[m]
        @:param D - The distribution vector; D ~ R[m]
        @:return The weak learner sets a decision stump threshold, coordinate and bias for the given data and distribution.
        '''

        assert X.shape[0] == Y.shape[0] == D.shape[0]
        m, d = X.shape

        for j in range(d):
            sorted_X = X[X[:, j].argsort()]
            sorted_X = np.concatenate((sorted_X, (sorted_X[m - 1] + 1)[None]), axis=0)  # Add X[m+1]
            Fpos = np.sum(D[np.where(Y > 0)])  # Sum where yi=1.   (yi is -1 or 1)
            Fneg = 1 - Fpos  # Sum where yi=-1.  (yi is -1 or 1)

            if (Fpos < self.F_star) or (Fneg < self.F_star):
                self.F_star = min(Fpos, Fneg)  # Save best objective found for pair of hypotheses
                theta_star = sorted_X[0, j] - 1
                j_star = j
                b_star = 1 if Fpos <= Fneg else -1  # Bias chooses 1 of the current two hypotheses in the decision stump

            for i in range(m):
                Fpos -= Y[i] * D[i]  # H uses threshold of (1,1,1,1, theta, -1,  ... -1, -1, -1)
                Fneg += Y[i] * D[i]  # H uses threshold of (-1,-1,-1,-1 theta, 1, ... 1, 1, 1)

                if (min(Fpos, Fneg) < self.F_star) and (sorted_X[i, j] != sorted_X[i + 1, j]):
                    self.F_star = min(Fpos, Fneg)
                    theta_star = 0.5 * (sorted_X[i, j] + sorted_X[i + 1, j])
                    j_star = j
                    b_star = 1 if Fpos <= Fneg else -1

        self.j = j_star
        self.theta = theta_star
        self.b = b_star

    def predict(self, x):
        '''
        :param x: A batch of samples with dimensions (m, d)
        :return: Predicted label
        '''
        assert all([self.j, self.theta, self.b])  # Verify the weak learner's params aren't None
        return np.sign(self.theta - x[:, self.j]) * self.b

    def __call__(self, x):
        return self.predict(x)


class AdaBoost:
    def __init__(self):
        self.h = None
        self.w = None

    @staticmethod
    def dist_update(wt, Hx, Y, Dt):
        nominators = np.multiply(Dt, np.exp(np.multiply(-wt, np.multiply(Y, Hx))))
        return np.divide(nominators, np.sum(nominators))

    def train(self, X, Y, T):
        '''
        @:param X - The training set data; X ~ R[m, d]
        @:param Y - The training set labels; Y ~ R[m]
        @:param T - Number of iterations to train AdaBoost
        @:return A strong classifier composed of multiple weighted Weak Learners
        '''
        assert X.shape[0] == Y.shape[0] and T > 0
        hypotheses = []
        weights = []
        m = X.shape[0]
        D = np.full(shape=(m), fill_value=(float(1) / m))  # Initialize with even distributed probabilities

        for i in range(T):
            print('AdaBoost iteration #{0}'.format(i+1))
            wl = WeakLearner()
            wl.train(X, Y, D)

            predictions = wl(X)
            err = np.sum(D[np.where(Y != predictions)])  # Sum where yi=1.   (yi is -1 or 1)
            wt = 0.5 * log((float(1) / err) - 1)
            D = self.dist_update(wt=wt, Hx=predictions, Y=Y, Dt=D)

            hypotheses.append(wl)
            weights.append(wt)

        self.h = np.array(hypotheses)
        self.w = np.array(weights)

    def predict(self, x):
        assert (self.h is not None) and (self.w is not None)  # Verify the AdaBoost parameters aren't None
        # wl_predictions = np.transpose(np.array([ht(x) for ht in self.h]))  # Query all weak learners in the array
        # return np.sign(np.sum(np.multiply(self.w, wl_predictions), axis=1))  # Return weighted result

        hx = []
        for xt in x:
            wl_predictions = np.array([ht(xt[None]) for ht in self.h])
            result = np.sign(np.sum(np.multiply(self.w, wl_predictions)))
            hx.append(result)

        return np.array(hx)

    def __call__(self, x):
        return self.predict(x)

    def limit(self, limit):
        '''
        :return: A clone of the classifier limited to "limit" rounds (neglects all hypotheses whose index > limit)
        '''
        limited_adaboost = AdaBoost()
        limited_adaboost.h = self.h[:limit]
        limited_adaboost.w = self.w[:limit]
        return limited_adaboost


def ex_5_a():
    wl = WeakLearner()
    m = hw5.train_data.shape[0]  # Num of samples
    wl.train(X=hw5.train_data, Y=hw5.train_labels, D=np.full(shape=(m), fill_value=(float(1) / m)))
    print('The Weak Learner have learned after one iteration: j={0}, theta={1}, bias={2}'.format(wl.j, wl.theta, wl.b))


def calculate_error(predictions, labels):
    ''' Calculate the zero-one error for the Adaboost algorithm, normalized by # of total predictions '''
    processed_results = np.subtract(predictions, labels)
    return float(np.count_nonzero(processed_results)) / len(predictions)


def plot(x, y1, y2, y1legend, y2legend, x_min, x_max, y_min, y_max, xlabel, ylabel, title):
    '''
    Plots the X & Y samples on the canvas, using min - max range for the axes
    '''
    plt.plot(x, y1, color='red', label=y1legend)
    plt.plot(x, y2, color='blue', label=y2legend)
    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.savefig(title + ".png")
    plt.show()


def ex_5_b():
    adaboost = AdaBoost()
    T = 100
    adaboost.train(X=hw5.train_data, Y=hw5.train_labels, T=T)

    training_errors = []
    test_errors = []

    for i in range(1, T + 1):
        adaboost_iteration = adaboost.limit(i)
        training_perdictions = adaboost_iteration.predict(hw5.train_data)
        training_errors.append(calculate_error(predictions=training_perdictions, labels=hw5.train_labels))
        test_perdictions = adaboost_iteration.predict(hw5.test_data)
        test_errors.append(calculate_error(predictions=test_perdictions, labels=hw5.test_labels))
        print('Iteration #{0} ; Training Error: {1:.2f} ; Test Error: {2:.2f}'.format(i, training_errors[i-1], test_errors[i-1]))

    plot(x=range(1, T + 1), y1=training_errors, y1legend='Training Error', y2=test_errors,
         y2legend='Test Error', x_min=1, x_max=T+1,
         y_min=min(min(test_errors), min(training_errors)) - 0.1,
         y_max=max(max(test_errors), max(training_errors)) + 0.1,
         xlabel='t [AdaBoost Iterations]', ylabel='Error', title='Section 1.B - AdaBoost Error per Iteration')


parser = argparse.ArgumentParser(description='Choose a subsection')
parser.add_argument('--section', help='A section of the ex')
args = parser.parse_args()
if args.section == '1':
    ex_5_a()
elif args.section == '2':
    ex_5_b()
