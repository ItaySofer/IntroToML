import numpy as np
import hw4
import argparse
import matplotlib.pyplot as plt


class SGD_SVM:
    '''
    A SVM classifier, trained with SGD of batch size=1.
    The SVM uses hinge loss.
    '''

    def __init__(self, dim):
        # Initialize weight by heuristic
        self.w = np.zeros(dim)

    def train(self, x, y, C, T, basic_step_size):
        '''
        Performs T steps of SGD for SVM with hinge-loss
        :param x: Data samples
        :param y: Data labels
        :param C: C parameter of SVM loss
        :param T: Number of SGD timesteps to perform
        :param basic_step_size: The basic learning rate, before decay is applied
        :return: SVM's weight matrix at time T+1
        '''
        assert (x.shape[1] == len(self.w))

        # Sample T samples of (x,y) at once
        sampled_indices = np.random.random_integers(low=0, high=len(x)-1, size=T)
        sampled_x = np.take(a=x, indices=sampled_indices, axis=0)
        sampled_y = np.take(a=y, indices=sampled_indices)

        # Perform T gradient steps
        for t in range(1, T+1):
            i = t - 1   # Index the data from 0, not 1..
            step_size = basic_step_size / t
            w_new = (1-step_size)*self.w

            if sampled_y[i]*np.dot(self.w, sampled_x[i]) < 1:
                w_new = np.add(w_new, step_size * C * sampled_y[i] * sampled_x[i])
            self.w = w_new

    def predict(self, x):
        '''
        Predict label for given x
        :param x: Vectors of data, matching the weight matrix in dimensions
        :return: The predicted label of the classifier
        '''
        assert (x.shape[1] == len(self.w))
        prediction = np.sign(np.dot(x, self.w))
        return prediction


def calculate_accuracy(predictions, labels):
    ''' Calculate the accuracy for the SGD SVM algorithm by percentage of correct predictions '''
    processed_results = np.subtract(predictions, labels)
    correct_results = np.count_nonzero(processed_results == 0)

    return float(correct_results) / len(predictions) * 100


def plot(x, y, legend, x_min, x_max, y_min, y_max, xlabel, ylabel, title):
    '''
    Plots the X & Y samples on the canvas, using min - max range for the axes, X is a logarithmic axis
    '''
    plt.plot(x, y, color='red', label=legend)
    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(title + ".png")
    plt.show()


def ex_4_a():
    feature_dim = hw4.train_data.shape[1]
    T = 1000
    C = 1
    iterations_per_validation = 10
    log_resolution_min = -5
    log_resolution_max = 2
    step_range = [10 ** x for x in np.arange(log_resolution_min, log_resolution_max+1, 0.25)]
    validation_accuracies = []

    for basic_step_size in step_range:
        accumulated_validation_accuracy = 0

        for _ in range(iterations_per_validation):
            classifier = SGD_SVM(dim=feature_dim)
            classifier.train(x=hw4.train_data, y=hw4.train_labels, C=C, T=T, basic_step_size=basic_step_size)
            validation_predictions = classifier.predict(x=hw4.validation_data)
            validation_accuracy = calculate_accuracy(validation_predictions, hw4.validation_labels)
            accumulated_validation_accuracy += validation_accuracy

        validation_accuracies.append(accumulated_validation_accuracy / iterations_per_validation)   # Plot the average

    y_min = min(validation_accuracies) - 1
    y_max = max(validation_accuracies) + 1
    plot(x=step_range, y=validation_accuracies, legend='Validation Accuracies',
         x_min=step_range[0], x_max=step_range[-1], y_min=y_min, y_max=y_max,
         xlabel=r'$\eta_0$' + ' [Step Size]', ylabel='Accuracy', title='Section 4.A - SGD-SVM Accuracy for ' + r'$\eta_0$')

def ex_4_b():
    pass

def ex_4_c():
    pass

def ex_4_d():
    pass

parser = argparse.ArgumentParser(description='Choose a subsection')
parser.add_argument('--section', help='A section of the ex')
args = parser.parse_args()
if args.section == '1':
    ex_4_a()
elif args.section == '2':
    ex_4_b()
elif args.section == '3':
    ex_4_c()
elif args.section == '4':
    ex_4_d()