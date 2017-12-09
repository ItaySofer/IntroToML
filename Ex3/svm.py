import numpy as np
import hw3
import argparse
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, c):
        self.model = LinearSVC(C=c, loss='hinge', fit_intercept=False)

    def train(self, x, y):
        self.model.fit(X=x, y=y)

    def predict(self, x):
        return self.model.predict(X=x)


def calculate_accuracy(predictions, labels):
    ''' Calculate the accuracy for the SVM algorithm on the test set against a subset of the training set'''
    processed_results = np.subtract(predictions, labels)
    correct_results = np.count_nonzero(processed_results == 0)

    return float(correct_results) / len(predictions) * 100


def plot(x, y1, y2, y1legend, y2legend,
         x_min, x_max, y_min, y_max,
         xlabel, ylabel, title):
    '''
    Plots the X & Y samples on the canvas, using min - max range for the axes, X is a logarithmic axis
    '''
    plt.plot(x, y1, color='red', label=y1legend)
    plt.plot(x, y2, color='blue', label=y2legend)
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


def ex_3_a():
    C_range = [10**x for x in range(-10, 11, 1)]

    training_accuracies = []
    validation_accuracies = []

    for c in C_range:
        svm = SVM(c)
        svm.train(x=hw3.train_data, y=hw3.train_labels)
        training_predictions = svm.predict(x=hw3.train_data)
        validation_predictions = svm.predict(x=hw3.validation_data)
        training_accuracy = calculate_accuracy(training_predictions, hw3.train_labels)
        validation_accuracy = calculate_accuracy(validation_predictions, hw3.validation_labels)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)

    y_min = min(min(training_accuracies), min(validation_accuracies)) - 1
    y_max = max(max(training_accuracies), max(validation_accuracies)) + 1
    plot(x=C_range, y1=training_accuracies, y1legend='Training Accuracies', y2=validation_accuracies, y2legend='Validation Accuracies',
         x_min=C_range[0], x_max=C_range[-1], y_min=y_min, y_max=y_max,
         xlabel='C [SVM penalty Regularization]', ylabel='Accuracy', title='Section 3.A - SVM Accuracy for C')


def ex_3_c():
    c = 2e-7
    svm = SVM(c)
    svm.train(x=hw3.train_data, y=hw3.train_labels)
    w = svm.model.coef_[0]

    # Normalize w to range [0, 255] for visualization
    weight_mat = np.divide(np.subtract(w, np.min(w)), np.subtract(np.max(w), np.min(w))) * 255
    plt.imshow(np.reshape(weight_mat, (28, 28)), interpolation='nearest', cmap='gray')
    plt.savefig("Section 3.C - SVM Weight Matrix.png")


def ex_3_d():
    c = 2e-7
    svm = SVM(c)
    svm.train(x=hw3.train_data, y=hw3.train_labels)

    test_predictions = svm.predict(x=hw3.test_data)
    test_accuracy = calculate_accuracy(test_predictions, hw3.test_labels)
    print("SVM's accuracy for the entire test set after being trained on entire training set is: " +
          "{:.2f}".format(test_accuracy))


def ex_3_e():
    c = 10
    gamma = 5e-7
    svm = SVC(C=c, gamma=gamma, kernel='rbf')
    svm.fit(X=hw3.train_data, y=hw3.train_labels)

    training_predictions = svm.predict(X=hw3.train_data)
    training_accuracy = calculate_accuracy(training_predictions, hw3.train_labels)
    print("SVC's accuracy for the entire training set after being trained on entire training set is: " +
          "{:.2f}".format(training_accuracy))

    test_predictions = svm.predict(X=hw3.test_data)
    test_accuracy = calculate_accuracy(test_predictions, hw3.test_labels)
    print("SVC's accuracy for the entire test set after being trained on entire training set is: " +
          "{:.2f}".format(test_accuracy))


parser = argparse.ArgumentParser(description='Choose a subsection')
parser.add_argument('--section', help='A section of the ex')
args = parser.parse_args()
if args.section == '1':
    ex_3_a()
elif args.section == '2':
    print ("Non existent")
elif args.section == '3':
    ex_3_c()
elif args.section == '4':
    ex_3_d()
elif args.section == '5':
    ex_3_e()