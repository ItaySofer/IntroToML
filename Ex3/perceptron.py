import numpy as np
import hw3
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, dim):
        self.w = [0] * dim

    def train(self, x, y):
        assert(x.shape[1] == len(self.w))
        x = self.normalize_data(x)

        for xt, yt in zip(x, y):
            predict = np.sign(np.dot(xt, self.w))
            if predict != yt:
                self.w += yt*xt

    def predict(self, x):
        assert (x.shape[1] == len(self.w))
        x = self.normalize_data(x)
        prediction = np.sign(np.dot(x, self.w))
        return prediction

    def normalize_data(self, x):
        return x / np.linalg.norm(x)


def calculate_accuracy(predictions, test_labels):
    ''' Calculate the accuracy for the Perceptron algorithm on the test set against a subset of the training set'''
    processed_results = np.subtract(predictions, test_labels)
    correct_results = np.count_nonzero(processed_results == 0)

    return float(correct_results) / len(predictions) * 100


def ex_2_a():
    n = [5, 10, 50, 100, 500, 1000, 5000]
    iterations_count = 100
    feature_dim = hw3.train_data.shape[1]

    statistics = []

    for current_n in n:
        accuracies_obtained = []

        for _ in range(1, iterations_count+1):
            perceptron = Perceptron(feature_dim)
            permutation = np.random.permutation(current_n)
            train_data_subset = hw3.train_data[:current_n, :][permutation]
            train_labels_subset = hw3.train_labels[:current_n][permutation]
            perceptron.train(train_data_subset, train_labels_subset)

            predictions = perceptron.predict(hw3.test_data)
            accuracy = calculate_accuracy(predictions, hw3.test_labels)
            accuracies_obtained.append(accuracy)

        # n_data = [n, mean, 5%, 95%]
        n_data = [ current_n,
                  "{:.2f}".format(float(sum(accuracies_obtained)) / iterations_count),
                  np.percentile(accuracies_obtained, 5),
                  np.percentile(accuracies_obtained, 95) ]
        statistics.append(n_data)

    print tabulate(statistics, headers=['n', 'Mean Accuracy', '5%', '95%'])


def ex_2_b():
    feature_dim = hw3.train_data.shape[1]

    perceptron = Perceptron(feature_dim)
    perceptron.train(hw3.train_data, hw3.train_labels)

    # Normalize w to range [0, 255] for visualization
    weight_mat = np.divide(np.subtract(perceptron.w, np.min(perceptron.w)), np.max(perceptron.w)) * 255
    plt.imshow(np.reshape(weight_mat, (28, 28)), interpolation='nearest', cmap='gray')
    plt.savefig("Section 2.B - Perceptron Weight Matrix.png")


def ex_2_c():
    feature_dim = hw3.train_data.shape[1]

    perceptron = Perceptron(feature_dim)
    perceptron.train(hw3.train_data, hw3.train_labels)
    predictions = perceptron.predict(hw3.test_data)
    accuracy = calculate_accuracy(predictions, hw3.test_labels)
    print("Perceptron's accuracy for the entire test set after being trained on entire training set is: " +
          "{:.2f}".format(accuracy))


def ex_2_d():
    total_results_to_store = 5  # Out of these 5 examples we chose indices 80, 506
    feature_dim = hw3.train_data.shape[1]

    perceptron = Perceptron(feature_dim)
    perceptron.train(hw3.train_data, hw3.train_labels)
    predictions = perceptron.predict(hw3.test_data)

    processed_results = np.subtract(predictions, hw3.test_labels)
    wrong_indices = np.nonzero(processed_results)[0]

    for result_count, wrong_index in enumerate(wrong_indices):
        if result_count >= total_results_to_store:
            break
        plt.imshow(np.reshape(hw3.test_data_unscaled[wrong_index], (28, 28)), interpolation='nearest', cmap='gray')
        plt.savefig("Section 2.D - Wrong prediction #" + str(wrong_index) + "_label_" + str(hw3.test_labels[wrong_index]) + ".png")
        result_count += 1


parser = argparse.ArgumentParser(description='Choose a subsection')
parser.add_argument('--section', help='A section of the ex')
args = parser.parse_args()
if args.section == '1':
    ex_2_a()
elif args.section == '2':
    ex_2_b()
elif args.section == '3':
    ex_2_c()
elif args.section == '4':
    ex_2_d()