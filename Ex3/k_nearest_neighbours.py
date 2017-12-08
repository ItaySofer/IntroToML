from sklearn.datasets import fetch_mldata
import numpy.random
import numpy as np
import scipy
import argparse
import matplotlib.pyplot as plt


def fetch_mnist():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    return data, labels


def prepare_dataset(data, labels):
    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]
    return train, train_labels, test, test_labels


def knn(images, labels, query_img, k):
    query_reshaped = query_img.reshape(1, len(query_img))   # Convert to 2d shape for scipy
    distances = scipy.spatial.distance.cdist(images, query_reshaped, 'euclidean')
    labeled_dists = list(zip(labels, distances))
    sorted_dists = sorted(labeled_dists, key=lambda p: p[1])
    k_nearest = sorted_dists[:k]
    cluster = np.argmax(np.bincount(zip(*k_nearest)[0]))    # Choose most frequent label

    return cluster


def calculate_accuracy(train, train_labels, test, test_labels, k):
    correct_predictions = 0
    total_predictions = len(test)

    for query_img, query_label in zip(test, test_labels):
        prediction = knn(train, train_labels, query_img, k)
        if prediction == query_label:
            correct_predictions += 1

    accuracy = (float(correct_predictions) / total_predictions) * 100
    return accuracy


def plot(xs, ys, x_min, x_max, xlabel, ylabel, title):
    '''
    Plots the X & Y samples on the canvas, using min - max range for the axis
    '''
    plt.scatter(xs, ys)
    plt.xlim([x_min, x_max])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True)
    plt.savefig(title + ".png")
    plt.show()


def ex_1_b():
    data, labels = fetch_mnist()
    train, train_labels, test, test_labels = prepare_dataset(data, labels)
    k = 10
    n = 1000
    limited_train = train[:n]
    limited_labels = train_labels[:n]

    accuracy = calculate_accuracy(limited_train, limited_labels, test, test_labels, k)
    print("KNN Accuracy for 1000 training samples on test is: " + "{:.2f}".format(accuracy))


def ex_1_c():
    data, labels = fetch_mnist()
    train, train_labels, test, test_labels = prepare_dataset(data, labels)
    n = 1000
    limited_train = train[:n]
    limited_labels = train_labels[:n]

    accuracies = []

    for k in range(1, 101):
        accuracy = calculate_accuracy(limited_train, limited_labels, test, test_labels, k)
        accuracies.append(accuracy)

    plot(xs=range(1, 101), ys=accuracies, xlabel='k [number of nearest neighbours]', ylabel='Accuracy',
         x_min=-10, x_max=110, title='Section 1.C - Accuracy for k neighbours')


def ex_1_d():
    data, labels = fetch_mnist()
    train, train_labels, test, test_labels = prepare_dataset(data, labels)
    k = 0
    accuracies = []

    for n in xrange(100, 5000, 100):
        limited_train = train[:n]
        limited_labels = train_labels[:n]
        accuracy = calculate_accuracy(limited_train, limited_labels, test, test_labels, k)
        accuracies.append(accuracy)

    plot(xs=range(100, 5000, 100), ys=accuracies, xlabel='n [number of training samples]', ylabel='Accuracy',
         x_min=0, x_max=5100, title='Section 1.D - Accuracy for n training samples')


parser = argparse.ArgumentParser(description='Choose a subsection')
parser.add_argument('--section', help='A section of the ex')
args = parser.parse_args()
if args.section == '1':
    print ("Non existent")
elif args.section == '2':
    ex_1_b()
elif args.section == '3':
    ex_1_c()
elif args.section == '4':
    ex_1_d()
