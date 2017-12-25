import data
import network
import matplotlib.pyplot as plt
import argparse
import numpy as np


def plot_multyple(xs, ys, x_min, x_max, xlabel, ylabel, title):
    for i in reversed(range(len(ys))):
        plt.plot(xs, ys[i], label=str(i))

    plt.xlim([x_min, x_max])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.savefig(title + ".png")
    plt.show()

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

def ex_2_b():
    training_data, test_data = data.load(train_size=10000, test_size=5000)
    net = network.Network([784, 40, 10])
    training_accuracy, training_loss, test_accuracy = net.SGD_with_plot(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)
    epochs = range(30)
    plot(epochs, training_accuracy, -1, 30, 'epoch number', 'training accuracy', 'Training accuracy across epochs')
    plot(epochs, training_loss, -1, 30, 'epoch number', 'training loss', 'Training loss across epochs')
    plot(epochs, test_accuracy, -1, 30, 'epoch number', 'test accuracy', 'Test accuracy across epochs')


def ex_2_c():
    training_data, test_data = data.load(train_size=50000, test_size=10000)
    net = network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

def ex_2_d():
    num_epochs = 30
    training_data, test_data = data.load(train_size=10000, test_size=5000)
    net = network.Network([784, 30, 30, 30, 30, 10])
    gradient_norms_per_epoch = net.SGD_with_gradient_norms(training_data, epochs=num_epochs, mini_batch_size=10000, learning_rate=0.1,
                 test_data=test_data)

    epochs = range(num_epochs)
    gradient_norms = np.array(gradient_norms_per_epoch).transpose()
    plot_multyple(epochs, gradient_norms, -1, num_epochs, 'epoch number', 'gradient norms', 'Gradient norms across epochs')



parser = argparse.ArgumentParser(description='Choose a subsection')
parser.add_argument('--section', help='A section of the ex')
args = parser.parse_args()
if args.section == '1':
    print ("Non existent")
elif args.section == '2':
    ex_2_b()
elif args.section == '3':
    ex_2_c()
elif args.section == '4':
    ex_2_d()