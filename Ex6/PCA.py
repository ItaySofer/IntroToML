import random
import matplotlib.pyplot as plt
import argparse
import hw6
import numpy as np

def PCA(k, data):
    x = data
    U, s, V = np.linalg.svd(x)
    V = np.transpose(V)
    eigvecs = [V[:, i] for i in range(V.shape[1])]
    eigvals = np.square(s)

    return eigvecs[:k], eigvals[:k]



def plot_multyple(xs, ys, x_min, x_max, xlabel, ylabel, title):
    for i in reversed(range(len(ys))):
        plt.plot(xs, ys[i], label=str(i))

    # plt.xlim([x_min, x_max])
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
    # plt.ylim([min(ys), max(ys)])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True)
    plt.savefig(title + ".png")
    plt.show()

def ex_6_a():
    pos = np.array(hw6.train_data_pos)
    pos_mean = np.array(pos).mean(0)
    pos_normalized = np.subtract(pos, pos_mean)
    eigvecs, eigvals = PCA(100, pos_normalized)

    plt.imshow(np.reshape(pos_mean, (28, 28)), interpolation='nearest', cmap='gray')
    plt.savefig("Section 1.A - Mean Image.png")

    for i in range(5):
        plt.imshow(np.reshape(eigvecs[i], (28, 28)), interpolation='nearest', cmap='gray')
        plt.savefig("Section 1.A - Eigenvector #" + str(i+1) + ".png")

    plt.close()
    plot(range(100), eigvals, -1, 101, "dimension", "eigenvalue", "Section 1.A - Eigenvalues as function of dimension")

def ex_6_b():
    neg = np.array(hw6.train_data_neg)
    neg_mean = np.array(neg).mean(0)
    neg_normalized = np.subtract(neg, neg_mean)
    eigvecs, eigvals = PCA(100, neg_normalized)

    plt.imshow(np.reshape(neg_mean, (28, 28)), interpolation='nearest', cmap='gray')
    plt.savefig("Section 1.B - Mean Image.png")

    for i in range(5):
        plt.imshow(np.reshape(eigvecs[i], (28, 28)), interpolation='nearest', cmap='gray')
        plt.savefig("Section 1.B - Eigenvector #" + str(i+1) + ".png")

    plt.close()
    plot(range(100), eigvals, -1, 101, "dimension", "eigenvalue", "Section 1.B - Eigenvalues as function of dimension")



def ex_6_c():
    pos = np.array(hw6.train_data_pos)
    neg = np.array(hw6.train_data_neg)
    all = np.concatenate((pos[:pos.shape[0]/2], neg[:neg.shape[0]/2]), axis=0)
    all_mean = np.array(all).mean(0)
    all_normalized = np.subtract(all, all_mean)
    eigvecs, eigvals = PCA(100, all_normalized)

    plt.imshow(np.reshape(all_mean, (28, 28)), interpolation='nearest', cmap='gray')
    plt.savefig("Section 1.C - Mean Image.png")

    for i in range(5):
        plt.imshow(np.reshape(eigvecs[i], (28, 28)), interpolation='nearest', cmap='gray')
        plt.savefig("Section 1.C - Eigenvector #" + str(i+1) + ".png")

    plt.close()
    plot(range(100), eigvals, -1, 101, "dimension", "eigenvalue", "Section 1.C - Eigenvalues as function of dimension")


def ex_6_d():
    pass

parser = argparse.ArgumentParser(description='Choose a subsection')
parser.add_argument('--section', help='A section of the ex')
args = parser.parse_args()

if args.section == '1':
    ex_6_a()
elif args.section == '2':
    ex_6_b()
elif args.section == '3':
    ex_6_c()
elif args.section == '4':
    ex_6_d()