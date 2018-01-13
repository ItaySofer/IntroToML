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



def plot_multyple(xs, ys, xlabel, ylabel, title):
    plt.scatter(xs[0], ys[0], label=str(8), marker=".")
    plt.scatter(xs[1], ys[1], label=str(0), marker="+")

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
    pos = np.array(hw6.train_data_pos)
    neg = np.array(hw6.train_data_neg)
    all = np.concatenate((pos[:pos.shape[0]/2], neg[:neg.shape[0]/2]), axis=0)
    all_mean = np.array(all).mean(0)
    all_normalized = np.subtract(all, all_mean)
    eigvecs, eigvals = PCA(2, all_normalized)

    v1 = eigvecs[0]
    v1_nomalized = v1 / np.linalg.norm(v1)

    v2 = eigvecs[1]
    v2_nomalized = v2 / np.linalg.norm(v2)

    xs_pos = np.dot(pos, np.transpose(v1_nomalized))
    ys_pos = np.dot(pos, np.transpose(v2_nomalized))

    xs_neg = np.dot(neg, np.transpose(v1_nomalized))
    ys_neg = np.dot(neg, np.transpose(v2_nomalized))

    plot_multyple([xs_pos, xs_neg], [ys_pos, ys_neg], "projection on V1", "projection on V2", "Section 1.D - Projection of 8 and 0 images on first two principal axes")

def ex_6_e():
    pos = np.array(hw6.train_data_pos)
    neg = np.array(hw6.train_data_neg)
    all = np.concatenate((pos[:pos.shape[0] / 2], neg[:neg.shape[0] / 2]), axis=0)
    all_mean = np.array(all).mean(0)
    all_normalized = np.subtract(all, all_mean)
    eigvecs, eigvals = PCA(50, all_normalized)

    images = [pos[0], pos[1], neg[0], neg[1]]
    ks = [10, 30, 50]
    for j in range(4):
        image = images[j]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.suptitle('Section 1.E image #' + str(j+1), fontsize=20)
        ax1.set_title('Original Image')
        ax1.imshow(np.reshape(image, (28, 28)), interpolation='nearest', cmap='gray')
        axs = [ax2, ax3, ax4]
        for i in range(3):
            k = ks[i]
            vT = eigvecs[:k]
            v = np.transpose(vT)
            vvT = np.dot(v, vT)
            x_hat = np.dot(vvT, image)
            ax = axs[i]
            ax.set_title('k=' + str(k))
            ax.imshow(np.reshape(x_hat, (28, 28)), interpolation='nearest', cmap='gray')
        plt.savefig("Section 1.E - reconstruction of image #" + str(j+1) + ".png")
        plt.close()

def ex_6_f():
    pos = np.array(hw6.train_data_pos)
    neg = np.array(hw6.train_data_neg)
    all = np.concatenate((pos[:pos.shape[0] / 2], neg[:neg.shape[0] / 2]), axis=0)
    all_mean = np.array(all).mean(0)
    all_normalized = np.subtract(all, all_mean)
    eigvecs, eigvals = PCA(100, all_normalized)

    xi_norm = [np.linalg.norm(x) for x in all_normalized]
    xi_norm_square = np.square(xi_norm)
    sum_xi_norm_square = sum(xi_norm_square)

    n = all_normalized.shape[0]

    lambda_sum = np.cumsum(eigvals)
    sum_xi_norm_square_array = np.repeat(sum_xi_norm_square, 100)
    objective = np.subtract(sum_xi_norm_square_array, lambda_sum)

    plot(range(1, 101), objective, 0, 101, 'dimension', 'PCA Objective', "Section 1.F - PCA Objective as Function of Dimension")



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
elif args.section == '5':
    ex_6_e()
elif args.section == '6':
    ex_6_f()