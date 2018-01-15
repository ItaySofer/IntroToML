import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error
import hw6


def pca(X, k):
    '''
    :param X: Set of images, mean centered
    :param k: Number of dimensions to reduce to
    :return: k eigenvectors corresponding to k largest eigenvalues
    '''

    X = np.divide(X, np.sqrt(len(X)))
    u, s, vt = np.linalg.svd(X, full_matrices=False)

    # Singular values already sorted in descending order..
    k_largest_eigenvals = s[:k]**2/(len(X) - 1)

    # Rows of vt are eigenvectors of X.T*X
    # k_largest_eigenvecs's columns are the PCs of Cov(X)
    k_largest_eigenvecs = vt[:k, :].T

    return k_largest_eigenvecs, k_largest_eigenvals


def feature_vec_to_img(img):
    '''
    784 dim feature vec to 28x28 img, colors equalized
    :param img:
    :return:
    '''
    normalized = np.divide(np.subtract(img, np.min(img)),
                           np.subtract(np.max(img), np.min(img))) * 255
    return np.reshape(normalized, (28, 28))


def plot(x, y, ylegend, x_min, x_max, y_min, y_max, xlabel, ylabel, title):
    '''
    Plots the X & Y samples on the canvas, using min - max range for the axes
    '''
    plt.plot(x, y, color='red', label=ylegend)
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


def ex_6_a():
    k = 100
    training_eights = hw6.train_data_pos
    pca_basis, variance = pca(X=training_eights, k=k)

    mean_digit = training_eights.mean(axis=0)

    fig = plt.figure()
    mean_plt = fig.add_subplot(2, 5, 1)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                    labelleft='off')
    plt.imshow(feature_vec_to_img(mean_digit), interpolation='nearest', cmap='gray')
    mean_plt.set_title('Mean Image')

    for i in range(1,6):
        next_plt = fig.add_subplot(2, 5, i+5)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                        labelleft='off')
        plt.imshow(feature_vec_to_img(pca_basis[:, i]), interpolation='nearest', cmap='gray')
        next_plt.set_title('u' + str(i))

    plt.savefig("Section 1.A.1 - PCA Basis.png")
    plt.show()

    plot(x=range(1, 101), y=variance, ylegend='Eigenvalues',
         x_min=1, x_max=101, y_min=min(variance) - 0.1,
         y_max=max(variance) + 0.1, xlabel='Dimensions', ylabel='Eigenvalues',
         title='Section 1.A.2 - PCA Eigenvalues')


def ex_6_b():
    k = 100
    training_zeros = hw6.train_data_neg
    pca_basis, variance = pca(X=training_zeros, k=k)

    mean_digit = training_zeros.mean(axis=0)

    fig = plt.figure()
    mean_plt = fig.add_subplot(2, 5, 1)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                    labelleft='off')
    plt.imshow(feature_vec_to_img(mean_digit), interpolation='nearest', cmap='gray')
    mean_plt.set_title('Mean Image')

    for i in range(1,6):
        next_plt = fig.add_subplot(2, 5, i+5)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                        labelleft='off')
        plt.imshow(feature_vec_to_img(pca_basis[:, i]), interpolation='nearest', cmap='gray')
        next_plt.set_title('u' + str(i))

    plt.savefig("Section 1.B.1 - PCA Basis.png")
    plt.show()

    plot(x=range(1, 101), y=variance, ylegend='Eigenvalues',
         x_min=1, x_max=101, y_min=min(variance) - 0.1,
         y_max=max(variance) + 0.1, xlabel='Dimensions', ylabel='Eigenvalues',
         title='Section 1.B.2 - PCA Eigenvalues')


def ex_6_c():
    k = 100
    training_joint = np.concatenate((hw6.train_data_pos, hw6.train_data_neg), axis=0)
    pca_basis, variance = pca(X=training_joint, k=k)

    mean_digit = training_joint.mean(axis=0)

    fig = plt.figure()
    mean_plt = fig.add_subplot(2, 5, 1)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                    labelleft='off')
    plt.imshow(feature_vec_to_img(mean_digit), interpolation='nearest', cmap='gray')
    mean_plt.set_title('Mean Image')

    for i in range(1,6):
        next_plt = fig.add_subplot(2, 5, i+5)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                        labelleft='off')
        plt.imshow(feature_vec_to_img(pca_basis[:, i]), interpolation='nearest', cmap='gray')
        next_plt.set_title('u' + str(i))

    plt.savefig("Section 1.C.1 - PCA Basis.png")
    plt.show()

    plot(x=range(1, 101), y=variance, ylegend='Eigenvalues',
         x_min=1, x_max=101, y_min=min(variance) - 0.1,
         y_max=max(variance) + 0.1, xlabel='Dimensions', ylabel='Eigenvalues',
         title='Section 1.C.2 - PCA Eigenvalues')


def ex_6_d():
    k = 2
    training_joint = np.concatenate((hw6.train_data_pos, hw6.train_data_neg), axis=0)
    pca_basis, variance = pca(X=training_joint, k=k)

    pca_proj = pca_basis.T
    projected_eights = np.dot(pca_proj, hw6.train_data_pos.T)
    projected_zeros = np.dot(pca_proj, hw6.train_data_neg.T)

    plt.scatter(projected_eights[0,:], projected_eights[1,:], color='red', label='8 digits', marker=".")
    plt.scatter(projected_zeros[0,:], projected_zeros[1,:], color='blue', label='0 digits', marker="+")
    x_min = min(min(projected_eights[0,:]), min(projected_zeros[0,:]))
    y_min = min(min(projected_eights[1,:]), min(projected_zeros[1,:]))
    x_max = max(max(projected_eights[0,:]), max(projected_zeros[0,:]))
    y_max = max(max(projected_eights[1,:]), max(projected_zeros[1,:]))
    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])
    plt.xlabel('First PC')
    plt.ylabel('Second PC')
    title = 'Section 1.D - PCA projections on 2 PCs'
    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.savefig(title + ".png")
    plt.show()


def ex_6_e():
    k = 50
    training_joint = np.concatenate((hw6.train_data_pos, hw6.train_data_neg), axis=0)
    pca_basis, variance = pca(X=training_joint, k=k)

    pca_proj = pca_basis.T
    results = [np.concatenate((hw6.train_data_pos[:2,], hw6.train_data_neg[:2,]))]
    originals = results[0].T
    results.append(np.dot(pca_basis[:,:10], np.dot(pca_proj[:10,], originals)).T)
    results.append(np.dot(pca_basis[:,:30], np.dot(pca_proj[:30,], originals)).T)
    results.append(np.dot(pca_basis, np.dot(pca_proj, originals)).T)

    fig = plt.figure()
    fig.suptitle('Reconstructions from PCA basis')
    title_k = [ 10, 30, 50 ]

    for i in range(1,5):
        for j in range(1, 5):
            next_plt = fig.add_subplot(4, 4, (i-1)*4 + j)
            plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off',
                            left='off', labelleft='off')
            plt.imshow(feature_vec_to_img(results[j-1][i-1]), interpolation='nearest', cmap='gray')

            if j == 1:
                title='Original'
            else:
                title='k=' + str(title_k[j-2])
            next_plt.set_title(title)

    plt.savefig("Section 1.E. - PCA projections on smaller subspaces.png")
    plt.show()


def ex_6_f():
    k = 100
    training_joint = np.concatenate((hw6.train_data_pos, hw6.train_data_neg), axis=0)
    pca_basis, variance = pca(X=training_joint, k=k)

    pca_proj = pca_basis.T
    originals = training_joint.T
    errors = []

    for i in range(1, 101):
        reconstructions = np.dot(pca_basis[:, :i], np.dot(pca_proj[:i, ], originals)).T
        mse = mean_squared_error(training_joint, reconstructions)
        errors.append(mse)

    plot(x=range(1, 101), y=errors, ylegend='PCA objective', x_min=1, x_max=101, y_min=min(errors) - 0.1,
         y_max=max(errors) + 0.1, xlabel='k (# of PCs)', ylabel='PCA Objective', title='Section 1.F - PCA Objective')


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