from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx_pos = numpy.random.RandomState(0).permutation(where(labels == pos)[0])
train_idx_neg = numpy.random.RandomState(0).permutation(where(labels == neg)[0])

# train_data_size = 2000
train_data_pos_unscaled = data[train_idx_pos, :].astype(float)
train_labels_pos = (labels[train_idx_pos] == pos)*2-1

train_data_neg_unscaled = data[train_idx_neg, :].astype(float)
train_labels_neg = (labels[train_idx_neg] == pos)*2-1

#validation_data_unscaled = data[train_idx[6000:], :].astype(float)
#validation_labels = (labels[train_idx[6000:]] == pos)*2-1

# Preprocessing
train_data_pos = sklearn.preprocessing.scale(train_data_pos_unscaled, axis=0, with_std=False)
train_data_neg = sklearn.preprocessing.scale(train_data_neg_unscaled, axis=0, with_std=False)