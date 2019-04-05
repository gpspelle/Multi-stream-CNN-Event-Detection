import argparse
import gc
import math
import sys
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from sklearn.externals import joblib
import numpy as np
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import backend as K
from keras.layers import Input, Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers.advanced_activations import ELU
from datetime import datetime
import matplotlib
import itertools
matplotlib.use('Agg')
from matplotlib import pyplot as plt

classes = ['Falls', 'NotFalls']
features_key = 'features' 
labels_key = 'labels'
samples_key = 'samples'
num_key = 'num'
nsplits = 5

streams = ['spatial', 'temporal', 'ritmo', 'pose']
id = 'FDD'
h5features = h5py.File(streams[0] + '_features_' + id + '.h5', 'r')
h5labels = h5py.File(streams[0] + '_labels_' + id + '.h5', 'r')
all_features = h5features[features_key]
all_labels = np.asarray(h5labels[labels_key])

labels = []
kf = []
for i in range(len(classes)):
    kf.append(KFold(n_splits=nsplits, shuffle=True))
    labels.append(np.asarray(np.where(all_labels==i)[0]))
    labels[-1].sort()

streams_combinations = []
for L in range(0, len(streams)+1):
    for subset in itertools.combinations(streams, L):
        if len(list(subset)) != 0:
            streams_combinations.append(list(subset))

for counter in range(nsplits):
    K.clear_session()
    train_index_label = []
    test_index_label = []
    for i in range(len(classes)):
        for (a, b) in kf[i].split(all_features[labels[i], ...]):
            print(counter, i)
            train_index_label.append(a)
            test_index_label.append(b)
            train_index_label[-1] = np.asarray(train_index_label[-1])
            test_index_label[-1] = np.asarray(test_index_label[-1])
            break

        print(a, b)


    for stream in streams:
        h5features = h5py.File(stream + '_features_' + id + '.h5', 'r')
        h5labels = h5py.File(stream + '_labels_' + id + '.h5', 'r')
        all_features = h5features[features_key]
        all_labels = np.asarray(h5labels[labels_key])

        #h5features.close()
        #h5labels.close()
