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
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, \
                            classification_report
from keras import backend as K
from keras.layers import Input, Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.layers.advanced_activations import ELU
from datetime import datetime
import matplotlib
import itertools
matplotlib.use('Agg')
from matplotlib import pyplot as plt

''' This code is based on Núñez-Marcos, A., Azkune, G., & Arganda-Carreras,
    I. (2017). "Vision-Based Fall Detection with Convolutional Neural Networks"
    Wireless Communications and Mobile Computing, 2017.
    Also, new features were added by Gabriel Pellegrino Silva working in
    Semantix.
'''

''' Documentation: class Train

    This class has a few methods:

    pre_train_cross
    pre_train
    cross_train
    train
    evaluate
    plot_training_info

    The methods that should be called outside of this class are:

    cross_train: perform a n_split cross_train on files passed by
    argument

    train: perfom a simple trainment on files passsed by argument
'''
class Train:

    def __init__(self, epochs, learning_rate,
    classes, weight, mini_batch_size, id, batch_norm, streams, fold_norm, kfold):

        '''
            Necessary parameters to train

        '''

        self.features_key = 'features'
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'
        self.streams = streams
        self.classes = classes
        self.classes.sort()
        self.num_classes = len(classes)
        self.fold_norm = fold_norm
        self.kfold = kfold

        self.id = id

        self.num_features = 4096
        self.sliding_height = 10
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_0 = weight
        self.mini_batch_size = mini_batch_size
        self.batch_norm = batch_norm

###     Number of streams for each combination

        self.num_streams = dict()


###     This three dicts are to see which strategy gives best results in
###     training stage

        self.taccuracies_avg = dict()
        self.taccuracies_avg_svm = dict()
        self.taccuracies_svm_1 = dict()
        self.taccuracies_svm_2 = dict()

###     This others dicts will give the parameters to all strategies

        self.sensitivities_avg = dict()
        self.specificities_avg = dict()
        self.fars_avg = dict()
        self.mdrs_avg = dict()
        self.accuracies_avg = dict()

        self.sensitivities_avg_svm = dict()
        self.specificities_avg_svm = dict()
        self.fars_avg_svm = dict()
        self.mdrs_avg_svm = dict()
        self.accuracies_avg_svm = dict()

        self.sensitivities_svm_1 = dict()
        self.specificities_svm_1 = dict()
        self.fars_svm_1 = dict()
        self.mdrs_svm_1 = dict()
        self.accuracies_svm_1 = dict()

        self.sensitivities_svm_2 = dict()
        self.specificities_svm_2 = dict()
        self.fars_svm_2 = dict()
        self.mdrs_svm_2 = dict()
        self.accuracies_svm_2 = dict()

    def calc_metrics(self, num_streams, y_test, y_train, test_predicteds,
                    train_predicteds, key):

        avg_predicted = np.zeros(shape=(len(y_test), len(self.classes)), dtype=np.float)
        train_avg_predicted = np.zeros(shape=(len(y_train), len(self.classes)),  dtype=np.float)

        for j in range(len(y_test)):
            for i in range(num_streams):
                for k in range(len(self.classes)):
                    avg_predicted[j][k] += (test_predicteds[i][j][k] / num_streams)

        for j in range(len(y_train)):
            for i in range(num_streams):
                for k in range(len(self.classes)):
                    train_avg_predicted[j][k] += (train_predicteds[i][j][k] / num_streams)

        test_predicteds = np.asarray(test_predicteds)
        train_predicteds = np.asarray(train_predicteds)
        svm_cont_2_test_predicteds = np.asarray([list(test_predicteds[:, i, j]) for i in range(len(y_test)) for j in range(len(self.classes))])
        svm_cont_2_test_predicteds = svm_cont_2_test_predicteds.reshape(len(y_test), len(self.classes) * num_streams)

        svm_cont_2_train_predicteds = np.asarray([list(train_predicteds[:, i, j]) for i in range(len(y_train)) for j in range(len(self.classes))])
        svm_cont_2_train_predicteds = svm_cont_2_train_predicteds.reshape(len(y_train), len(self.classes) * num_streams)

        svm_cont_1_test_predicteds = []
        svm_cont_1_train_predicteds = []
        for i in range(num_streams):
            aux_svm = svm.SVC(class_weight=None, gamma='auto')
            aux_svm.fit(train_predicteds[i], y_train)

            svm_cont_1_test_predicteds.append(aux_svm.predict(test_predicteds[i]))
            svm_cont_1_train_predicteds.append(aux_svm.predict(train_predicteds[i]))
            joblib.dump(aux_svm, 'svm_' + self.streams[i] + '_1_aux.pkl')

        svm_cont_1_test_predicteds = np.asarray(svm_cont_1_test_predicteds)
        svm_cont_1_train_predicteds = np.asarray(svm_cont_1_train_predicteds)

        svm_cont_1_test_predicteds = np.reshape(svm_cont_1_test_predicteds, svm_cont_1_test_predicteds.shape[::-1])
        svm_cont_1_train_predicteds = np.reshape(svm_cont_1_train_predicteds, svm_cont_1_train_predicteds.shape[::-1])

####
####        TREINAMENTO COM MEDIA E MAX
####

        print('EVALUATE WITH average and max')
        tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy = self.evaluate_max(np.array(avg_predicted, copy=True), y_test)

        self.sensitivities_avg[key].append(sensitivity)
        self.specificities_avg[key].append(specificity)
        self.fars_avg[key].append(fpr)
        self.mdrs_avg[key].append(fnr)
        self.accuracies_avg[key].append(accuracy)

        print('TRAIN WITH average and max')
        tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy = self.evaluate_max(np.array(train_avg_predicted, copy=True), y_train)

        self.taccuracies_avg[key].append(accuracy)

####
####        TREINAMENTO COM MEDIA E SVM
####

        # Not using. Instead: class_weight=None
        #class_weight = dict()
        #for i in range(len(self.classes)):
        #    class_weight[i] = 1

        clf_avg = svm.SVC(class_weight=None, gamma='auto')
        clf_avg.fit(train_avg_predicted, y_train)
        avg_predicted = clf_avg.predict(avg_predicted)
        train_avg_predicted = clf_avg.predict(train_avg_predicted)

        joblib.dump(clf_avg, 'svm_avg_' + key + '.pkl')

        del clf_avg
        gc.collect()

        print('EVALUATE WITH average and SVM')
        tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy = self.evaluate(avg_predicted, y_test)

        self.sensitivities_avg_svm[key].append(sensitivity)
        self.specificities_avg_svm[key].append(specificity)
        self.fars_avg_svm[key].append(fpr)
        self.mdrs_avg_svm[key].append(fnr)
        self.accuracies_avg_svm[key].append(accuracy)

        print('TRAIN WITH average and SVM')
        tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy = self.evaluate(train_avg_predicted, y_train)
        self.taccuracies_avg_svm[key].append(accuracy)

####
####        TREINAMENTO CONTINUO E SVM 2
####
####
####        EXAMPLE:
####
####        train_predicteds is the result of our CNN
####        train_predicteds = np.asarray([
####            [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.7, 0.8, 0.9]],
####            [[0.3, 0.1, 0.4], [0.4, 0.5, 0.6], [0.5, 0.6, 0.1]]])
####
####        train_predicteds[x, y, z]:
####
####        x -> streams domain
####        y -> number of data
####        z -> number of classes
####
####        TRANSFORMED TO
####        svm_cont_2_train_predicteds = array([
####                [ 0.1,  0.3,
####                  0.2,  0.1,
####                  0.3,  0.4],
####
####                [ 0.5,  0.4,
####                  0.6,  0.5,
####                  0.7,  0.6],
####
####                [ 0.7,  0.5,
####                  0.8,  0.6,
####                  0.9,  0.1]])
####
####        x -> number of data
####        y -> number of classes
####        z -> number of streams
####
####        Now     svm_cont_2_train_predicteds[0] has a label,
####                svm_cont_2_train_predicteds[1] has a label
####                svm_cont_2_train_predicteds[2] has a label
####        and so on... Because we're analysing the effects of the use of
####        multiple streams and the ways of combining theirs results
####
####        TODO: a third option that can be implemented is
####
####        apply a svm to fit for every class, for each data, the vector of
####        values containing informations from all streams.
####        and fit it to say if this input goes is a 0 or a 1, if it's likely
####        to be this class or not. And then, feed this binary vector (one
####        position for each class) to another svm, to fit which class is rlly
####        true.
####

        clf_continuous = svm.SVC(class_weight=None, gamma='auto')

        clf_continuous.fit(svm_cont_2_train_predicteds, y_train)

        test_2_continuous = clf_continuous.predict(svm_cont_2_test_predicteds)
        train_2_continuous = clf_continuous.predict(svm_cont_2_train_predicteds)

        joblib.dump(clf_continuous, 'svm_' + key + '_cont_2.pkl')
        print('EVALUATE WITH continuous values and SVM 2')
        tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy = self.evaluate(test_2_continuous, y_test)

        self.sensitivities_svm_2[key].append(sensitivity)
        self.specificities_svm_2[key].append(specificity)
        self.fars_svm_2[key].append(fpr)
        self.mdrs_svm_2[key].append(fnr)
        self.accuracies_svm_2[key].append(accuracy)

        print('TRAIN WITH continuous values and SVM 2')
        tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy = self.evaluate(train_2_continuous, y_train)
        self.taccuracies_svm_2[key].append(accuracy)

        del clf_continuous
        gc.collect()

####
####        TREINAMENTO CONTINUO E SVM 1
####

        clf_continuous = svm.SVC(class_weight=None, gamma='auto')
        clf_continuous.fit(svm_cont_1_train_predicteds, y_train)

        test_1_continuous = clf_continuous.predict(svm_cont_1_test_predicteds)
        train_1_continuous = clf_continuous.predict(svm_cont_1_train_predicteds)

        joblib.dump(clf_continuous, 'svm_' + key + '_cont_1.pkl')
        print('EVALUATE WITH continuous values and SVM 1')
        tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy = self.evaluate(test_1_continuous, y_test)

        self.sensitivities_svm_1[key].append(sensitivity)
        self.specificities_svm_1[key].append(specificity)
        self.fars_svm_1[key].append(fpr)
        self.mdrs_svm_1[key].append(fnr)
        self.accuracies_svm_1[key].append(accuracy)

        print('TRAIN WITH continuous values and SVM 1')
        tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy = self.evaluate(train_1_continuous, y_train)
        self.taccuracies_svm_1[key].append(accuracy)

        del clf_continuous
        gc.collect()

    def real_cross_train(self, nsplits):

        h5features_start = h5py.File(self.streams[0] + '_features_' + self.id + '.h5', 'r')
        h5labels_start = h5py.File(self.streams[0] + '_labels_' + self.id + '.h5', 'r')
        h5samples_start = h5py.File(self.streams[0] + '_samples_'+ self.id + '.h5', 'r')
        h5num_start = h5py.File(self.streams[0] + '_num_'+ self.id + '.h5', 'r')

        all_features_start = h5features_start[self.features_key]
        all_labels_start = np.asarray(h5labels_start[self.labels_key])
        all_samples_start = np.asarray(h5samples_start[self.samples_key])
        all_num_start = np.asarray(h5num_start[self.num_key])

        videos_index = []

        for i in range(len(self.classes)):
            videos_index.append([x for x in range(all_num_start[i][0])])

        labels = []
        labels_start_index = [0]
        kf = []
        for i in range(len(self.classes)):
            atual = labels_start_index[-1]
            kf.append(KFold(n_splits=nsplits, shuffle=True))
            labels.append(np.where(all_labels_start==i)[0])
            labels_start_index.append(len(labels[-1]) + atual)
            print("Labels da classe " + self.classes[i]+ " ", end='')
            print(labels[-1])

        streams_combinations = []
        for L in range(0, len(self.streams)+1):
            for subset in itertools.combinations(self.streams, L):
                if len(list(subset)) != 0:
                    streams_combinations.append(list(subset))

        for comb in streams_combinations:
            key = ''.join(comb)

            self.num_streams[key] = len(comb)

            self.taccuracies_avg[key] = []
            self.taccuracies_avg_svm[key] = []
            self.taccuracies_svm_1[key] = []
            self.taccuracies_svm_2[key] = []

            self.sensitivities_avg[key] = []
            self.specificities_avg[key] = []
            self.fars_avg[key] = []
            self.mdrs_avg[key] = []
            self.accuracies_avg[key] = []

            self.sensitivities_avg_svm[key] = []
            self.specificities_avg_svm[key] = []
            self.fars_avg_svm[key] = []
            self.mdrs_avg_svm[key] = []
            self.accuracies_avg_svm[key] = []

            self.sensitivities_svm_1[key] = []
            self.specificities_svm_1[key] = []
            self.fars_svm_1[key] = []
            self.mdrs_svm_1[key] = []
            self.accuracies_svm_1[key] = []

            self.sensitivities_svm_2[key] = []
            self.specificities_svm_2[key] = []
            self.fars_svm_2[key] = []
            self.mdrs_svm_2[key] = []
            self.accuracies_svm_2[key] = []

        all_train_index_label = []
        all_test_index_label = []
        # CROSS-VALIDATION: Stratified partition of the dataset into train/test setes
        for i in range(len(self.classes)):
            fold_train_index_label = []
            fold_test_index_label = []
            print("Analisando a classe: " + self.classes[i])
            print("Quantidade de labels: " + str(len(labels[i])))
            for (a, b) in kf[i].split(all_features_start[labels[i], ...]):
                a = np.asarray(a)
                b = np.asarray(b)

                fold_train_index_label.append(a)
                fold_test_index_label.append(b)

            all_train_index_label.append(fold_train_index_label)
            all_test_index_label.append(fold_test_index_label)

        for counter in range(nsplits):
            K.clear_session()
            for stream in self.streams:
                print("Analisando a stream " + stream)
                h5features = h5py.File(stream + '_features_' + self.id + '.h5', 'r')
                h5labels = h5py.File(stream + '_labels_' + self.id + '.h5', 'r')
                all_features = h5features[self.features_key]
                all_labels = np.asarray(h5labels[self.labels_key])

                X_train = np.empty(shape=(0, self.num_features), dtype=int)
                y_train = np.empty(shape=(0,1), dtype=int)
                X_test = np.empty(shape=(0, self.num_features), dtype=int)
                y_test = np.empty(shape=(0,1), dtype=int)
                for i in range(len(self.classes)):
                    X_train = np.concatenate((X_train, all_features[labels[i], ...][all_train_index_label[i][counter], ...]))
                    y_train = np.concatenate((y_train, all_labels[labels[i], ...][all_train_index_label[i][counter], ...]))
                    X_test = np.concatenate((X_test, all_features[labels[i], ...][all_test_index_label[i][counter], ...]))
                    y_test = np.concatenate((y_test, all_labels[labels[i], ...][all_test_index_label[i][counter], ...]))

                # Balance the number of positive and negative samples so that there is the same amount of each of them
                all_ = []
                len_min = float("inf")
                ind_min = -1

                for i in range(len(self.classes)):
                    all_.append(np.where(y_train==i)[0])
                    print("Tamanho da classe " + self.classes[i] + " " + str(len(all_[-1])))
                    if len(all_[-1]) < len_min:
                        ind_min = i
                        len_min = len(all_[-1])


                for i in range(len(self.classes)):
                    all_[i] = np.random.choice(all_[i], len_min, replace=False)

                allin = np.empty(0, dtype=int)
                for i in range(len(self.classes)):
                    allin = np.concatenate((allin.flatten(), all_[i].flatten()))

                allin.sort()
                X_train = X_train[allin,...]
                y_train = y_train[allin]


                classifier = self.set_classifier_vgg16()
                class_weight = dict()

                for i in range(len(self.classes)):
                    class_weight[i] = 1

                # Batch training
                if self.mini_batch_size == 0:
                    history = classifier.fit(X_train, to_categorical(y_train),
                            validation_data=(X_test, to_categorical(y_test)),
                            batch_size=X_train.shape[0], epochs=self.epochs,
                            shuffle='batch', class_weight=class_weight)
                else:
                    history = classifier.fit(X_train, to_categorical(y_train),
                            validation_data=(X_test, to_categorical(y_test)),
                            batch_size=self.mini_batch_size, nb_epoch=self.epochs,
                            shuffle=True, class_weight=class_weight, verbose=2)

                exp = 'lr{}_batchs{}_batchnorm{}_w0_{}'.format(self.learning_rate, self.mini_batch_size, self.batch_norm, self.weight_0)
                self.plot_training_info(exp, ['accuracy', 'loss'], True,
                                   history.history)

                classifier.save(stream + '_classifier_' + self.id + '.h5')
                h5features.close()
                h5labels.close()

            test_predicteds = dict()
            train_predicteds = dict()

            for comb in streams_combinations:
                key = ''.join(comb)
                test_predicteds[key] = []
                train_predicteds[key] = []

            for stream in self.streams:
                h5features = h5py.File(stream + '_features_' + self.id + '.h5', 'r')
                h5labels = h5py.File(stream + '_labels_' + self.id + '.h5', 'r')
                all_features = h5features[self.features_key]
                all_labels = np.asarray(h5labels[self.labels_key])
                classifier = load_model(stream + '_classifier_' + self.id + '.h5')

                X_train = np.empty(shape=(0, self.num_features), dtype=int)
                y_train = np.empty(shape=(0, 1), dtype=int)
                X_test = np.empty(shape=(0, self.num_features), dtype=int)
                y_test = np.empty(shape=(0, 1), dtype=int)
                for i in range(len(self.classes)):
                    X_train = np.concatenate((X_train, all_features[labels[i], ...][all_train_index_label[i][counter], ...]))
                    y_train = np.concatenate((y_train, all_labels[labels[i], ...][all_train_index_label[i][counter], ...]))
                    X_test = np.concatenate((X_test, all_features[labels[i], ...][all_test_index_label[i][counter], ...]))
                    y_test = np.concatenate((y_test, all_labels[labels[i], ...][all_test_index_label[i][counter], ...]))

                # Balance the number of positive and negative samples so that there is the same amount of each of them
                all_ = []
                len_min = float("inf")
                ind_min = -1
                for i in range(len(self.classes)):
                    all_.append(np.asarray(np.where(y_train==i)[0]))
                    if len(all_[-1]) < len_min:
                        ind_min = i
                        len_min = len(all_[-1])

                for i in range(len(self.classes)):
                    all_[i] = np.random.choice(all_[i], len_min, replace=False)

                allin = np.empty(0, dtype=int)
                for i in range(len(self.classes)):
                    allin = np.concatenate((allin.flatten(), all_[i].flatten()))

                allin.sort()
                X_train = X_train[allin,...]
                y_train = y_train[allin]

                train_predicted = []
                test_predicted = []

                for train in X_train:
                    pred = classifier.predict(np.asarray(train.reshape(1, -1)))
                    pred = pred.flatten()
                    train_predicted.append(pred)

                for test in X_test:
                    pred = classifier.predict(np.asarray(test.reshape(1, -1)))
                    pred = pred.flatten()
                    test_predicted.append(pred)

                test_predicted = np.asarray(test_predicted)
                train_predicted = np.asarray(train_predicted)

                for key in list(test_predicteds.keys()):
                    if stream in key:
                        test_predicteds[key].append(test_predicted)
                        train_predicteds[key].append(train_predicted)

                h5features.close()
                h5labels.close()

            for key in list(test_predicteds.keys()):
                print('########## TESTS WITH  ' + key)
                self.calc_metrics(self.num_streams[key], y_test, y_train,
                        test_predicteds[key], train_predicteds[key], key)

        h5features_start.close()
        h5labels_start.close()
        sensitivities_best = dict()
        specificities_best = dict()
        accuracies_best = dict()
        fars_best = dict()
        mdrs_best = dict()
        best_acc = dict()
        v = dict()

        for comb in streams_combinations:
            key = ''.join(comb)

            sensitivities_best[key] = []
            specificities_best[key] = []
            accuracies_best[key] = []
            fars_best[key] = []
            mdrs_best[key] = []

            best_acc[key] = -1
            v[key] = -1

        sensitivities_final = []
        specificities_final = []
        accuracies_final = []
        fars_final = []
        mdrs_final = []

        final_acc = -1
        final = None
        for key in list(self.taccuracies_avg.keys()):
            print('########## BESTS WITH  ' + ''.join(key))
            for i in range(nsplits):

                if self.taccuracies_avg[key][i] > best_acc[key]:
                    best_acc[key] = self.taccuracies_avg[key][i]
                    v[key] = 0
                if self.taccuracies_avg_svm[key][i] > best_acc[key]:
                    best_acc[key] = self.taccuracies_avg_svm[key][i]
                    v[key] = 1
                if self.taccuracies_svm_1[key][i] > best_acc[key]:
                    best_acc[key] = self.taccuracies_svm_1[key][i]
                    v[key] = 2
                if self.taccuracies_svm_2[key][i] > best_acc[key]:
                    best_acc[key] = self.taccuracies_svm_2[key][i]
                    v[key] = 3

                if v[key] == 0:

                    print("AVERAGE IS BEST")
                    sensitivities_best[key].append(self.sensitivities_avg[key][i])
                    specificities_best[key].append(self.specificities_avg[key][i])
                    accuracies_best[key].append(self.accuracies_avg[key][i])
                    fars_best[key].append(self.fars_avg[key][i])
                    mdrs_best[key].append(self.mdrs_avg[key][i])
                elif v[key] == 1:

                    print("AVERAGE SVM IS BEST")
                    sensitivities_best[key].append(self.sensitivities_avg_svm[key][i])
                    specificities_best[key].append(self.specificities_avg_svm[key][i])
                    accuracies_best[key].append(self.accuracies_avg_svm[key][i])
                    fars_best[key].append(self.fars_avg_svm[key][i])
                    mdrs_best[key].append(self.mdrs_avg_svm[key][i])
                elif v[key] == 2:

                    print("SVM 1 IS BEST")
                    sensitivities_best[key].append(self.sensitivities_svm_1[key][i])
                    sensitivities_best[key].append(self.sensitivities_svm_1[key][i])
                    specificities_best[key].append(self.specificities_svm_1[key][i])
                    accuracies_best[key].append(self.accuracies_svm_1[key][i])
                    fars_best[key].append(self.fars_svm_1[key][i])
                    mdrs_best[key].append(self.mdrs_svm_1[key][i])

                elif v[key] == 3:

                    print("SVM 2 IS BEST")
                    sensitivities_best[key].append(self.sensitivities_svm_2[key][i])
                    sensitivities_best[key].append(self.sensitivities_svm_2[key][i])
                    specificities_best[key].append(self.specificities_svm_2[key][i])
                    accuracies_best[key].append(self.accuracies_svm_2[key][i])
                    fars_best[key].append(self.fars_svm_2[key][i])
                    mdrs_best[key].append(self.mdrs_svm_2[key][i])

            if best_acc[key] > final_acc:
                final_acc = best_acc[key]
                final = key

            self.print_result('3-stream AVG', self.sensitivities_avg[key], self.specificities_avg[key], self.fars_avg[key], self.mdrs_avg[key], self.accuracies_avg[key])
            self.print_result('3-stream AVG_SVM', self.sensitivities_avg_svm[key], self.specificities_avg_svm[key], self.fars_avg_svm[key], self.mdrs_avg_svm[key], self.accuracies_avg_svm[key])
            self.print_result('3-stream SVM 1', self.sensitivities_svm_1[key], self.specificities_svm_1[key], self.fars_svm_1[key], self.mdrs_svm_1[key], self.accuracies_svm_1[key])
            self.print_result('3-stream SVM 2', self.sensitivities_svm_2[key], self.specificities_svm_2[key], self.fars_svm_2[key], self.mdrs_svm_2[key], self.accuracies_svm_2[key])
            self.print_result('3-stream BEST', sensitivities_best[key], specificities_best[key], fars_best[key], mdrs_best[key], accuracies_best[key])

        print(''.join(final))

    def print_result(self, proc, sensitivities, specificities, fars, mdrs, accuracies):

        print('5-FOLD CROSS-VALIDATION RESULTS ' + proc + '===================')
        print("Sensitivity: %.4f%% (+/- %.4f%%)" % (np.mean(sensitivities), np.std(sensitivities)))
        print("Specificity: %.4f%% (+/- %.4f%%)" % (np.mean(specificities), np.std(specificities)))
        print("FAR: %.4f%% (+/- %.4f%%)" % (np.mean(fars), np.std(fars)))
        print("MDR: %.4f%% (+/- %.4f%%)" % (np.mean(mdrs), np.std(mdrs)))
        print("Accuracy: %.4f%% (+/- %.4f%%)" % (np.mean(accuracies), np.std(accuracies)))

    def video_random_generator(self, stream, test_size):
        random.seed(datetime.now())
        s = h5py.File(stream + '_samples_'+ self.id + '.h5', 'r')
        all_s = np.asarray(s[self.samples_key])
        num = h5py.File(stream + '_num_' + self.id + '.h5', 'r')
        all_num = np.asarray(num[self.num_key])

        test_videos = [ [] for x in range(len(all_num)) ]
        train_videos = [ [] for x in range(len(all_num)) ]

        for video in range(1, len(all_s)):
            all_s[video] += all_s[video-1]

        start = 0
        c_fall = 0
        c_nfall = 0
        for j in range(int(all_num[0][0] * test_size)):

            x = random.randint(start, start + all_num[0][0]-1)
            while x in test_videos[0]:
                x = random.randint(start, start + all_num[0][0]-1)

            test_videos[0].append(x)

        for j in range(start, start + all_num[0][0]):
            if j not in test_videos[0]:
                if j != 0:
                    tam = len(list(range(all_s[j-1][0], all_s[j][0])))
                    c_fall += tam
                else:
                    tam = len(list(range(0, all_s[j][0])))
                    c_fall += tam
                train_videos[0].append(j)

        start += all_num[0][0]
        for i in range(1, len(all_num)):
            for j in range(int(all_num[i][0] * test_size)):

                x = random.randint(start, start + all_num[i][0]-1)
                while x in test_videos[i]:
                    x = random.randint(start, start + all_num[i][0]-1)

                test_videos[i].append(x)

            for j in range(start, start + all_num[i][0]):
                if j not in test_videos[i]:
                    if j != 0:
                        tam = len(list(range(all_s[j-1][0], all_s[j][0])))
                        if i != 0:
                            c_nfall += tam
                    else:
                        tam = len(list(range(0, all_s[j][0])))
                        if i != 0:
                            c_nfall += tam

                    train_videos[i].append(j)

                    if c_nfall >= 100*c_fall:
                        break

            start += all_num[i][0]

        s.close()
        num.close()

        return train_videos, test_videos

    def video_random_split(self, stream, train_videos, test_videos):

        f = h5py.File(stream + '_features_' + self.id + '.h5', 'r')
        all_f = np.asarray(f[self.features_key])
        s = h5py.File(stream + '_samples_'+ self.id + '.h5', 'r')
        all_s = np.asarray(s[self.samples_key])
        l = h5py.File(stream + '_labels_' + self.id + '.h5', 'r')
        all_l = np.asarray(l[self.labels_key])
        num = h5py.File(stream + '_num_' + self.id + '.h5', 'r')
        all_num = np.asarray(num[self.num_key])

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        # For every class
        c_test = 0
        c_train = 0

        for video in range(1, len(all_s)):
            all_s[video] += all_s[video-1]

        for c in range(len(all_num)):

            # Pass through test_videos from c-th class
            for video in test_videos[c]:
                if video != 0:
                    tam = len(list(range(all_s[video-1][0], all_s[video][0])))
                    X_test[c_test:c_test+tam] = all_f[all_s[video-1][0]:all_s[video][0]]
                    y_test[c_test:c_test+tam] = all_l[all_s[video-1][0]:all_s[video][0]]
                else:
                    tam = len(list(range(0, all_s[video][0])))
                    X_test[c_test:c_test+tam] = all_f[0:all_s[video][0]]
                    y_test[c_test:c_test+tam] = all_l[0:all_s[video][0]]
                c_test+=tam

            # Pass through traint_videos from c-th class
            for video in train_videos[c]:
                if video != 0:
                    tam = len(list(range(all_s[video-1][0], all_s[video][0])))
                    X_train[c_train:c_train+tam] = all_f[all_s[video-1][0]:all_s[video][0]]
                    y_train[c_train:c_train+tam] = all_l[all_s[video-1][0]:all_s[video][0]]
                else:
                    tam = len(list(range(0, all_s[video][0])))
                    X_train[c_train:c_train+tam] = all_f[0:all_s[video][0]]
                    y_train[c_train:c_train+tam] = all_l[0:all_s[video][0]]
                c_train+=tam


        s.close()
        f.close()
        l.close()
        num.close()
        return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

    def evaluate_max(self, avg_predicted, y):

        predicted = np.zeros(len(y), dtype=np.float)
        for i in range(len(y)):
            predicted[i] = np.argmax(avg_predicted[i])

        # Array of predictions containing 0, 1, 2, ..., n
        # with n being the total number of classes

        return self.evaluate(predicted, y)

    def evaluate(self, predicted, y):

        # please add this: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        print("Classification report for classifier \n%s\n"
            % (classification_report(y, predicted)))
        print("Confusion matrix:\n%s" % confusion_matrix(y, predicted))

        # Compute metrics and print them
        cm = confusion_matrix(y, predicted, labels=[i for i in range(len(self.classes))])

        # This doesnt't make sense anymore.
        # With multiclasses tp isn't only cm[0][0], for example.
        # Need to be improved.
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp/float(tp+fn)
        fpr = fp/float(fp+tn)
        fnr = fn/float(fn+tp)
        tnr = tn/float(tn+fp)
        try:
            precision = tp/float(tp+fp)
        except ZeroDivisionError:
            precision = 1.0
        sensitivity = tp/float(tp+fn)
        specificity = tn/float(tn+fp)
        try:
            f1 = 2*float(precision*sensitivity)/float(precision+sensitivity)
        except ZeroDivisionError:
            f1 = 1.0

        accuracy = accuracy_score(y, predicted)

        #print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
        #print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
        #print('Sensitivity/Recall: {}'.format(sensitivity))
        #print('Specificity: {}'.format(specificity))
        #print('Precision: {}'.format(precision))
        #print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))
        print('Matthews: {}'.format(matthews_corrcoef(y, predicted)))

        # Store the metrics for this epoch
        return tpr, fpr, fnr, tnr, precision, sensitivity, specificity, f1, accuracy

    def set_classifier_resnet50(self):
        extracted_features = Input(shape=(self.num_features,), dtype='float32',
                                   name='input')
        if self.batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99,
                                   epsilon=0.001)(extracted_features)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(extracted_features)

        x = Dropout(0.9)(x)
        x = Dense(1, name='predictions', kernel_initializer='glorot_uniform')(x)
        x = Activation('sigmoid')(x)

        adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, decay=0.0005)

        classifier = Model(input=extracted_features, output=x,
                           name='classifier')
        classifier.compile(optimizer=adam, loss='binary_crossentropy',
                           metrics=['accuracy'])

        return classifier

    def set_classifier_vgg16(self):
        extracted_features = Input(shape=(self.num_features,), dtype='float32',
                                   name='input')
        if self.batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99,
                                   epsilon=0.001)(extracted_features)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(extracted_features)

        x = Dropout(0.9)(x)
        x = Dense(self.num_features, name='fc2',
                  kernel_initializer='glorot_uniform')(x)
        if self.batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(x)
        x = Dropout(0.8)(x)
        x = Dense(self.num_classes, name='predictions',
                  kernel_initializer='glorot_uniform')(x)
        x = Activation('softmax')(x)

        adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, decay=0.0005)

        classifier = Model(name="classifier", inputs=extracted_features, outputs=x)
        classifier.compile(optimizer=adam, loss='categorical_crossentropy',
                           metrics=['accuracy'])

        return classifier

    def plot_training_info(self, case, metrics, save, history):
        '''
        Function to create plots for train and validation loss and accuracy
        Input:
        * case: name for the plot, an 'accuracy.png' or 'loss.png' will be concatenated after the name.
        * metrics: list of metrics to store: 'loss' and/or 'accuracy'
        * save: boolean to store the plots or only show them.
        * history: History object returned by the Keras fit function.
        '''
        plt.ioff()
        if 'accuracy' in metrics:
            fig = plt.figure()
            plt.plot(history['acc'])
            plt.plot(history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            if save == True:
                plt.savefig(case + 'accuracy.png')
                plt.gcf().clear()
            else:
                plt.show()
            plt.close(fig)

        # summarize history for loss
        if 'loss' in metrics:
            fig = plt.figure()
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            #plt.ylim(1e-3, 1e-2)
            plt.yscale("log")
            plt.legend(['train', 'val'], loc='upper left')
            if save == True:
                plt.savefig(case + 'loss.png')
                plt.gcf().clear()
            else:
                plt.show()
            plt.close(fig)

if __name__ == '__main__':
    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)
    print("For a simple training -nsplits flag isn't used.", file = sys.stderr)
    print("For a cross-training set -nsplits <k>, with k beeing the", file=sys.stderr)
    print("number of folders you want to split up your data.", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)

    argp = argparse.ArgumentParser(description='Do training tasks')
    argp.add_argument("-actions", dest='actions', type=str, nargs=1,
            help='Usage: -actions cross-train', required=True)

    '''
        todo: make this weight_0 (w0) more general for multiple classes
    '''

    '''
        todo: verify if all these parameters are really required
    '''

    argp.add_argument("-streams", dest='streams', type=str, nargs='+',
            help='Usage: -streams spatial temporal (to use 2 streams example)',
            required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+',
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-ep", dest='ep', type=int, nargs=1,
            help='Usage: -ep <num_of_epochs>', required=True)
    argp.add_argument("-lr", dest='lr', type=float, nargs=1,
            help='Usage: -lr <learning_rate_value>', required=True)
    argp.add_argument("-w0", dest='w0', type=float, nargs=1,
            help='Usage: -w0 <weight_for_fall_class>', required=True)
    argp.add_argument("-mini_batch", dest='mini_batch', type=int, nargs=1,
            help='Usage: -mini_batch <mini_batch_size>', required=True)
    argp.add_argument("-id", dest='id', type=str, nargs=1,
        help='Usage: -id <identifier_to_this_features_and_classifier>',
        required=True)
    argp.add_argument("-batch_norm", dest='batch_norm', type=bool, nargs=1,
        help='Usage: -batch_norm <True/False>', required=True)
    argp.add_argument("-fold_norm", dest='fold_norm', type=int, nargs=1,
        help='Usage: -fold_norm {0, 1, 2}. 0: no normalization. 1: truncated normalization. 2: random normalization (recommended)',
        required=False, default=2)
    argp.add_argument("-kfold", dest='kfold', type=str, nargs=1,
        help='Usage: -kfold {video, info}. video: kfold in video indexes. info: kfold in frames or stacks (depends on the stream)',
        required=False, default='video')
    argp.add_argument("-nsplits", dest='nsplits', type=int, nargs=1,
    help='Usage: -nsplits <K: many splits you want (>1)>', required=False, default=5)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    train = Train(args.ep[0], args.lr[0], args.classes,
            args.w0[0], args.mini_batch[0], args.id[0],
            args.batch_norm[0], args.streams, args.fold_norm[0],
            args.kfold[0])

    # Need to sort
    args.streams.sort()
    random.seed(1)
    train.real_cross_train(args.nsplits[0])

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
