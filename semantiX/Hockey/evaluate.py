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
    train
    evaluate
    plot_training_info

    The methods that should be called outside of this class are:

    train: perfom a simple trainment on files passsed by argument
'''
class Evaluate:

    def __init__(self, threshold, epochs, learning_rate, 
    weight, mini_batch_size, train_id, test_id, batch_norm):

        '''
            Necessary parameters to train

        '''

        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'

        self.train_id = train_id
        self.test_id = test_id

        self.threshold = threshold
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
        self.taccuracies_svm = dict()

###     This others dicts will give the parameters to all strategies

        self.sensitivities_svm = dict()
        self.specificities_svm = dict()
        self.fars_svm = dict()
        self.mdrs_svm = dict()
        self.accuracies_svm = dict()

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

    def calc_metrics(self, num_streams, y_test, y_train, test_predicteds, 
                    train_predicteds, key):

        avg_predicted = np.zeros(len(y_test), dtype=np.float)
        train_avg_predicted = np.zeros(len(y_train), dtype=np.float)
        clf_train_predicteds = np.zeros(shape=(len(y_train), num_streams), dtype=np.float )

        for j in range(len(y_test)):
            for i in range(num_streams):
                avg_predicted[j] += test_predicteds[i][j] 
            
            avg_predicted[j] /= (num_streams)

        for j in range(len(y_train)):
            for i in range(num_streams):
                train_avg_predicted[j] += train_predicteds[i][j] 

            train_avg_predicted[j] /= (num_streams)
         
        for j in range(len(y_train)):
            clf_train_predicteds[j] = [item[j] for item in train_predicteds]
        
####
####        TREINAMENTO COM TRESHOLD E MEDIA
####

        print('EVALUATE WITH average and threshold')
        tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate_threshold(np.array(avg_predicted, copy=True), y_test)

        self.sensitivities_avg[key].append(recall)
        self.specificities_avg[key].append(specificity)
        self.fars_avg[key].append(fpr)
        self.mdrs_avg[key].append(fnr)
        self.accuracies_avg[key].append(accuracy)

        print('TRAIN WITH average and threshold')
        tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate_threshold(np.array(train_avg_predicted, copy=True), y_train)

        self.taccuracies_avg[key].append(accuracy)

####
####        TREINAMENTO COM MEDIA E SVM
####

        class_weight = {0: self.weight_0, 1: 1}
        clf_avg = svm.SVC(class_weight=class_weight)                                                                 
        clf_avg.fit(train_avg_predicted.reshape(-1, 1), y_train)
        for i in range(len(avg_predicted)):
            avg_predicted[i] = clf_avg.predict(avg_predicted[i])

        joblib.dump(clf_avg, 'svm_avg.pkl') 

        del clf_avg
        gc.collect()

        print('EVALUATE WITH average and SVM')
        tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate(avg_predicted, y_test)

        self.sensitivities_avg_svm[key].append(recall)
        self.specificities_avg_svm[key].append(specificity)
        self.fars_avg_svm[key].append(fpr)
        self.mdrs_avg_svm[key].append(fnr)
        self.accuracies_avg_svm[key].append(accuracy)
        
        print('TRAIN WITH average and SVM')
        tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate(train_avg_predicted, y_train)
        self.taccuracies_avg_svm[key].append(accuracy)


####
####        TREINAMENTO CONTINUO E SVM
####

        clf_continuous = svm.SVC(class_weight=class_weight)

        clf_continuous.fit(clf_train_predicteds, y_train)
       
        avg_continuous = np.array(avg_predicted, copy=True)
        avg_train_continuous = np.array(train_avg_predicted, copy=True)

        for i in range(len(avg_continuous)):
            avg_continuous[i] = clf_continuous.predict(np.asarray([item[i] for item in test_predicteds]).reshape(1, -1))
        
        for i in range(len(avg_train_continuous)):
            avg_train_continuous[i] = clf_continuous.predict(np.asarray([item[i] for item in train_predicteds]).reshape(1, -1))

        joblib.dump(clf_continuous, 'svm_cont.pkl') 
        print('EVALUATE WITH continuous values and SVM')
        tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate(avg_continuous, y_test)
        
        self.sensitivities_svm[key].append(recall)
        self.specificities_svm[key].append(specificity)
        self.fars_svm[key].append(fpr)
        self.mdrs_svm[key].append(fnr)
        self.accuracies_svm[key].append(accuracy)
        
        print('TRAIN WITH continuous values and SVM')
        tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate(avg_train_continuous, y_train)
        self.taccuracies_svm[key].append(accuracy)

        del clf_continuous
        gc.collect()

    def train(self, streams):

        streams_combinations = []
        for L in range(0, len(streams)+1):
            for subset in itertools.combinations(streams, L):
                if len(list(subset)) != 0:
                    streams_combinations.append(list(subset))

        for comb in streams_combinations:
            key = ''.join(comb)
            
            self.num_streams[key] = len(comb)

            self.taccuracies_avg[key] = []
            self.taccuracies_avg_svm[key] = []
            self.taccuracies_svm[key] = []

            self.sensitivities_svm[key] = []
            self.specificities_svm[key] = []
            self.fars_svm[key] = []
            self.mdrs_svm[key] = []
            self.accuracies_svm[key] = []

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

        for stream in streams:
            train_h5features = h5py.File(stream + '_features_' + self.train_id + '.h5', 'r')
            train_h5labels = h5py.File(stream + '_labels_' + self.train_id + '.h5', 'r')
            X_train = train_h5features[self.features_key]
            y_train = np.asarray(train_h5labels[self.labels_key])

            test_h5features = h5py.File(stream + '_features_' + self.test_id + '.h5', 'r')
            test_h5labels = h5py.File(stream + '_labels_' + self.test_id + '.h5', 'r')
            X_test = test_h5features[self.features_key]
            y_test = np.asarray(test_h5labels[self.labels_key])

            # Balance the number of positive and negative samples so that there is the same amount of each of them
            all0 = np.asarray(np.where(y_train==0)[0])
            all1 = np.asarray(np.where(y_train==1)[0])  
            if len(all0) < len(all1):
                all1 = np.random.choice(all1, len(all0), replace=False)
            else:
                all0 = np.random.choice(all0, len(all1), replace=False)

            allin = np.concatenate((all0.flatten(),all1.flatten()))
            allin.sort()
            X_train = X_train[allin,...]
            y_train = y_train[allin]

            classifier = self.set_classifier_vgg16()
            class_weight = {0: self.weight_0, 1: 1}
            # Batch training
            if self.mini_batch_size == 0:
                history = classifier.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        batch_size=X_train.shape[0], epochs=self.epochs, 
                        shuffle='batch', class_weight=class_weight)
            else:
                history = classifier.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        batch_size=self.mini_batch_size, nb_epoch=self.epochs, 
                        shuffle=True, class_weight=class_weight, verbose=2)

            exp = 'lr{}_batchs{}_batchnorm{}_w0_{}'.format(self.learning_rate, self.mini_batch_size, self.batch_norm, self.weight_0)
            self.plot_training_info(exp, ['accuracy', 'loss'], True, 
                               history.history)

            classifier.save(stream + '_classifier_' + self.train_id + '.h5')

            train_h5features.close()
            train_h5labels.close()
            test_h5features.close()
            test_h5labels.close()

        test_predicteds = dict()
        train_predicteds = dict()
        
        for comb in streams_combinations:
            key = ''.join(comb)
            test_predicteds[key] = []
            train_predicteds[key] = []

        for stream in streams:

            train_h5features = h5py.File(stream + '_features_' + self.train_id + '.h5', 'r')
            train_h5labels = h5py.File(stream + '_labels_' + self.train_id + '.h5', 'r')
            X_train = train_h5features[self.features_key]
            y_train = np.asarray(train_h5labels[self.labels_key])

            test_h5features = h5py.File(stream + '_features_' + self.test_id + '.h5', 'r')
            test_h5labels = h5py.File(stream + '_labels_' + self.test_id + '.h5', 'r')
            X_test = test_h5features[self.features_key]
            y_test = np.asarray(test_h5labels[self.labels_key])

            classifier = load_model(stream + '_classifier_' + self.train_id + '.h5')

            # Balance the number of positive and negative samples so that there is the same amount of each of them
            all0 = np.asarray(np.where(y_train==0)[0])
            all1 = np.asarray(np.where(y_train==1)[0])  
            if len(all0) < len(all1):
                all1 = np.random.choice(all1, len(all0), replace=False)
            else:
                all0 = np.random.choice(all0, len(all1), replace=False)

            allin = np.concatenate((all0.flatten(),all1.flatten()))
            allin.sort()
            X_train = X_train[allin,...]
            y_train = y_train[allin]

            test_predicted = np.asarray(classifier.predict(np.asarray(X_test)))
            train_predicted = np.asarray(classifier.predict(np.asarray(X_train)))

            for key in list(test_predicteds.keys()):
                if stream in key:
                    test_predicteds[key].append(test_predicted)
                    train_predicteds[key].append(train_predicted)

            train_h5features.close()
            train_h5labels.close()
            test_h5features.close()
            test_h5labels.close()

        for key in list(test_predicteds.keys()):
            print('########## TESTS WITH  ' + ''.join(key))
            self.calc_metrics(self.num_streams[key], y_test, y_train, 
                    test_predicteds[key], train_predicteds[key], key)
        
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
            for i in range(1):

                if self.taccuracies_avg[key][i] > best_acc[key]:
                    best_acc[key] = self.taccuracies_avg[key][i]
                    v[key] = 0
                if self.taccuracies_avg_svm[key][i] > best_acc[key]:
                    best_acc[key] = self.taccuracies_avg_svm[key][i]
                    v[key] = 1
                if self.taccuracies_svm[key][i] > best_acc[key]:
                    best_acc[key] = self.taccuracies_svm[key][i]
                    v[key] = 2

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
                    specificities_best[key].append(self.specificities_avg_svmt[key][i])
                    accuracies_best[key].append(self.accuracies_avg_svm[key][i])
                    fars_best[key].append(self.fars_avg_svm[key][i])
                    mdrs_best[key].append(self.mdrs_avg_svm[key][i])
                elif v[key] == 2:

                    print("SVM IS BEST")
                    sensitivities_best[key].append(self.sensitivities_svm[key][i])
                    sensitivities_best[key].append(self.sensitivities_svm[key][i])
                    specificities_best[key].append(self.specificities_svm[key][i])
                    accuracies_best[key].append(self.accuracies_svm[key][i])
                    fars_best[key].append(self.fars_svm[key][i])
                    mdrs_best[key].append(self.mdrs_svm[key][i])

            if best_acc[key] > final_acc:
                final_acc = best_acc[key]
                final = key
        
            self.print_result('3-stream AVG', self.sensitivities_avg[key], self.specificities_avg[key], self.fars_avg[key], self.mdrs_avg[key], self.accuracies_avg[key])
            self.print_result('3-stream AVG_SVM', self.sensitivities_avg_svm[key], self.specificities_avg_svm[key], self.fars_avg_svm[key], self.mdrs_avg_svm[key], self.accuracies_avg_svm[key])
            self.print_result('3-stream SVM', self.sensitivities_svm[key], self.specificities_svm[key], self.fars_svm[key], self.mdrs_svm[key], self.accuracies_svm[key])
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
        s = h5py.File(stream + '_samples_'+ self.train_id + '.h5', 'r')
        all_s = np.asarray(s[self.samples_key])
        num = h5py.File(stream + '_num_' + self.train_id + '.h5', 'r')
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

        f = h5py.File(stream + '_features_' + self.train_id + '.h5', 'r')
        all_f = np.asarray(f[self.features_key])
        s = h5py.File(stream + '_samples_'+ self.train_id + '.h5', 'r')
        all_s = np.asarray(s[self.samples_key])
        l = h5py.File(stream + '_labels_' + self.train_id + '.h5', 'r')
        all_l = np.asarray(l[self.labels_key])
        num = h5py.File(stream + '_num_' + self.train_id + '.h5', 'r')
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

    def evaluate_threshold(self, predicted, _y2):

       for i in range(len(predicted)):
           if predicted[i] < self.threshold:
               predicted[i] = 0
           else:
               predicted[i] = 1
       #  Array of predictions 0/1

       return self.evaluate(predicted, _y2)

    def evaluate(self, predicted, _y2):

        predicted = np.asarray(predicted).astype(int)
        # Compute metrics and print them
        cm = confusion_matrix(_y2, predicted,labels=[0,1])
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
        recall = tp/float(tp+fn)
        specificity = tn/float(tn+fp)
        try:
            f1 = 2*float(precision*recall)/float(precision+recall)
        except ZeroDivisionError:
            f1 = 1.0

        accuracy = accuracy_score(_y2, predicted)

        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))   
        print('Sensitivity/Recall: {}'.format(recall))
        print('Specificity: {}'.format(specificity))
        print('Precision: {}'.format(precision))
        print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))

        # Store the metrics for this epoch
        return tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy

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
        x = Dense(1, name='predictions', 
                  kernel_initializer='glorot_uniform')(x)
        x = Activation('sigmoid')(x)
        
        adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, 
                    epsilon=1e-08, decay=0.0005)

        classifier = Model(input=extracted_features, output=x, 
                           name='classifier')
        classifier.compile(optimizer=adam, loss='binary_crossentropy',
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

    argp = argparse.ArgumentParser(description='Do training tasks')

    '''
        todo: make this weight_0 (w0) more general for multiple classes
    '''

    '''
        todo: verify if all these parameters are really required
    '''

    argp.add_argument("-streams", dest='streams', type=str, nargs='+',
            help='Usage: -streams spatial temporal (to use 2 streams example)',
            required=True)
    argp.add_argument("-thresh", dest='thresh', type=float, nargs=1,
            help='Usage: -thresh <x> (0<=x<=1)', required=True)
    argp.add_argument("-ep", dest='ep', type=int, nargs=1,
            help='Usage: -ep <num_of_epochs>', required=True)
    argp.add_argument("-lr", dest='lr', type=float, nargs=1,
            help='Usage: -lr <learning_rate_value>', required=True)
    argp.add_argument("-w0", dest='w0', type=float, nargs=1,
            help='Usage: -w0 <weight_for_fall_class>', required=True)
    argp.add_argument("-mini_batch", dest='mini_batch', type=int, nargs=1,
            help='Usage: -mini_batch <mini_batch_size>', required=True)
    argp.add_argument("-test_id", dest='test_id', type=str, nargs=1,
        help='Usage: -id <identifier_to_test_data>>', 
        required=True)
    argp.add_argument("-train_id", dest='train_id', type=str, nargs=1,
        help='Usage: -id <identifier_to_train_data>>', 
        required=True)
    argp.add_argument("-batch_norm", dest='batch_norm', type=bool, nargs=1,
        help='Usage: -batch_norm <True/False>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    eval = Evaluate(args.thresh[0], args.ep[0], args.lr[0], args.w0[0], 
           args.mini_batch[0], args.train_id[0], args.test_id[0],
           args.batch_norm[0])

    args.streams.sort()
    random.seed(1)
    eval.train(args.streams)

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
