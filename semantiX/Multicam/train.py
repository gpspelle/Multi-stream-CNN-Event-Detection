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
import keras
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import backend as K
from keras.layers import Input, Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers.advanced_activations import ELU
import matplotlib
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

    def __init__(self, threshold, epochs, learning_rate, 
    weight_0, mini_batch_size, id, batch_norm):

        '''
            Necessary parameters to train

        '''

        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'

        self.id = id
        self.classes = ['Falls', 'NotFalls']

        self.threshold = threshold
        self.num_features = 4096
        self.sliding_height = 10
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_0 = weight_0
        self.mini_batch_size = mini_batch_size
        self.batch_norm = batch_norm 

    def cross_train(self, streams):

        # Big TODO's: 
        # 1. it isn't exactly k-fold because the data istn't partitioned once.
        # it's divided in a random way at each self.train call
        # 2. it isn't being chosen a model, but first, a criteria must be find
    
        cams = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7', 'cam8']
        self.real_cross_train(streams, cams)

    def video_cam_split(self, stream, cams, camera):
        f = h5py.File(stream + '_features_' + self.id + '.h5', 'r')
        l = h5py.File(stream + '_labels_' + self.id + '.h5', 'r')

        c_train = 0
        c_test = 0
        for c in self.classes:
            for cam in cams:
                tam = len(list(f[c][cam][cam]))
                if cam == camera:
                    c_test += tam
                else:
                    c_train += tam

        X_train = np.zeros( shape=(c_train, self.num_features), dtype=np.float64)
        y_train = np.zeros( shape=(c_train, 1), dtype=np.int8)
        X_test = np.zeros( shape=(c_test, self.num_features), dtype=np.float64)
        y_test = np.zeros( shape=(c_test, 1), dtype=np.int8)

        c_test = 0
        c_train = 0
        for c in self.classes:
            for cam in cams:
                tam = len(list(f[c][cam][cam]))
                if cam == camera:
                    X_test[c_test:c_test + tam].flat = f[c][camera][camera][0:tam]
                    y_test[c_test:c_test + tam].flat = l[c][camera][camera][0:tam]
                    c_test += tam
                else:
                    X_train[c_train:c_train + tam].flat = f[c][cam][cam][0:tam]
                    y_train[c_train:c_train + tam].flat = l[c][cam][cam][0:tam]
                    c_train += tam
        l.close()

        return X_train, X_test, y_train, y_test

    def real_cross_train(self, streams, cams):
  
        VGG16 = True
        nsplits = 8
        
        taccuracies_t = []
        taccuracies_s = []
        taccuracies_p = []
        taccuracies_avg = []
        taccuracies_avg_svm = []
        taccuracies_svm = []

        sensitivities_t = []
        specificities_t = []
        fars_t = []
        mdrs_t = []
        accuracies_t = []

        sensitivities_p = []
        specificities_p = []
        fars_p = []
        mdrs_p = []
        accuracies_p = []

        sensitivities_s = []
        specificities_s = []
        fars_s = []
        mdrs_s = []
        accuracies_s = []

        sensitivities_svm = []
        specificities_svm = []
        fars_svm = []
        mdrs_svm = []
        accuracies_svm = []

        sensitivities_avg = []
        specificities_avg = []
        fars_avg = []
        mdrs_avg = []
        accuracies_avg = []
                    
        sensitivities_avg_svm = []
        specificities_avg_svm = []
        fars_avg_svm = []
        mdrs_avg_svm = []
        accuracies_avg_svm = []

        for camera in cams:
            predicteds = []
            train_predicteds = []
            K.clear_session()
            for stream in streams:

                if VGG16:
                    classifier = self.set_classifier_vgg16()
                else:
                    classifier = self.set_classifier_resnet50()

                h5features = h5py.File(stream + '_features_' + self.id + '.h5', 'r')
                h5labels = h5py.File(stream + '_labels_' + self.id + '.h5', 'r')
                h5samples = h5py.File(stream + '_samples_' + self.id + '.h5', 'r')

                sensitivities = []
                specificities = []
                fars = []
                mdrs = []
                accuracies = []

                X_train, X_test, y_train, y_test = self.video_cam_split(stream, cams, camera)

                #all0 = np.asarray(np.where(y_train==0)[0])
                #all1 = np.asarray(np.where(y_train==1)[0])
                #all1 = np.random.choice(all1, len(all0), replace=False)
                #allin = np.concatenate((all0.flatten(),all1.flatten()))
                #allin.sort()
                #X_train = np.asarray(X_train[allin,...])
                #y_train = np.asarray(y_train[allin])

                print("###### Data divided in train and test")
                # ==================== TRAINING ========================     
                # weighting of each class: only the fall class gets a different weight
                class_weight = {0: self.weight_0, 1: 1}
                tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
                # Batch training
                if self.mini_batch_size == 0:
                    history = classifier.fit(X_train, y_train, 
                            validation_data=(X_test, y_test), 
                            batch_size=X_train.shape[0], epochs=self.epochs, 
                            shuffle='batch', class_weight=class_weight,
                            callbacks=[tbCallBack])
                else:
                    history = classifier.fit(X_train, y_train, 
                            validation_data=(X_test, y_test), 
                            batch_size=self.mini_batch_size, nb_epoch=self.epochs, 
                            shuffle=True, class_weight=class_weight,
                            callbacks=[tbCallBack])

                print("###### Train ended")
                exp = 'lr{}_batchs{}_batchnorm{}_w0_{}'.format(self.learning_rate, self.mini_batch_size, self.batch_norm, self.weight_0)
                self.plot_training_info(exp, ['accuracy', 'loss'], True, 
                                   history.history)

                classifier.save(stream + '_classifier_' + self.id + '.h5')
                predicted = np.asarray(classifier.predict(np.asarray(X_test)))
                train_predicted = np.asarray(classifier.predict(np.asarray(X_train)))

                predicteds.append(predicted)
                train_predicteds.append(train_predicted)

                print('EVALUATE WITH ' + stream)
                tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate_threshold(predicted, y_test)

                if stream == 'temporal':
                    sensitivities_t.append(recall)
                    specificities_t.append(specificity)
                    fars_t.append(fpr)
                    mdrs_t.append(fnr)
                    accuracies_t.append(accuracy)
                elif stream == 'pose':
                    sensitivities_p.append(recall)
                    specificities_p.append(specificity)
                    fars_p.append(fpr)
                    mdrs_p.append(fnr)
                    accuracies_p.append(accuracy)
                elif stream == 'spatial':
                    sensitivities_s.append(recall)
                    specificities_s.append(specificity)
                    fars_s.append(fpr)
                    mdrs_s.append(fnr)
                    accuracies_s.append(accuracy)
                
                print('TRAIN WITH ' + stream)
                tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate_threshold(train_predicted, y_train)

                if stream == 'temporal':
                    taccuracies_t.append(accuracy)
                elif stream == 'pose':
                    taccuracies_p.append(accuracy)
                elif stream == 'spatial':
                    taccuracies_s.append(accuracy)
                     
            avg_predicted = np.zeros(len(y_test), dtype=np.float)
            train_avg_predicted = np.zeros(len(y_train), dtype=np.float)
            clf_train_predicteds = np.zeros((len(y_train), len(streams)))

            for j in range(len(y_test)):
                for i in range(len(streams)):
                    avg_predicted[j] += predicteds[i][j] 

                avg_predicted[j] /= (len(streams))

            for j in range(len(y_train)):
                for i in range(len(streams)):
                    train_avg_predicted[j] += train_predicteds[i][j] 

                train_avg_predicted[j] /= (len(streams))
             
            for j in range(len(y_train)):
                clf_train_predicteds[j] = [item[j] for item in train_predicteds]
            
            print('EVALUATE WITH average and threshold')
            tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate_threshold(np.array(avg_predicted, copy=True), y_test)

            sensitivities_avg.append(recall)
            specificities_avg.append(specificity)
            fars_avg.append(fpr)
            mdrs_avg.append(fnr)
            accuracies_avg.append(accuracy)

            print('TRAIN WITH average and threshold')
            tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate_threshold(np.array(train_avg_predicted, copy=True), y_train)

            taccuracies_avg.append(accuracy)

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

            sensitivities_avg_svm.append(recall)
            specificities_avg_svm.append(specificity)
            fars_avg_svm.append(fpr)
            mdrs_avg_svm.append(fnr)
            accuracies_avg_svm.append(accuracy)
            
            print('TRAIN WITH average and SVM')
            tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate(train_avg_predicted, y_train)
            taccuracies_avg_svm.append(accuracy)

            clf_continuous = svm.SVC(class_weight=class_weight)

            clf_continuous.fit(clf_train_predicteds, y_train)
            avg_continuous = np.array(avg_predicted, copy=True)
            avg_train_continuous = np.array(train_avg_predicted, copy=True)

            for i in range(len(avg_continuous)):
                avg_continuous[i] = clf_continuous.predict(np.asarray([item[i] for item in predicteds]).reshape(1, -1))
            
            for i in range(len(avg_train_continuous)):
                avg_train_continuous[i] = clf_continuous.predict(np.asarray([item[i] for item in train_predicteds]).reshape(1, -1))

            joblib.dump(clf_continuous, 'svm_cont.pkl') 
            print('EVALUATE WITH continuous values and SVM')
            del clf_continuous
            gc.collect()

            tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate(avg_continuous, y_test)
            
            sensitivities_svm.append(recall)
            specificities_svm.append(specificity)
            fars_svm.append(fpr)
            mdrs_svm.append(fnr)
            accuracies_svm.append(accuracy)
       
            print('TRAIN WITH continuous values and SVM')
            tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.evaluate(avg_train_continuous, y_train)
            taccuracies_svm.append(accuracy)

        sensitivities_best = []
        specificities_best = []
        accuracies_best = []
        fars_best = []
        mdrs_best = []

        best_acc = -1
        v = -1
        for i in range(nsplits):
            if taccuracies_t[i] > best_acc:
                best_acc = taccuracies_t[i]
                v = 0
            if taccuracies_s[i] > best_acc:
                best_acc = taccuracies_s[i]
                v = 1
            if taccuracies_p[i] > best_acc:
                best_acc = taccuracies_p[i]
                v = 2
            if taccuracies_avg[i] > best_acc:
                best_acc = taccuracies_avg[i]
                v = 3
            if taccuracies_avg_svm[i] > best_acc:
                best_acc = taccuracies_avg_svm[i]
                v = 4
            if taccuracies_svm[i] > best_acc:
                best_acc = taccuracies_svm[i]
                v = 5

            if v == 0:
                print("TEMPORAL IS BEST")
                sensitivities_best.append(sensitivities_t[i])
                specificities_best.append(specificities_t[i])
                accuracies_best.append(accuracies_t[i])
                fars_best.append(fars_t[i])
                mdrs_best.append(mdrs_t[i])
            elif v == 1:
                print("SPATIAL IS BEST")
                sensitivities_best.append(sensitivities_s[i])
                specificities_best.append(specificities_s[i])
                accuracies_best.append(accuracies_s[i])
                fars_best.append(fars_s[i])
                mdrs_best.append(mdrs_s[i])
            elif v == 2:
                print("POSE IS BEST")
                sensitivities_best.append(sensitivities_p[i])
                specificities_best.append(specificities_p[i])
                accuracies_best.append(accuracies_p[i])
                fars_best.append(fars_p[i])
                mdrs_best.append(mdrs_p[i])
            elif v == 3:
                print("AVERAGE IS BEST")
                sensitivities_best.append(sensitivities_avg[i])
                specificities_best.append(specificities_avg[i])
                accuracies_best.append(accuracies_avg[i])
                fars_best.append(fars_avg[i])
                mdrs_best.append(mdrs_avg[i])
            elif v == 4:
                print("AVERAGE SVM IS BEST")
                sensitivities_best.append(sensitivities_avg_svm[i])
                specificities_best.append(specificities_avg_svmt[i])
                accuracies_best.append(accuracies_avg_svm[i])
                fars_best.append(fars_avg_svm[i])
                mdrs_best.append(mdrs_avg_svm[i])
            elif v == 5:
                print("SVM IS BEST")
                sensitivities_best.append(sensitivities_svm[i])
                sensitivities_best.append(sensitivities_svm[i])
                specificities_best.append(specificities_svm[i])
                accuracies_best.append(accuracies_svm[i])
                fars_best.append(fars_svm[i])
                mdrs_best.append(mdrs_svm[i])
        
        self.print_result('TEMPORAL', sensitivities_t, specificities_t,fars_t, mdrs_t, accuracies_t)
        self.print_result('SPATIAL', sensitivities_s, specificities_s,fars_s, mdrs_s, accuracies_s)
        self.print_result('POSE', sensitivities_p, specificities_p, fars_p, mdrs_p, accuracies_p)
        self.print_result('AVG', sensitivities_avg, specificities_avg, fars_avg, mdrs_avg, accuracies_avg)
        self.print_result('AVG_SVM', sensitivities_avg_svm, specificities_avg_svm, fars_avg_svm, mdrs_avg_svm, accuracies_avg_svm)
        self.print_result('SVM', sensitivities_svm, specificities_svm, fars_svm, mdrs_svm, accuracies_svm)
        self.print_result('BEST', sensitivities_best, specificities_best, fars_best, mdrs_best, accuracies_best)

    def print_result(self, proc, sensitivities, specificities, fars, mdrs, accuracies):

        print('5-FOLD CROSS-VALIDATION RESULTS ' + proc + '===================')
        print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities), np.std(sensitivities)))
        print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities), np.std(specificities)))
        print("FAR: %.2f%% (+/- %.2f%%)" % (np.mean(fars), np.std(fars)))
        print("MDR: %.2f%% (+/- %.2f%%)" % (np.mean(mdrs), np.std(mdrs)))
        print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))


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
    print("For a simple training -nsplits flag isn't used.", file = sys.stderr)
    print("For a cross-training set -nsplits <k>, with k beeing the", file=sys.stderr)
    print("number of folders you want to split up your data.", file=sys.stderr)
    print("***********************************************************", 
            file=sys.stderr)

    argp = argparse.ArgumentParser(description='Do training tasks')
    argp.add_argument("-actions", dest='actions', type=str, nargs=1,
            help='Usage: -actions train or -actions cross-train', required=True)

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
    argp.add_argument("-id", dest='id', type=str, nargs=1,
        help='Usage: -id <identifier_to_this_features_and_classifier>', 
        required=True)
    argp.add_argument("-batch_norm", dest='batch_norm', type=bool, nargs=1,
        help='Usage: -batch_norm <True/False>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    train = Train(args.thresh[0], args.ep[0], args.lr[0], 
            args.w0[0], args.mini_batch[0], args.id[0], args.batch_norm[0])

    args.streams.sort()
    random.seed(42)
    if args.actions[0] == 'train':
        train.train(args.streams, 'cam1')
    elif args.actions[0] == 'cross-train':
        train.cross_train(args.streams)
    else:
        '''
        Invalid value for actions
        '''
        parser.print_help(sys.stderr)
        exit(1)

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: use model parameter to load model for training
'''

'''
    todo: nomes diferentes para classificadores
'''
