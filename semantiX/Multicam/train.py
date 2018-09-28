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

        f_tpr = 0
        f_fpr = 0
        f_fnr = 0
        f_tnr = 0
        f_precision = 0
        f_recall = 0
        f_specificity = 0
        f_f1 = 0
        f_accuracy = 0

        # Big TODO's: 
        # 1. it isn't exactly k-fold because the data istn't partitioned once.
        # it's divided in a random way at each self.train call
        # 2. it isn't being chosen a model, but first, a criteria must be find
    
        cams = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7', 'cam8']
        for cam in cams: 
            tpr, fpr, fnr, tnr, precision, recall, specificity, f1, accuracy = self.train(streams, cams, cam)
            K.clear_session()
            f_tpr += tpr
            f_fpr += fpr
            f_fnr += fnr
            f_tnr += tnr
            f_precision += precision
            f_recall += recall
            f_specificity += specificity
            f_f1 += f1
            f_accuracy += accuracy

        nsplits = 8
        f_tpr /= nsplits
        f_fpr /= nsplits
        f_fnr /= nsplits
        f_tnr /= nsplits
        f_precision /= nsplits
        f_recall /= nsplits
        f_specificity /= nsplits
        f_f1 /= nsplits
        f_accuracy /= nsplits

        print("***********************************************************")
        print("             SEMANTIX - UNICAMP DATALAB 2018")
        print("***********************************************************")
        print("CROSS VALIDATION MEAN RESULTS: %d splits" % (nsplits))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(f_tpr,f_tnr,f_fpr,f_fnr))   
        print('Sensitivity/Recall: {}'.format(f_recall))
        print('Specificity: {}'.format(f_specificity))
        print('Precision: {}'.format(f_precision))
        print('F1-measure: {}'.format(f_f1))
        print('Accuracy: {}'.format(f_accuracy))


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

    def train(self, streams, cams, camera):
  
        VGG16 = True
        predicteds = []
        train_predicteds = []
        temporal = 'temporal' in streams
        len_RGB = 0
        train_len_RGB = 0
        len_STACK = 0
        train_len_STACK = 0

        # this local seed guarantee that all the streams will use the same videos
        local_seed = random.randint(0, 10000)
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

            if stream == 'spatial' or stream == 'pose':
                len_RGB = len(y_test)
                train_len_RGB = len(y_train)

                print('EVALUATE WITH %s' % (stream))

                # ==================== EVALUATION ======================== 
                self.evaluate_threshold(predicted, y_test)

                if not temporal:
                    truth = y_test
                    train_truth = y_train
                    predicteds.append(predicted)
                    train_predicteds.append(train_predicted)
                else:    
                    truth = y_test
                    train_truth = y_train
                    pos = 0
                    train_pos = 0
                    index = []
                    train_index = []
                    for c in self.classes:
                        for cam in cams:
                            for bla in range(len(h5samples[c][cam][cam])):
                                num_samples = h5samples[c][cam][cam][bla][0]
                                if cam == camera:
                                    index += list(range(pos + num_samples - self.sliding_height, pos + num_samples))
                                    pos += num_samples
                                else:
                                    train_index += list(range(train_pos + num_samples - self.sliding_height, train_pos + num_samples))
                                    train_pos += num_samples

                    truth = np.delete(truth, index)
                    train_truth = np.delete(train_truth, train_index)
                    clean_predicted = np.delete(predicted, index)
                    train_clean_predicted = np.delete(train_predicted, train_index)
                    predicteds.append(clean_predicted)
                    train_predicteds.append(train_clean_predicted)

            elif stream == 'temporal':
                # Checking if temporal is the only stream
                if len(streams) == 1:
                    truth = y_test
                    train_truth = y_train

                len_STACK = len(y_test)
                train_len_STACK = len(y_train)
                print('EVALUATE WITH %s' % (stream))

                predicteds.append(np.copy(predicted)) 
                train_predicteds.append(np.copy(train_predicted)) 
                # ==================== EVALUATION ======================== 
                self.evaluate_threshold(predicted, y_test)
                
        if temporal:
            avg_predicted = np.zeros(len_STACK, dtype=np.float)
            train_avg_predicted = np.zeros(train_len_STACK, dtype=np.float)
            clf_train_predicteds = np.zeros((train_len_STACK, len(streams)))

            for j in range(len_STACK):
                for i in range(len(streams)):
                    avg_predicted[j] += predicteds[i][j] 

                avg_predicted[j] /= (len(streams))

            for j in range(train_len_STACK):
                for i in range(len(streams)):
                    train_avg_predicted[j] += train_predicteds[i][j] 

                train_avg_predicted[j] /= (len(streams))
             
            for j in range(train_len_STACK):
                clf_train_predicteds[j] = [item[j] for item in train_predicteds]
        else:
            avg_predicted = np.zeros(len_RGB, dtype=np.float)
            train_avg_predicted = np.zeros(train_len_RGB, dtype=np.float)
            clf_train_predicteds = np.zeros( (train_len_RGB, len(streams)) )
            for j in range(len_RGB):
                for i in range(len(streams)):
                    avg_predicted[j] += predicteds[i][j] 

                avg_predicted[j] /= (len(streams))

            for j in range(train_len_RGB):
                for i in range(len(streams)):
                    train_avg_predicted[j] += train_predicteds[i][j] 

                train_avg_predicted[j] /= (len(streams))

            for j in range(train_len_RGB):
                clf_train_predicteds[j] = [item[j] for item in train_predicteds]
        
        del X_test, X_train, y_test, y_train
        gc.collect()
        print('EVALUATE WITH average and threshold')
        self.evaluate_threshold(np.array(avg_predicted, copy=True), truth)

        clf_avg = svm.SVC()                                                                 
        clf_avg.fit(train_avg_predicted.reshape(-1, 1), train_truth)
        for i in range(len(avg_predicted)):
            avg_predicted[i] = clf_avg.predict(avg_predicted[i])

        joblib.dump(clf_avg, 'svm_avg.pkl') 

        del clf_avg
        gc.collect()

        print('EVALUATE WITH average and SVM')
        self.evaluate(avg_predicted, truth)

        clf_continuous = svm.SVC()

        clf_continuous.fit(clf_train_predicteds, train_truth)
        avg_continuous = np.array(avg_predicted, copy=True)
        for i in range(len(avg_continuous)):
            avg_continuous[i] = clf_continuous.predict(np.asarray([item[i] for item in predicteds]).reshape(1, -1))

        joblib.dump(clf_continuous, 'svm_cont.pkl') 
        print('EVALUATE WITH continuous values and SVM')
        del clf_continuous
        gc.collect()
        return self.evaluate(avg_continuous, truth)

    def evaluate_threshold(self, predicted, _y2):

       for i in range(len(predicted)):
           if predicted[i] < self.threshold:
               predicted[i] = 0
           else:
               predicted[i] = 1
       #  Array of predictions 0/1

       self.evaluate(predicted, _y2)

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
        precision = tp/float(tp+fp)
        recall = tp/float(tp+fn)
        specificity = tn/float(tn+fp)
        f1 = 2*float(precision*recall)/float(precision+recall)
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
