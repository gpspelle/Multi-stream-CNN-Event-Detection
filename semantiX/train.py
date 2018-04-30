import argparse
import sys
from sklearn.model_selection import KFold
import numpy as np
from numpy.random import seed
seed(1)
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score
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

    def __init__(self, threshold, num_features, epochs, opt, learning_rate, 
    weight_0, mini_batch_size, extract_id, batch_norm):

        '''
            Parameters needed to train

        '''

        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'

        self.features_file = "features_" + extract_id + ".h5"
        self.labels_file = "labels_" + extract_id + ".h5"
        self.samples_file = "samples_" + extract_id + ".h5"
        self.num_file = "num_" + extract_id + ".h5"

        self.threshold = threshold
        self.num_features = num_features
        self.epochs = epochs
        self.opt = opt
        self.learning_rate = learning_rate
        self.weight_0 = weight_0
        self.mini_batch_size = mini_batch_size
        self.batch_norm = batch_norm 

        self.kf_falls = None
        self.kf_nofalls = None
        self.falls = None
        self.no_falls = None
        self.classifier = None

    def pre_cross_train(self, nsplits):

        self.pre_train()
        # Use a 'nsplits' fold cross-validation
        self.kf_falls = KFold(n_splits=nsplits)
        self.kf_nofalls = KFold(n_splits=nsplits)
        

    def pre_train(self):
        h5features = h5py.File(self.features_file, 'r')
        h5labels = h5py.File(self.labels_file, 'r')
        
        # all_features will contain all the feature vectors extracted from 
        # optical flow images
        self.all_features = h5features[self.features_key]
        self.all_labels = np.asarray(h5labels[self.labels_key])
        
        # Falls are related to 0 and not falls to 1
        self.falls = np.asarray(np.where(self.all_labels==0)[0])
        self.no_falls = np.asarray(np.where(self.all_labels==1)[0])
        self.falls.sort()
        self.no_falls.sort() 

    def cross_train(self, nsplits):

        self.pre_cross_train(nsplits)
        sensitivities = []
        specificities = []
        fars = []
        mdrs = []
        accuracies = []
       
        first = 0

        # CROSS-VALIDATION: Stratified partition of the dataset into 
        # train/test sets
        # todo : split this line
        for (train_falls, test_falls), (train_nofalls, test_nofalls) in zip(self.kf_falls.split(self.all_features[self.falls, ...]), self.kf_nofalls.split(self.all_features[self.no_falls, ...])):

            train_falls = np.asarray(train_falls)
            test_falls = np.asarray(test_falls)
            train_nofalls = np.asarray(train_nofalls)
            test_nofalls = np.asarray(test_nofalls)

            # todo: change this X, _y, X2 and _y2 variables name
            X = np.concatenate((self.all_features[train_falls, ...], 
                self.all_features[train_nofalls, ...]))
            _y = np.concatenate((self.all_labels[train_falls, ...],
                self.all_labels[train_nofalls, ...]))
            X2 = np.concatenate((self.all_features[test_falls, ...],
                self.all_features[test_nofalls, ...]))
            _y2 = np.concatenate((self.all_labels[test_falls, ...], 
                self.all_labels[test_nofalls, ...]))   
            
            # Balance the number of positive and negative samples so that there
            # is the same amount of each of them
            all0 = np.asarray(np.where(_y==0)[0])
            all1 = np.asarray(np.where(_y==1)[0])  
            if len(all0) < len(all1):
                all1 = np.random.choice(all1, len(all0), replace=False)
            else:
                all0 = np.random.choice(all0, len(all1), replace=False)
            allin = np.concatenate((all0.flatten(),all1.flatten()))
            allin.sort()
            X_t = X[allin,...]
            _y_t = _y[allin]

            self.set_classifier() 

            # ==================== TRAINING ========================     
            # weighting of each class: only the fall class gets a different
            # weight
            class_weight = {0: self.weight_0, 1: 1}
            # Batch training
            if self.mini_batch_size == 0:
                history = self.classifier.fit(X_t,_y_t, validation_data=(X2,_y2), 
                        batch_size=X.shape[0], epochs=self.epochs, 
                        shuffle='batch', class_weight=class_weight)
            else:
                history = self.classifier.fit(X_t, _y_t, validation_data=(X2,_y2), 
                        batch_size=self.mini_batch_size, nb_epoch=self.epochs, 
                        shuffle='batch', class_weight=class_weight)

            exp = 'lr{}_batchs{}_batchnorm{}_w0_{}'.format(self.learning_rate, self.mini_batch_size, self.batch_norm, self.weight_0)
            self.plot_training_info(exp, ['accuracy', 'loss'], True, 
                               history.history)

            # Store only the first classifier
            if first == 0:
                self.classifier.save('urfd_classifier.h5')
                first = 1

            # ==================== EVALUATION ======================== 
            predicted = self.classifier.predict(np.asarray(X2))
            self.evaluate(predicted, X2, _y2, sensitivities, 
            specificities, fars, mdrs, accuracies)
            
        print('5-FOLD CROSS-VALIDATION RESULTS ===================')
        print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities), 
                                                    np.std(sensitivities)))
        print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities),
                                                    np.std(specificities)))
        print("FAR: %.2f%% (+/- %.2f%%)" % (np.mean(fars), np.std(fars)))
        print("MDR: %.2f%% (+/- %.2f%%)" % (np.mean(mdrs), np.std(mdrs)))
        print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies), 
                                                 np.std(accuracies)))

    def train(self):
        sensitivities = []
        specificities = []
        fars = []
        mdrs = []
        accuracies = []

        self.pre_train()

        # todo: change this X, _y
        X = np.concatenate((self.all_features[self.falls, ...], 
            self.all_features[self.no_falls, ...]))
        _y = np.concatenate((self.all_labels[self.falls, ...],
            self.all_labels[self.no_falls, ...]))
        
        # Balance the number of positive and negative samples so that there
        # is the same amount of each of them
        # todo: check if it's really necessary
        all0 = np.asarray(np.where(_y==0)[0])
        all1 = np.asarray(np.where(_y==1)[0])  
        if len(all0) < len(all1):
            all1 = np.random.choice(all1, len(all0), replace=False)
        else:
            all0 = np.random.choice(all0, len(all1), replace=False)
        allin = np.concatenate((all0.flatten(),all1.flatten()))
        allin.sort()
        X_t = X[allin,...]
        _y_t = _y[allin]

        self.set_classifier()

        # ==================== TRAINING ========================     
        # weighting of each class: only the fall class gets a different weight
        class_weight = {0: self.weight_0, 1: 1}
        # Batch training
        if self.mini_batch_size == 0:
            history = self.classifier.fit(X_t, _y_t, validation_split=0.20, 
                    batch_size=X.shape[0], epochs=self.epochs, shuffle='batch',
                    class_weight=class_weight)
        else:
            history = self.classifier.fit(X_t, _y_t, validation_split=0.15, 
                    batch_size=self.mini_batch_size, nb_epoch=self.epochs, 
                    shuffle='batch', class_weight=class_weight)

        exp = 'lr{}_batchs{}_batchnorm{}_w0_{}'.format(self.learning_rate, self.mini_batch_size, self.batch_norm, self.weight_0)
        self.plot_training_info(exp, ['accuracy', 'loss'], True, 
                           history.history)

        self.classifier.save('urfd_classifier.h5')

        # ==================== EVALUATION ========================        
        predicted = self.classifier.predict(np.asarray(X))
        self.evaluate(predicted, X, _y, sensitivities, 
        specificities, fars, mdrs, accuracies)

    def evaluate(self, predicted, X2, _y2, sensitivities, 
    specificities, fars, mdrs, accuracies):
        for i in range(len(predicted)):
            if predicted[i] < self.threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        # Array of predictions 0/1
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
        sensitivities.append(tp/float(tp+fn))
        specificities.append(tn/float(tn+fp))
        fars.append(fpr)
        mdrs.append(fnr)
        accuracies.append(accuracy)

    def set_classifier(self):
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
        
        if self.opt == 'adam':
            adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, 
                        epsilon=1e-08, decay=0.0005)

        self.classifier = Model(input=extracted_features, output=x, 
                           name='classifier')
        self.classifier.compile(optimizer=adam, loss='binary_crossentropy',
                           metrics=['accuracy'])

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
            help='Usage: -actions <train/cross-train> \
                  Example: -actions train \
                           -actions cross-train', required=True)

    '''
        todo: make this weight_0 (w0) more general for multiple classes
    '''

    '''
        todo: verify if all these parameters are really required
    '''

    argp.add_argument("-thresh", dest='thresh', type=float, nargs=1,
            help='Usage: -thresh <x> (0<=x<=1)', required=True)
    argp.add_argument("-num_feat", dest='num_feat', type=int, nargs=1,
            help='Usage: -num_feat <size_of_features_array>', required=True)
    argp.add_argument("-ep", dest='ep', type=int, nargs=1,
            help='Usage: -ep <num_of_epochs>', required=True)
    argp.add_argument("-optim", dest='opt', type=str, nargs=1,
            help='Usage: -optim <optimizer_used> \
                  Example: -optim adam', required=True)
    argp.add_argument("-lr", dest='lr', type=float, nargs=1,
            help='Usage: -lr <learning_rate_value>', required=True)
    argp.add_argument("-w0", dest='w0', type=float, nargs=1,
            help='Usage: -w0 <weight_for_fall_class>', required=True)
    argp.add_argument("-mini_batch", dest='mini_batch', type=int, nargs=1,
            help='Usage: -mini_batch <mini_batch_size>', required=True)
    argp.add_argument("-id", dest='extract_id', type=str, nargs=1,
        help='Usage: -id <identifier_to_this_features>', required=True)
    argp.add_argument("-batch_norm", dest='batch_norm', type=bool, nargs=1,
        help='Usage: -batch_norm <True/False>', required=True)
    argp.add_argument("-cnn_arch", dest='cnn_arch', type=str, nargs=1,
            help='Usage: -cnn_arch <path_to_your_stored_architecture>', 
            required=True)
    argp.add_argument("-nsplits", dest='nsplits', type=int, nargs=1, 
    help='Usage: -nsplits <K: many splits you want (>1)>', required=False)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    train = Train(args.thresh[0], args.num_feat[0], args.ep[0], args.opt[0], 
            args.lr[0], args.w0[0], args.mini_batch[0], args.extract_id[0], 
            args.batch_norm[0])

    if args.actions[0] == 'train':
        train.train()
    elif args.actions[0] == 'cross-train':
        if args.nsplits == None:
            print("***********************************************************", 
                file=sys.stderr)
            print("You're performing a cross-traing but not giving -nsplits value")
            print("***********************************************************", 
                file=sys.stderr)
            
        else:
            train.cross_train(args.nsplits[0])
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
