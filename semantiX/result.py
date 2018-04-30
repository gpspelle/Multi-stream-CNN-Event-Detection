import sys
import argparse
from sklearn.model_selection import KFold
import numpy as np
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

''' Documentation: class Result
    
    This class has a few methods:

    pre_result
    result
    check_videos

    The methods that should be called outside of this class are:

    result: show the results of a prediction based on a feed forward on the
    classifier of this worker.

'''

class Result:

    def __init__(self, class0, class1, threshold, extract_id):

        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'

        self.features_file = "features_" + extract_id + ".h5"
        self.labels_file = "labels_" + extract_id + ".h5"
        self.samples_file = "samples_" + extract_id + ".h5"
        self.num_file = "num_" + extract_id + ".h5"

        self.class0 = class0
        self.class1 = class1
        self.threshold = threshold
        self.falls = None
        self.no_falls = None
        self.classifier = None

    def pre_result(self):
        self.classifier = load_model('urfd_classifier.h5')

        # Reading information extracted
        h5features = h5py.File(self.features_file, 'r')
        h5labels = h5py.File(self.labels_file, 'r')

        # all_features will contain all the feature vectors extracted from
        # optical flow images
        self.all_features = h5features[self.features_key]
        self.all_labels = np.asarray(h5labels[self.labels_key])

        self.falls = np.asarray(np.where(self.all_labels==0)[0])
        self.no_falls = np.asarray(np.where(self.all_labels==1)[0])
   
        self.falls.sort()
        self.no_falls.sort()

        # todo: change X and Y variable names
        X = np.concatenate((self.all_features[self.falls, ...], 
            self.all_features[self.no_falls, ...]))
        Y = np.concatenate((self.all_labels[self.falls, ...], 
            self.all_labels[self.no_falls, ...]))
       
        predicted = self.classifier.predict(np.asarray(X))

        return X, Y, predicted

    def result(self):

        # todo: change X and Y variable names
        X, Y, predicted = self.pre_result()

        for i in range(len(predicted)):
            if predicted[i] < self.threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)
        # Compute metrics and print them
        cm = confusion_matrix(Y, predicted,labels=[0,1])
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
        accuracy = accuracy_score(Y, predicted)

        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))   
        print('Sensitivity/Recall: {}'.format(recall))
        print('Specificity: {}'.format(specificity))
        print('Precision: {}'.format(precision))
        print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))

        self.check_videos(Y, predicted)

    def check_videos(self, _y2, predicted):
        h5samples = h5py.File(self.samples_file, 'r')
        h5num = h5py.File(self.num_file, 'r')

        all_samples = np.asarray(h5samples[self.samples_key])
        all_num = np.asarray(h5num[self.num_key])

        video = 1
        inic = 0
        misses = 0

        msage_fall = list("###### Fall videos ")
        msage_fall.append(str(all_num[0][0]))
        msage_fall.append(" ######")
        msage_not_fall = list("###### Not fall videos ")
        msage_not_fall.append(str(all_num[1][0]))
        msage_not_fall.append(" ######")
        print(all_num)

        for x in range(len(all_samples)):
            correct = 1

            if all_samples[x][0] == 0:
                continue

            if x == 0:
                print(''.join(msage_fall))
            elif x == all_num[0][0]:
                print(''.join(msage_not_fall))
                video = 1 

            for i in range(inic, inic + all_samples[x][0]):
                if i >= len(predicted):
                   break 
                elif predicted[i] != _y2[i]:
                    misses+=1
                    correct = 0

            if correct == 1:
               print("Hit video:      " + str(video))
            else:
               print("Miss video:     " + str(video))

            video += 1
            inic += all_samples[x][0]

if __name__ == '__main__':

    '''
        todo: make this weight_0 (w0) more general for multiple classes
    '''

    '''
        todo: verify if all these parameters are really required
    '''

    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)

    argp = argparse.ArgumentParser(description='Do result  tasks')
    argp.add_argument("-thresh", dest='thresh', type=float, nargs=1,
            help='Usage: -thresh <x> (0<=x<=1)', required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-id", dest='id', type=str, nargs=1,
        help='Usage: -id <identifier_to_this_features>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    result = Result(args.classes[0], args.classes[1], args.thresh[0], 
                args.id[0])

    result.result()

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
