import sys
import argparse
from sklearn.model_selection import KFold
from sklearn.externals import joblib
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

    def __init__(self, classes, threshold, fid, cid):

        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'
        self.classes = classes

        self.fid = fid
        self.cid = cid
        self.sliding_height = 10

        self.threshold = threshold

    def pre_result(self, stream):
        self.classifier = load_model(stream + '_classifier_' + self.cid + '.h5')

        # Reading information extracted
        h5features = h5py.File(stream + '_features_' +  self.fid + '.h5', 'r')
        h5labels = h5py.File(stream + '_labels_' +  self.fid + '.h5', 'r')

        # all_features will contain all the feature vectors extracted from
        # optical flow images
        self.all_features = h5features[self.features_key]
        self.all_labels = np.asarray(h5labels[self.labels_key])

        predicteds = []
        
        for data in self.all_features:
            pred = classifier.predict(np.asarray(data.reshape(1, -1)))
            predicteds.append(np.argmax(pred))

        return self.all_features, self.all_labels, predicteds

    def evaluate_threshold(self, truth, predicted):

        for i in range(len(predicted)):
            if predicted[i] < self.threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1

        self.evaluate(truth, predicted)

    def evaluate(self, truth, predicted):
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)

        # Compute metrics and print them
        cm = confusion_matrix(truth, predicted, labels=[0,1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]

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

        accuracy = accuracy_score(truth, predicted)

        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))   
        print('Sensitivity/Recall: {}'.format(recall))
        print('Specificity: {}'.format(specificity))
        print('Precision: {}'.format(precision))
        print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))

    def result(self, streams, f_classif):

        predicteds = []
        len_STACK = 0
        Truth = 0
        for stream in streams:
            X, Y, predicted = self.pre_result(stream)
            len_STACK = len(Y)
            Truth = Y
            predicted = np.asarray(predicted.flat)
            predicteds.append(np.copy(predicted)) 

        cont_predicteds = np.zeros(len_STACK, dtype=np.float)
                   
        if f_classif == 'thresh':
            for j in range(len(cont_predicteds)):
                for i in range(len(streams)):
                    cont_predicteds[j] += predicteds[i][j] 

                cont_predicteds[j] /= (len(streams))

            self.check_videos(Truth, cont_predicteds, streams[0])
            
        elif f_classif == 'svm_avg':
            for j in range(len(cont_predicteds)):
                for i in range(len(streams)):
                    cont_predicteds[j] += predicteds[i][j] 

                cont_predicteds[j] /= (len(streams))

            clf = joblib.load('svm_avg.pkl')
            print('EVALUATE WITH average and svm')
            for i in range(len(cont_predicteds)):
                cont_predicteds[i] = clf.predict(cont_predicteds[i])

            self.check_videos(Truth, cont_predicteds, streams[0])

        elif f_classif == 'svm_cont':
            clf = joblib.load('svm_cont.pkl')
            print('EVALUATE WITH continuous values and svm')
            for i in range(len(cont_predicteds)):
                cont_predicteds[i] = clf.predict(np.asarray([item[i] for item in predicteds]).reshape(1, -1))

            self.check_videos(Truth, cont_predicteds, streams[0])

    def check_videos(self, _y2, predicted, stream):
        h5samples = h5py.File(stream + '_samples_' + self.fid + '.h5', 'r')
        h5num = h5py.File(stream + '_num_' + self.fid + '.h5', 'r')

        all_samples = np.asarray(h5samples[self.samples_key])
        all_num = np.asarray(h5num[self.num_key])

        stack_c = 0
        class_c = 0
        video_c = 0
        all_num = [y for x in all_num for y in x]
        for amount_videos in all_num:
            cl = self.classes[class_c]
            message = '###### ' + cl + ' videos ' + str(amount_videos)+' ######' 
            print(message)
            for num_video in range(amount_videos):
                num_miss = 0
                FP = 0
                FN = 0

                for num_stack in range(stack_c, stack_c + all_samples[video_c+num_video][0]):
                    if num_stack >= len(predicted):
                        break
                    elif predicted[num_stack] != _y2[num_stack]:
                        if _y2[num_stack] == 0:
                            FN += 1
                        else:
                            FP += 1
                        num_miss+=1
                    
                if num_miss == 0:
                    print("Hit video    %3d  [%5d miss  %5d stacks  %5d FP  %5d FN]" %(num_video+1, num_miss, all_samples[video_c+num_video][0], FP, FN))
                else:
                    print("Miss video   %3d  [%5d miss  %5d stacks  %5d FP  %5d FN]" %(num_video+1, num_miss, all_samples[video_c+num_video][0], FP, FN))

                stack_c += all_samples[video_c + num_video][0]

            video_c += amount_videos
            class_c += 1

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
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-streams", dest='streams', type=str, nargs='+',
            help='Usage: -streams spatial temporal (to use 2 streams example)',
            required=True)
    argp.add_argument("-thresh", dest='thresh', type=float, nargs=1,
            help='Usage: -thresh <x> (0<=x<=1)', required=True)
    argp.add_argument("-fid", dest='fid', type=str, nargs=1,
        help='Usage: -id <identifier_to_features>', 
        required=True)
    argp.add_argument("-cid", dest='cid', type=str, nargs=1,
        help='Usage: -id <identifier_to_classifier>', 
        required=True)
    argp.add_argument("-f_classif", dest='f_classif', type=str, nargs=1,
        help='Usage: -f_classif <thresh> or <svm_avg> or <svm_cont>', 
        required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    result = Result(args.classes, args.thresh[0], args.fid[0], args.cid[0])

    # Need to sort
    args.streams.sort()
    result.result(args.streams, args.f_classif[0])

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
