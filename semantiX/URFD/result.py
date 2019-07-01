import sys
import argparse
from sklearn.model_selection import KFold
from sklearn.externals import joblib
import numpy as np
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, \
                            classification_report
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

    def __init__(self, streams, classes, fid, cid, fold):

        self.features_key = 'features'
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'
        self.classes = classes
        self.streams = streams

        self.fid = fid
        self.cid = cid
        self.fold = fold
        self.sliding_height = 10


    def pre_result(self, stream):
        self.classifier = load_model(self.fold + '_' + stream + '_classifier_' + self.cid + '.h5')

        # Reading information extracted
        h5features = h5py.File(stream + '_features_' +  self.fid + '.h5', 'r')
        h5labels = h5py.File(stream + '_labels_' +  self.fid + '.h5', 'r')

        # all_features will contain all the feature vectors extracted from
        # optical flow images
        self.all_features = h5features[self.features_key]
        self.all_labels = np.asarray(h5labels[self.labels_key])

        predicteds = []

        for data in self.all_features:
            pred = self.classifier.predict(np.asarray(data.reshape(1, -1)))
            pred = pred.flatten()
            predicteds.append(pred)

        return self.all_features, self.all_labels, np.asarray(predicteds)

    def evaluate_max(self, truth, avg_predicted):

        predicted = np.zeros(len(truth), dtype=np.float)
        for i in range(len(truth)):
            predicted[i] = np.argmax(avg_predicted[i])

        return self.evaluate(truth, predited)

    def evaluate(self, truth, predicted):

        print("Classification report for classifier \n%s\n"
            % (classification_report(truth, predicted)))
        print("Confusion matrix:\n%s" % confusion_matrix(truth, predicted))

        # Compute metrics and print them
        cm = confusion_matrix(truth, predicted, labels=[i for i in range(len(self.classes))])

        accuracy = accuracy_score(truth, predicted)
        print('Accuracy: {}'.format(accuracy))
        print('Matthews: {}'.format(matthews_corrcoef(truth, predicted)))

        return predicted

    def result(self, f_classif):

        predicteds = []
        len_STACK = 0
        Truth = 0
        key = ''.join(self.streams)

        for stream in self.streams:
            X, Y, predicted = self.pre_result(stream)
            len_STACK = len(Y)
            Truth = Y
            predicteds.append(np.copy(predicted))

        predicteds = np.asarray(predicteds)
        cont_predicteds = np.zeros(shape=(len_STACK, len(self.classes)), dtype=np.float)

        if f_classif == 'max_avg':
            for j in range(len_STACK):
                for i in range(len(self.streams)):
                    for k in range(len(self.classes)):
                        cont_predicteds[j][k] += (predicteds[i][j][k] / len(self.streams))

            cont_predicteds = self.evaluate_max(Truth, cont_predicteds)

        elif f_classif == 'svm_avg':
            for j in range(len_STACK):
                for i in range(len(self.streams)):
                    for k in range(len(self.classes)):
                        cont_predicteds[j][k] += (predicteds[i][j][k] / len(self.streams))

            clf = joblib.load(self.fold + '_' + 'svm_avg_' + key + '.pkl')
            print('EVALUATE WITH average and svm')
            cont_predicteds = clf.predict(cont_predicteds)

            cont_predicteds = self.evaluate(Truth, cont_predicteds)

        elif f_classif == 'svm_1':

            svm_cont_1_test_predicteds = []
            for i in range(len(self.streams)):
                aux_svm = joblib.load(self.fold + '_' + 'svm_' + self.streams[i] + '_1_aux.pkl')

                svm_cont_1_test_predicteds.append(aux_svm.predict(predicteds[i]))

            svm_cont_1_test_predicteds = np.asarray(svm_cont_1_test_predicteds)
            svm_cont_1_test_predicteds = np.reshape(svm_cont_1_test_predicteds, svm_cont_1_test_predicteds.shape[::-1])

            clf = joblib.load(self.fold + '_' + 'svm_' + key + '_cont_1.pkl')
            print('EVALUATE WITH continuous values and SVM 1')
            cont_predicteds = clf.predict(svm_cont_1_test_predicteds)

            cont_predicteds = self.evaluate(Truth, cont_predicteds)

        elif f_classif == 'svm_2':
            clf = joblib.load(self.fold + '_' + 'svm_' + key + '_cont_2.pkl')

            svm_cont_2_test_predicteds = np.asarray([list(predicteds[:, i, j]) for i in range(len(Truth)) for j in range(len(self.classes))])
            svm_cont_2_test_predicteds = svm_cont_2_test_predicteds.reshape(len(Truth), len(self.classes) * len(self.streams))

            print('EVALUATE WITH continuous values and SVM 2')
            cont_predicteds = clf.predict(svm_cont_2_test_predicteds)

            cont_predicteds = self.evaluate(Truth, cont_predicteds)

        else:
            print("FUNCAO CLASSIFICADORA INVALIDA!!!!")
            return

        self.check_videos(Truth, cont_predicteds, self.streams[0])

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
    argp.add_argument("-fid", dest='fid', type=str, nargs=1,
        help='Usage: -id <identifier_to_features>',
        required=True)
    argp.add_argument("-cid", dest='cid', type=str, nargs=1,
        help='Usage: -id <identifier_to_classifier>',
        required=True)
    argp.add_argument("-f_classif", dest='f_classif', type=str, nargs=1,
        help='Usage: -f_classif <max_avg> or <svm_avg> or <svm_1> or <svm_2>',
        required=True)
    argp.add_argument("-fold", dest='fold', type=str, nargs=1,
        help='Usage: Fold index from train.py',
        required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    result = Result(args.streams, args.classes, args.fid[0], args.cid[0], args.fold[0])

    # Need to sort
    args.streams.sort()
    result.result(args.f_classif[0])

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
