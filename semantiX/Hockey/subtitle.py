import sys
import argparse
import numpy as np
import h5py
import os
import cv2
import datetime
from keras.models import load_model
''' Documentation: class Subtitle
    
    This class has a few methods:

    pre_result
    result
    check_videos

    The methods that should be called outside of this class are:

    result: show the results of a prediction based on a feed forward on the
    classifier of this worker.

'''

class Subtitle:

    def __init__(self, data, classes, threshold, fid, cid, ext):
       
        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'
        self.fid = fid
        self.cid = cid
        self.sliding_height = 10
        self.ext = ext

        self.classes = classes
        self.classes_videos = []
        self.classes_dirs = []
        self.data = data
        self.classes = classes 
        self.threshold = threshold

    def get_dirs(self, data_folder):

        for c in self.classes:
            self.classes_dirs.append([f for f in os.listdir(data_folder + c) 
                        if os.path.isdir(os.path.join(data_folder, c, f))])
            self.classes_dirs[-1].sort()

            self.classes_videos.append([])
            for f in self.classes_dirs[-1]:
                self.classes_videos[-1].append(data_folder + c+ '/' + f +
                                   '/' + f + self.ext)

            self.classes_videos[-1].sort()

    def create_subtitle(self, streams, f_classif):

        self.get_dirs(self.data)

        predicteds = []
        len_STACK = 0
        Truth = 0
        key = ''.join(streams)

        for stream in streams:
            X, Y, predicted = self.pre_result(stream)
            len_STACK = len(Y)
            Truth = Y
            predicteds.append(np.copy(predicted)) 

        cont_predicteds = np.zeros(len_STACK, dtype=np.float)
                   
        if f_classif == 'thresh':
            for j in range(len(cont_predicteds)):
                for i in range(len(streams)):
                    for k in range(len(self.classes):
                        cont_predicteds[j] += (predicteds[i][j][k] / len(streams)) 

            self.evaluate_max(Truth, cont_predicteds)

        elif f_classif == 'svm_avg':
            for j in range(len(cont_predicteds)):
                for i in range(len(streams)):
                    for k in range(len(self.classes):
                        cont_predicteds[j] += (predicteds[i][j][k] / len(streams)) 

            clf = joblib.load('svm_avg_ ' + key + '.pkl')
            print('EVALUATE WITH average and svm')
            for i in range(len(cont_predicteds)):
                cont_predicteds[i] = clf.predict(cont_predicteds[i].reshape(-1, 1))

            self.evaluate(Truth, cont_predicteds)

        elif f_classif == 'svm_1':

            svm_cont_1_test_predicteds = []
            for i in range(len(self.streams)):
                aux_svm = joblib.load('svm_' + self.streams[i] + '_1_aux.pkl')

                svm_cont_1_test_predicteds.append(aux_svm.predict(predicteds[i]))

            svm_cont_1_test_predicteds = np.asarray(svm_cont_1_test_predicteds)
            svm_cont_1_test_predicteds = np.reshape(svm_cont_1_test_predicteds, svm_cont_1_test_predicteds.shape[::-1])

            clf = joblib.load('svm_' + key + '_cont_1.pkl')
            print('EVALUATE WITH continuous values and SVM 1')
            cont_predicteds = clf.predict(svm_cont_1_test_predicteds) 

            self.evaluate(Truth, cont_predicteds)

        elif f_classif == 'svm_2':
            clf = joblib.load('svm_ ' + key '_cont_2.pkl')

            svm_cont_2_test_predicteds = np.asarray([list(predicteds[:, i, j]) for i in range(len(Truth)) for j in range(len(self.classes))])
            svm_cont_2_test_predicteds = svm_cont_2_test_predicteds.reshape(len(Truth), len(self.classes) * len(streams))

            print('EVALUATE WITH continuous values and SVM 2')
            cont_predicteds = clf.predict(svm_cont_2_test_predicteds) 
            
            self.evaluate(Truth, cont_predicteds)

        else:
            print("FUNCAO CLASSIFICADORA INVALIDA!!!!")
            return
        
        self.generate_subtitle(Truth, cont_predicteds, streams[0])

    def generate_subtitle(self, Truth, predicted, stream)
        h5samples = h5py.File(stream + '_samples_' + self.fid + '.h5', 'r')
        h5num = h5py.File(stream + '_num_' + self.fid + '.h5', 'r')

        all_samples = np.asarray(h5samples[self.samples_key])
        all_num = np.asarray(h5num[self.num_key])

        stack_c = 0
        class_c = 0
        video_c = 0
        all_num = [y for x in all_num for y in x]
        for amount_videos in all_num:
            list_video = self.classes_videos[class_c][:]
            save_dir = self.classes_dirs[class_c][:]
            cl = self.classes[class_c]
            for num_video in range(amount_videos):
                
                cap = cv2.VideoCapture(list_video[num_video]) 
                fps = cap.get(cv2.CAP_PROP_FPS) 
                subtitle_c = 0
                time = 0.0
                file_write = open(self.data + cl + '/' + save_dir[num_video] +
                        '/' + save_dir[num_video] + '.srt' , 'w')  

                for num_stack in range(stack_c, stack_c + all_samples[video_c+num_video][0]):
                    subtitle_c += 1

                    if num_stack >= len(predicted):
                        break

                    time += (1 / fps + 0.001)
                    file_write.write("\n")
                    file_write.write("%d\n" % subtitle_c)
                    time_init = str(datetime.timedelta(seconds=time))
                    time_end = str(datetime.timedelta(seconds=time + 1/fps + 0.001))
                    time_init = time_init[:-3]
                    time_end = time_end[:-3]
                    file_write.write(time_init.replace(".", ",", 1) + ' --> ' + time_end.replace(".", ",", 1) + '\n')
                    file_write.write("Output: %d\n" % int(predicted[num_stack]))
                    file_write.write("Truth: %d\n" % Truth[num_stack])

                stack_c += all_samples[video_c + num_video][0]
                cap.release()

            video_c += amount_videos
            class_c += 1

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
            pred = self.classifier.predict(np.asarray(data.reshape(1, -1)))
            pred = pred.flatten()
            predicteds.append(pred)

        return self.all_features, self.all_labels, predicteds

if __name__ == '__main__':

    '''
        todo: verify if all these parameters are really required
    '''

    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)

    argp = argparse.ArgumentParser(description='Do subtitle tasks')
    argp.add_argument("-streams", dest='streams', type=str, nargs='+',
            help='Usage: -streams spatial temporal (to use 2 streams example)',
            required=True)
    argp.add_argument("-data", dest='data', type=str, nargs=1, 
            help='Usage: -data <path_to_your_data_folder>', required=True)
    argp.add_argument("-thresh", dest='thresh', type=float, nargs=1,
            help='Usage: -thresh <x> (0<=x<=1)', required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-fid", dest='fid', type=str, nargs=1,
        help='Usage: -id <identifier_to_features>', 
        required=True)
    argp.add_argument("-cid", dest='cid', type=str, nargs=1,
        help='Usage: -id <identifier_to_classifier>', 
        required=True)
    argp.add_argument("-ext", dest='ext', type=str, nargs=1, 
        help='Usage: -ext <file_extension> .mp4 | .avi | ...', required=True)
    argp.add_argument("-f_classif", dest='f_classif', type=str, nargs=1,
        help='Usage: -f_classif <thresh> or <svm_avg> or <svm_cont>', 
        required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    subt = Subtitle(args.data[0], args.classes, args.thresh[0], args.fid[0], 
                    args.cid[0], args.ext[0])

    # Need to sort 
    args.streams.sort()
    subt.create_subtitle(args.streams, args.f_classif[0])

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
