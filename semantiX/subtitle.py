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

    def __init__(self, data, class0, class1, threshold, id):
       
        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'
        self.id = id

        self.features_file = "features_" + id + ".h5"
        self.labels_file = "labels_" + id + ".h5"
        self.samples_file = "samples_" + id + ".h5"
        self.num_file = "num_" + id + ".h5"
    
        self.fall_videos = []
        self.not_fall_videos = []
        self.data = data
        self.class0 = class0
        self.class1 = class1
        self.threshold = threshold

    def create_subtitle(self):

        # Fill the folders and classes arrays with all the paths to the data
        self.fall_dirs = [f for f in os.listdir(self.data + self.class0) 
                        if os.path.isdir(os.path.join(self.data, 
                        self.class0, f))]

        

        self.not_fall_dirs = [f for f in os.listdir(self.data + self.class1) 
                         if os.path.isdir(os.path.join(self.data, 
                         self.class1, f))]

        self.fall_dirs.sort()
        self.not_fall_dirs.sort()

        for f in self.fall_dirs:
            self.fall_videos.append(self.data + self.class0 + '/' + f +
                                '/' + f + '.mp4')

        for f in self.not_fall_dirs:
            self.not_fall_videos.append(self.data + self.class1 + '/' +
                                f + '/' + f + '.mp4')

        self.fall_videos.sort()
        self.not_fall_videos.sort()

        # todo: change X and Y variable names
        X, Y, predicted = self.pre_result()

        for i in range(len(predicted)):
            if predicted[i] < self.threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)
       
        h5samples = h5py.File(self.samples_file, 'r')
        h5num = h5py.File(self.num_file, 'r')

        all_samples = np.asarray(h5samples[self.samples_key])
        all_num = np.asarray(h5num[self.num_key])

        list_video = []
        cnt = 0
        sliding_height = 10
        inic = 0
        for x in range(len(all_samples)):

            if all_samples[x][0] == 0:
                continue

            if x == 0:
                list_video = self.fall_videos[:]
                save_dir = self.fall_dirs[:] 
                cl = self.class0
                # save subtitle on Falls
            elif x == all_num[0][0]:
                cnt = 0
                list_video = self.not_fall_videos[:]
                save_dir = self.not_fall_dirs[:]
                cl = self.class1
                # save subtitle on NotFalls

            cap = cv2.VideoCapture(list_video[cnt]) 
            fps = cap.get(cv2.CAP_PROP_FPS) 

            total_frames = all_samples[x][0] + sliding_height - 1

            video_duration = total_frames / fps
            subtitle_cnt = 0

            file_write = open(self.data + cl + '/' + save_dir[cnt] + '/' + 
                        save_dir[cnt] + '.srt' , 'w')  
            cnt += 1
            time = 0.0
            for i in range(inic, inic + all_samples[x][0]):
                
                subtitle_cnt += 1
                if i >= len(predicted):
                   break 

                #m0, s0 = divmod(time, 60.0)
                #h0, m0 = divmod(m0, 60.0)
                #m1, s1 = divmod(time + 1 / fps, 60.0)
                #h1, m1 = divmod(m1, 60.0)
                
                #print(time)
                time += (1 / fps + 0.001)
                file_write.write("\n")
                file_write.write("%d\n" % subtitle_cnt)
                time_init = str(datetime.timedelta(seconds=time))
                time_end = str(datetime.timedelta(seconds=time + 1/fps + 0.001))
                time_init = time_init[:-3]
                time_end = time_end[:-3]
                file_write.write(time_init.replace(".", ",", 1) + ' --> ' + time_end.replace(".", ",", 1) + '\n')
                file_write.write("Output: %d\n" % predicted[i])
                file_write.write("Truth: %d\n" % Y[i])
                
            inic += all_samples[x][0]

    def pre_result(self):
        self.classifier = load_model('classifier_' + self.id + '.h5')

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
    argp.add_argument("-data", dest='data', type=str, nargs=1, 
            help='Usage: -data <path_to_your_data_folder>', required=True)
    argp.add_argument("-thresh", dest='thresh', type=float, nargs=1,
            help='Usage: -thresh <x> (0<=x<=1)', required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-id", dest='id', type=str, nargs=1,
        help='Usage: -id <identifier_to_this_features_and_classifier>', 
        required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    subt = Subtitle(args.data[0], args.classes[0], args.classes[1], args.thresh[0], 
                args.id[0])

    subt.create_subtitle()

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
