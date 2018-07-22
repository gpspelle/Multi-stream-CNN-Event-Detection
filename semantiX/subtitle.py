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

    def __init__(self, data, class0, class1, threshold, fid, cid):
       
        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'
        self.fid = fid
        self.cid = cid
        self.sliding_height = 10

        self.fall_videos = []
        self.not_fall_videos = []
        self.data = data
        self.class0 = class0
        self.class1 = class1
        self.threshold = threshold

    def create_subtitle(self, streams):

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

        predicteds = []
        for stream in streams:
            X, Y, predicted = self.pre_result(stream)
    
            if stream == 'spatial':
                Truth = Y
                h5samples = h5py.File(stream + '_samples_' + self.fid + '.h5', 'r')
                all_samples = np.asarray(h5samples[self.samples_key])
                pos = 0
                index = []
                for x in all_samples:
                    index += list(range(pos+x[0]-self.sliding_height, pos+x[0]))
                    pos+=x[0]

                Truth = np.delete(Truth, index)
                predicted = np.delete(predicted, index) 

            predicteds.append(predicted)

        for j in range(len(predicteds[0])):
            for i in range(1, len(streams)):
                predicteds[0][j] += predicteds[i][j] 
            predicteds[0][j] /= len(streams)

        for i in range(len(predicted)):
            if predicteds[0][i] < self.threshold:
                predicteds[0][i] = 0
            else:
                predicteds[0][i] = 1

        # Array of predictions 0/1
        predicted = np.asarray(predicteds[0]).astype(int)

        if 'temporal' not in streams:
            print("Por enquanto temporal tem de ser um dos streams", 
                  file=sys.stderr)
            exit(1)

        h5samples = h5py.File('temporal_samples_' + self.fid + '.h5', 'r')
        h5num = h5py.File('temporal_num_' + self.fid + '.h5', 'r')

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

            subtitle_cnt = 0

            file_write = open(self.data + cl + '/' + save_dir[cnt] + '/' + 
                        save_dir[cnt] + '.srt' , 'w')  
            cnt += 1
            time = 0.0
            for i in range(inic, inic + all_samples[x][0]):
                
                subtitle_cnt += 1
                if i >= len(predicted):
                   break 

                time += (1 / fps + 0.001)
                file_write.write("\n")
                file_write.write("%d\n" % subtitle_cnt)
                time_init = str(datetime.timedelta(seconds=time))
                time_end = str(datetime.timedelta(seconds=time + 1/fps + 0.001))
                time_init = time_init[:-3]
                time_end = time_end[:-3]
                file_write.write(time_init.replace(".", ",", 1) + ' --> ' + time_end.replace(".", ",", 1) + '\n')
                file_write.write("Output: %d\n" % int(predicted[i]))
                file_write.write("Truth: %d\n" % Y[i])
                
            inic += all_samples[x][0]

    def pre_result(self, stream):
        self.classifier = load_model(stream + '_classifier_' + self.cid + '.h5')

        # Reading information extracted
        h5features = h5py.File(stream + '_features_' +  self.fid + '.h5', 'r')
        h5labels = h5py.File(stream + '_labels_' +  self.fid + '.h5', 'r')

        # all_features will contain all the feature vectors extracted from
        # optical flow images
        self.all_features = h5features[self.features_key]
        self.all_labels = np.asarray(h5labels[self.labels_key])

        predicted = self.classifier.predict(np.asarray(self.all_features))

        return self.all_features, self.all_labels, predicted

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

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    subt = Subtitle(args.data[0], args.classes[0], args.classes[1], 
                    args.thresh[0], args.fid[0], args.cid[0])

    subt.create_subtitle(args.streams)

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
