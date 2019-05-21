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

    def __init__(self, data, classes, threshold, fid, cid):
       
        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'
        self.fid = fid
        self.cid = cid
        self.sliding_height = 10

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
                                   '/' + f + '.mp4')

            self.classes_videos[-1].sort()

    def create_subtitle(self, streams):

        self.get_dirs(self.data)

        predicteds = []
        for stream in streams:
            X, Y, predicted = self.pre_result(stream)
   
            predicted = np.asarray(predicted.flat)
            if stream == 'spatial' or stream == 'pose':
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
                predicteds[0][j] +=  1 * predicteds[i][j] 
            predicteds[0][j] /= (1 + 1 * len(range(1, len(streams))))

        for i in range(len(predicteds[0])):
            if predicteds[0][i] < self.threshold:
                predicteds[0][i] = 0
            else:
                predicteds[0][i] = 1

        # Array of predictions 0/1
        predicted = np.asarray(predicteds[0]).astype(int)

        h5samples = h5py.File('temporal_samples_' + self.fid + '.h5', 'r')
        h5num = h5py.File('temporal_num_' + self.fid + '.h5', 'r')

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

    subt = Subtitle(args.data[0], args.classes, args.thresh[0], args.fid[0], 
                    args.cid[0])

    args.streams.sort()
    subt.create_subtitle(args.streams)

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
