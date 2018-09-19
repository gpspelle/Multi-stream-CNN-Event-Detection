import keras
from keras.models import load_model
import math
import sys
import argparse
import numpy as np
import scipy.io as sio
import os
import glob
import h5py
import cv2
import gc

''' This code is based on Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, 
    I. (2017). "Vision-Based Fall Detection with Convolutional Neural Networks"
    Wireless Communications and Mobile Computing, 2017.
    Also, new features were added by Gabriel Pellegrino Silva working in 
    Semantix. 
'''

''' Documentation: class Fextractor
    
    This class has a few methods:

    extract

    The only method that should be called outside of this class is:

    extract: receives a CNN already trained until the last two full connected
    layers and extract features from optical flows extracted from a video.
    A feature is the result from a feedforward using a stack of optical flows,
    later these features will be used for training these last two layers.
'''


class Fextractor:

    def __init__(self, classes, id):

        self.classes = classes

        self.num_features = 4096
        self.folders = []

        self.classes_dirs = []
        self.classes_videos = []

        self.sliding_height = 10
        self.class_value = []

        self.frames = []
        self.x_size = 224
        self.y_size = 224
        self.id = id
        self.nb_total_frames = 0

    def extract(self, model, data_folder):

        self.get_dirs(data_folder)

        print("### Model loading", flush=True)
        extractor_model = load_model(model)
        
        features_file = 'spatial_features_' + self.id  + '.h5'
        labels_file = 'spatial_labels_' + self.id  + '.h5'
        samples_file = 'spatial_samples_' + self.id  + '.h5'
        num_file = 'spatial_num_' + self.id  + '.h5'

        features_key = 'features' 
        labels_key = 'labels'
        samples_key = 'samples'
        num_key = 'num'

        '''
        Function to load the optical flow stacks, do a feed-forward through 
        the feature extractor (VGG16) and store the output feature vectors in 
        the file 'features_file' and the labels in 'labels_file'.
        Input:
        * extractor_model: CNN model until the last two layers.
        * features_file: path to the hdf5 file where the extracted features are
        going to be stored
        * labels_file: path to the hdf5 file where the labels of the features
        are going to be stored
        * samples_file: path to the hdf5 file where the number of stacks in 
        each video is going to be stored
        * num_file: path to the hdf5 file where the number of fall and not fall
        videos are going to be stored
        * features_key: name of the key for the hdf5 file to store the features
        * labels_key: name of the key for the hdf5 file to store the labels
        * samples_key: name of the key for the hdf5 file to store the samples
        * num_key: name of the key for the hdf5 file to store the num
        * data_folder: folder with class0 and class1 folders
        '''

        dirs = []

        # File to store the extracted features and datasets to store them
        # IMPORTANT NOTE: 'w' mode totally erases previous data
        print("### Creating h5 files", flush=True)
        h5features = h5py.File(features_file,'w')
        h5labels = h5py.File(labels_file,'w')
        h5samples = h5py.File(samples_file, 'w')
        h5num_classes = h5py.File(num_file, 'w')
        cams = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7', 'cam8']

        stacks_in_cam = dict()
        for cam in cams:
            stacks_in_cam[cam] = 0

        for c in self.classes:
            h5features.create_group(c)
            h5labels.create_group(c)
            for cam in cams:
                h5features[c].create_group(cam)
                h5labels[c].create_group(cam)

        for c in range(len(self.classes)):

            if self.classes[c] != 'Falls' and self.classes[c] != 'NotFalls':
                print("Sorry. Classes possibles are Falls and NotFalls, it's \
                    hardcoded and will be expanded really soon. It's being \
                    used inside Extracting Features for, setting label value")
                exit(1)

            for dir in self.classes_dirs[c]: 
                self.frames = glob.glob(data_folder + self.classes[c] + '/' + 
                              dir + '/frame_*.jpg')

                if int(len(self.frames)) >= self.sliding_height:

                    # search with cam is being used in this dir
                    # dir is something like: chute01cam2 or chute01cam2_00
                    for cam in cams:
                        if cam in dir:
                            stacks_in_cam[cam] = stacks_in_cam[cam] + len(self.frames)
                     
                    self.folders.append(data_folder + self.classes[c] + '/' + dir)
                    dirs.append(dir)
                    self.class_value.append(self.classes[c])
                    self.nb_total_frames += len(self.frames)

        datasets_f = dict()
        datasets_l = dict()
        for c in self.classes:
            datasets_f[c] = dict()
            datasets_l[c] = dict()
            for cam in cams:
                datasets_f[c][cam] = h5features[c][cam].create_dataset(cam, shape=(stacks_in_cam[cam], self.num_features), dtype='float64')
                datasets_l[c][cam] = h5labels[c][cam].create_dataset(cam, shape=(stacks_in_cam[cam], 1), dtype='float64')

        dataset_samples = h5samples.create_dataset(samples_key, 
                shape=(len(self.class_value), 1), 
                dtype='int32')  
        dataset_num = h5num_classes.create_dataset(num_key, shape=(len(self.classes), 1), 
                dtype='int32')  
        
        for c in range(len(self.classes)):
            dataset_num[c] = len(self.classes_dirs[c])

        cont = dict()
        for cam in cams:
            cont[cam] = 0
        number = 0

        for c in range(len(self.classes)):
            dataset_num[c] = len(self.classes_dirs[c])

        print("### Extracting Features", flush=True)
        for folder, dir, classe in zip(self.folders, dirs, self.class_value):
            for cam in cams:
                if cam in dir:
                    self.update_progress(cont[cam]/stacks_in_cam[cam])

            self.frames = glob.glob(folder + '/frame_*.jpg')
            self.frames.sort()

            label = -1
            if classe == 'Falls':
                label = 0
            else:
                label = 1

            #label = glob.glob(data_folder + classe + '/' + dir + '/' + '*.npy')
            #label_values = np.load(label[0])
            
            nb_frames = len(self.frames)

            amount_frames = 100 
            fraction_frames = nb_frames // amount_frames
            iterr = iter(self.frames)
            for fraction in range(fraction_frames):
                predictions = np.zeros((amount_frames, self.num_features), 
                        dtype=np.float64)
                truth = np.zeros((amount_frames, 1), dtype='int8')
                # Process each stack: do the feed-forward pass and store in the 
                # hdf5 file the output
                for i in range(amount_frames):
                    frame = next(iterr)
                    frame = cv2.imread(frame)
                    predictions[i, ...] = extractor_model.predict(np.expand_dims(frame, 0))
                    #truth[i] = label_values[i+fraction*amount_frames]
                    truth[i] = label

                for cam in cams:
                    if cam in dir:
                        datasets_f[classe][cam][cont[cam]:cont[cam]+amount_frames,:] = predictions
                        datasets_l[classe][cam][cont[cam]:cont[cam]+amount_frames,:] = truth
                        cont[cam] += amount_frames


            amount_frames = nb_frames % amount_frames

            predictions = np.zeros((amount_frames, self.num_features), 
                    dtype=np.float64)
            truth = np.zeros((amount_frames, 1), dtype='int8')
            # Process each stack: do the feed-forward pass and store in the 
            # hdf5 file the output
            for i in range(amount_frames):
                frame = next(iterr)
                frame = cv2.imread(frame)
                predictions[i, ...] = extractor_model.predict(np.expand_dims(frame, 0))
                # todo: this 100 value is related to initial amount_frames
                #truth[i] = label_values[fraction_frames * 100 + i]
                truth[i] = label

            for cam in cams:
                if cam in dir:
                    datasets_f[classe][cam][cont[cam]:cont[cam]+amount_frames,:] = predictions
                    datasets_l[classe][cam][cont[cam]:cont[cam]+amount_frames,:] = truth
                    cont[cam] += amount_frames


            dataset_samples[number] = nb_frames
            number+=1

        h5features.close()
        h5labels.close()
        h5samples.close()
        h5num_classes.close()

    def update_progress(self, workdone):
            print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)

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

        
if __name__ == '__main__':
    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)
    argp = argparse.ArgumentParser(description='Do feature extraction tasks')
    argp.add_argument("-data", dest='data_folder', type=str, nargs=1, 
            help='Usage: -data <path_to_your_data_folder>', required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-cnn_arch", dest='cnn_arch', type=str, nargs=1,
            help='Usage: -cnn_arch <path_to_your_stored_architecture>', 
            required=True)
    argp.add_argument("-id", dest='id', type=str, nargs=1,
            help='Usage: -id <identifier_to_this_features>', required=True)
    
    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    fextractor = Fextractor(args.classes, args.id[0])
    fextractor.extract(args.cnn_arch[0], args.data_folder[0])

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: impressao dupla de help se -h ou --help eh passado
'''
