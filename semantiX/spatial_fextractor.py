import keras
from resnet152 import Scale
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope 
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

    def __init__(self, class0, class1, num_features, x_size, y_size):
        self.class0 = class0
        self.class1 = class1
        self.num_features = num_features
        self.folders = []
        self.fall_dirs = []
        self.not_fall_dirs = []
        self.fall_videos = []
        self.not_fall_videos = []
        self.classes = []
        self.frames = []
        self.x_size = x_size
        self.y_size = y_size
        self.nb_total_frames = 0

    def extract(self, extract_id, model, data_folder):

        self.get_dirs(data_folder)

        extractor_model = load_model(model, custom_objects={'Scale': Scale})
        
        features_file = "spatial_features_" + extract_id + ".h5"
        labels_file = "spatial_labels_" + extract_id + ".h5"
        samples_file = "spatial_samples_" + extract_id + ".h5"
        num_file = "spatial_num_" + extract_id + ".h5"

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
        * samples_key: name of the key for the hdf5 file to store the labels
        * samples_key: name of the key for the hdf5 file to store the samples
        * num_key: name of the key for the hdf5 file to store the num
        * data_folder: folder with class0 and class1 folders
        '''

        dirs = []

        for fall_dir in self.fall_dirs:
            self.frames = glob.glob(data_folder + self.class0 + '/' + 
                                 fall_dir + '/frame_*.jpg')
            if int(len(self.frames)) >= 10:
                self.folders.append(data_folder + self.class0 + '/' + fall_dir)
                dirs.append(fall_dir)
                self.classes.append(self.class0)

        for not_fall_dir in self.not_fall_dirs:
            self.frames = glob.glob(data_folder + self.class1 + '/' +
                                 not_fall_dir + '/frame_*.jpg')
            if int(len(self.frames)) >= 10:
                self.folders.append(data_folder + self.class1 + '/' + 
                        not_fall_dir)
                dirs.append(not_fall_dir)
                self.classes.append(self.class1)

        for folder in self.folders:
            self.frames = glob.glob(folder + '/frame_*.jpg')
            self.nb_total_frames += len(self.frames)

        # File to store the extracted features and datasets to store them
        # IMPORTANT NOTE: 'w' mode totally erases previous data
        h5features = h5py.File(features_file,'w')
        h5labels = h5py.File(labels_file,'w')
        h5samples = h5py.File(samples_file, 'w')
        h5num_classes = h5py.File(num_file, 'w')

        dataset_features = h5features.create_dataset(features_key, 
                shape=(self.nb_total_frames, self.num_features), dtype='float64')
        dataset_labels = h5labels.create_dataset(labels_key, 
                shape=(self.nb_total_frames, 1), dtype='float64')  
        dataset_samples = h5samples.create_dataset(samples_key, 
                shape=(len(self.classes), 1), 
                dtype='int32')  
        dataset_num = h5num_classes.create_dataset(num_key, shape=(2, 1), 
                dtype='int32')  
        
        dataset_num[0] = len(self.fall_dirs)
        dataset_num[1] = len(self.not_fall_dirs)

        cont = 0
        number = 0
        
        for folder, dir, classe in zip(self.folders, dirs, self.classes):
            self.frames = glob.glob(folder + '/frame_*.jpg')
            self.frames.sort()
            label = glob.glob(data_folder + classe + '/' + dir + '/' + '*.npy')
            label_values = np.load(label[0])
            
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
                    #predictions[i, ...] = extractor_model.predict(frame)
                    truth[i] = label_values[i+fraction*amount_frames]

                dataset_features[cont:cont+amount_frames,:] = predictions
                dataset_labels[cont:cont+amount_frames,:] = truth
                cont += amount_frames

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
                #predictions[i, ...] = extractor_model.predict(frame)
                # todo: this 100 value is related to initial amount_frames
                truth[i] = label_values[fraction_frames * 100 + i]

            dataset_features[cont:cont+amount_frames,:] = predictions
            dataset_labels[cont:cont+amount_frames,:] = truth
            dataset_samples[number] = nb_frames
            number+=1
            cont += amount_frames

        h5features.close()
        h5labels.close()
        h5samples.close()
        h5num_classes.close()

    def get_dirs(self, data_folder):

        # Fill the folders and classes arrays with all the paths to the data
        self.fall_dirs = [f for f in os.listdir(data_folder + self.class0) 
                        if os.path.isdir(os.path.join(data_folder, 
                        self.class0, f))]

        self.not_fall_dirs = [f for f in os.listdir(data_folder + self.class1) 
                         if os.path.isdir(os.path.join(data_folder, 
                         self.class1, f))]

        self.fall_dirs.sort()
        self.not_fall_dirs.sort()

        for f in self.fall_dirs:
            self.fall_videos.append(data_folder + self.class0 + '/' + f +
                                '/' + f + '.mp4')

        for f in self.not_fall_dirs:
            self.not_fall_videos.append(data_folder + self.class1 + '/' +
                                f + '/' + f + '.mp4')

        self.fall_videos.sort()
        self.not_fall_videos.sort()

        
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
    argp.add_argument("-num_feat", dest='num_features', type=int, nargs=1,
            help='Usage: -num_feat <size_of_features_array>', required=True)
    argp.add_argument("-input_dim", dest='input_dim', type=int, nargs=2, 
            help='Usage: -input_dim <x_dimension> <y_dimension>', required=True)
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

    fextractor = Fextractor(args.classes[0], args.classes[1], 
                args.num_features[0], args.input_dim[0], args.input_dim[1])
    fextractor.extract(args.id[0], args.cnn_arch[0], args.data_folder[0])

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: impressao dupla de help se -h ou --help eh passado
'''
