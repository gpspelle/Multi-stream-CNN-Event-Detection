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

    generator
    extract

    The only method that should be called outside of this class is:

    extract: receives a CNN already trained until the last two full connected
    layers and extract features from optical flows extracted from a video.
    A feature is the result from a feedforward using a stack of optical flows,
    later these features will be used for training these last two layers.
'''


class Fextractor:

    def __init__(self, classes, num_features, x_size, y_size, id):

        self.num_features = num_features
        self.folders = []
        
        self.classes = classes
        self.classes_dirs = []
        self.classes_videos = []

        self.class_value = []
        self.x_images = []
        self.y_images = []
        self.x_size = x_size
        self.y_size = y_size
        self.id = id
        # Total amount of stacks with sliding window=num_images-sliding_height+1
        self.nb_total_stacks = 0

    def generator(self, list1, list2):
        '''
        Auxiliar generator: returns the ith element of both given list with 
        each call to next() 
        '''
        for x,y in zip(list1,list2):
            yield x, y

    def extract(self, model, data_folder):

        self.get_dirs(data_folder)

        extractor_model = load_model(model, custom_objects={'Scale': Scale})
        
        features_file = 'temporal_features_' + self.id  + '.h5'
        labels_file = 'temporal_labels_' + self.id  + '.h5'
        samples_file = 'temporal_samples_' + self.id  + '.h5'
        num_file = 'temporal_num_' + self.id  + '.h5'

        features_key = 'features' 
        labels_key = 'labels'
        samples_key = 'samples'
        num_key = 'num'
        sliding_height = 10

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
        * sliding_height: height of stack to process
        '''

        try:
            flow_mean = sio.loadmat('flow_mean.mat')['image_mean']
        except:
            print("***********************************************************",
                file=sys.stderr)
            print("A flow_mean.mat file with mean values for your trained CNN",
                    file=sys.stderr)
            print("should be in the same directory as fextractor.py. This",
                    file=sys.stderr)
            print("file also needs a image_mean key", file=sys.stderr)
            print("***********************************************************",
                file=sys.stderr)
            exit(1)

        dirs = []

        for c in range(len(self.classes)):
            for dir in self.classes_dirs[c]: 
                self.frames = glob.glob(data_folder + self.classes[c] + '/' + 
                              dir + '/flow_x*.jpg')

                if int(len(self.frames)) >= sliding_height:
                    self.folders.append(data_folder + self.classes[c] + '/' + dir)
                    dirs.append(dir)
                    self.class_value.append(self.classes[c])
                    self.nb_total_stacks += len(self.frames)


        # File to store the extracted features and datasets to store them
        # IMPORTANT NOTE: 'w' mode totally erases previous data
        h5features = h5py.File(features_file,'w')
        h5labels = h5py.File(labels_file,'w')
        h5samples = h5py.File(samples_file, 'w')
        h5num_classes = h5py.File(num_file, 'w')

        dataset_features = h5features.create_dataset(features_key, 
                shape=(self.nb_total_stacks, self.num_features), dtype='float64')
        dataset_labels = h5labels.create_dataset(labels_key, 
                shape=(self.nb_total_stacks, 1), dtype='float64')  
        dataset_samples = h5samples.create_dataset(samples_key, 
                shape=(len(self.class_value), 1), 
                dtype='int32')  
        dataset_num = h5num_classes.create_dataset(num_key, shape=(len(self.classes), 1), 
                dtype='int32')  
        
        for c in range(len(self.classes)):
            dataset_num[c] = len(self.classes_dirs[c])

        cont = 0
        number = 0
        
        for folder, dir, classe in zip(self.folders, dirs, self.class_value):
            self.x_images = glob.glob(folder + '/flow_x*.jpg')
            self.x_images.sort()
            self.y_images = glob.glob(folder + '/flow_y*.jpg')
            self.y_images.sort()
            label = glob.glob(data_folder + classe + '/' + dir + '/' + '*.npy')
            label_values = np.load(label[0])

            nb_stacks = len(self.x_images)-sliding_height+1
            # Here nb_stacks optical flow stacks will be stored

            amount_stacks = 100 
            fraction_stacks = nb_stacks // amount_stacks
            gen = self.generator(self.x_images, self.y_images)
            for fraction in range(fraction_stacks):
                flow = np.zeros(shape=(self.x_size, self.y_size, 2*sliding_height, 
                                amount_stacks), dtype=np.float64)
                for i in range(amount_stacks):
                    flow_x_file, flow_y_file = next(gen)
                    img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                    img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                    # Assign an image i to the jth stack in the kth position,
                    # but also in the j+1th stack in the k+1th position and so 
                    # on (for sliding window) 
                    for s in list(reversed(range(min(sliding_height,i+1)))):
                        if i-s < amount_stacks:
                            flow[:,:,2*s,  i-s] = img_x
                            flow[:,:,2*s+1,i-s] = img_y
                    del img_x,img_y
                    gc.collect()
                    
                # Subtract mean
                flow = flow - np.tile(flow_mean[...,np.newaxis], 
                        (1, 1, 1, flow.shape[3]))
                # Transpose for channel ordering (Tensorflow in this case)
                flow = np.transpose(flow, (3, 0, 1, 2)) 
                predictions = np.zeros((amount_stacks, self.num_features), 
                        dtype=np.float64)
                truth = np.zeros((amount_stacks, 1), dtype='int8')
                # Process each stack: do the feed-forward pass and store in the 
                # hdf5 file the output
                for i in range(amount_stacks):
                    prediction = extractor_model.predict(
                                                np.expand_dims(flow[i, ...], 0))
                    predictions[i, ...] = prediction
                    truth[i] = self.get_media_optflow(label_values, i+(fraction*amount_stacks), sliding_height)

                dataset_features[cont:cont+amount_stacks,:] = predictions
                dataset_labels[cont:cont+amount_stacks,:] = truth
                cont += amount_stacks

            amount_stacks = nb_stacks % amount_stacks
            flow = np.zeros(shape=(self.x_size, self.y_size, 2*sliding_height, 
                            amount_stacks), dtype=np.float64)

            for i in range(amount_stacks + sliding_height - 1):
                flow_x_file, flow_y_file = next(gen)
                img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                # Assign an image i to the jth stack in the kth position,
                # but also in the j+1th stack in the k+1th position and so on 
                # (for sliding window) 
                for s in list(reversed(range(min(sliding_height,i+1)))):
                    if i-s < amount_stacks:
                        flow[:,:,2*s,  i-s] = img_x
                        flow[:,:,2*s+1,i-s] = img_y
                del img_x,img_y
                gc.collect()
                
            # Subtract mean
            flow = flow - np.tile(flow_mean[...,np.newaxis], 
                    (1, 1, 1, flow.shape[3]))
            # Transpose for channel ordering (Tensorflow in this case)
            flow = np.transpose(flow, (3, 0, 1, 2)) 
            predictions = np.zeros((amount_stacks, self.num_features), 
                    dtype=np.float64)
            truth = np.zeros((amount_stacks, 1), dtype='int8')
            # Process each stack: do the feed-forward pass and store in the 
            # hdf5 file the output
            for i in range(amount_stacks):
                prediction = extractor_model.predict(np.expand_dims(flow[i, ...],
                                                                             0))
                predictions[i, ...] = prediction
                # todo: this 100 value is related to initial amount_stacks
                truth[i] = self.get_media_optflow(label_values, fraction_stacks* 100 + i, sliding_height)

            dataset_features[cont:cont+amount_stacks,:] = predictions
            dataset_labels[cont:cont+amount_stacks,:] = truth
            dataset_samples[number] = nb_stacks
            number+=1
            cont += amount_stacks

        h5features.close()
        h5labels.close()
        h5samples.close()
        h5num_classes.close()

    def get_media_optflow(self, label_values, i, sliding_height):
        soma = 0
        for j in range(i, i + sliding_height):
            soma += label_values[i]

        if soma / sliding_height >= 0.5:
            return 1
        else:
            return 0

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

    fextractor = Fextractor(args.classes, args.num_features[0], 
                args.input_dim[0], args.input_dim[1], args.id[0])
    fextractor.extract(args.cnn_arch[0], args.data_folder[0])

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: impressao dupla de help se -h ou --help eh passado
'''
