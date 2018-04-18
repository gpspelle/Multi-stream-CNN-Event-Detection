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

    def __init__(self, class0, class1, num_features, x_size, y_size):
        self.class0 = class0
        self.class1 = class1
        self.num_features = num_features
        self.folders = []
        self.classes = []
        self.x_size = x_size
        self.y_size = y_size
        # Total amount of stacks with sliding window=num_images-sliding_height+1
        self.nb_total_stacks = 0

    def generator(self, list1, list2):
        '''
        Auxiliar generator: returns the ith element of both given list with 
        each call to next() 
        '''
        for x,y in zip(list1,list2):
            yield x, y

    def extract(self, extract_id, extractor_model, data_folder):

        '''
            todo: import the extractor_model
        '''
        
        features_file = "features_" + extract_id
        labels_file = "labels_" + extract_id
        samples_file = "samples_" + extract_id
        num_file = "num_" + extract_id

        features_key = 'features' 
        labels_key = 'labels'
        samples_key = 'samples'
        num_key = 'num'
        sliding_heigth = 10

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
        * mean_file: mean value for CNN file
        '''

        flow_mean = sio.loadmat(mean_file)['image_mean']

        # Fill the folders and classes arrays with all the paths to the data
        fall_videos = [f for f in os.listdir(data_folder + self.class0) 
                       if os.path.isdir(os.path.join(data_folder + 
                           self.class0, f))]
        fall_videos.sort()
        for fall_video in fall_videos:
            x_images = glob.glob(data_folder + self.class0 + '/' + 
                                 fall_video + '/flow_x*.jpg')
            if int(len(x_images)) >= 10:
                self.folders.append(data_folder + self.class0 + '/' + fall_video)
                self.classes.append(0)

        not_fall_videos = [f for f in os.listdir(data_folder + self.class1) 
                    if os.path.isdir(os.path.join(data_folder + 
                        self.class1, f))]
        not_fall_videos.sort()
        for not_fall_video in not_fall_videos:
            x_images = glob.glob(data_folder + self.class1 + '/' +
                                 not_fall_video + '/flow_x*.jpg')
            if int(len(x_images)) >= 10:
                self.folders.append(data_folder + self.class1 + '/' + 
                        not_fall_video)
                self.classes.append(1)

        for folder in self.folders:
            x_images = glob.glob(folder + '/flow_x*.jpg')
            self.nb_total_stacks += len(x_images)-sliding_height+1
        
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
                shape=(len(fall_videos) + len(not_fall_videos), 1), 
                dtype='int32')  
        dataset_num = h5num_classes.create_dataset(num_key, shape=(2, 1), 
                dtype='int32')  
        
        dataset_num[0] = len(fall_videos)
        dataset_num[1] = len(not_fall_videos)

        cont = 0
        number = 0
        
        for folder, label in zip(self.folders, self.classes):
            x_images = glob.glob(folder + '/flow_x*.jpg')
            x_images.sort()
            y_images = glob.glob(folder + '/flow_y*.jpg')
            y_images.sort()
            nb_stacks = len(x_images)-sliding_height+1
            # Here nb_stacks optical flow stacks will be stored
            flow = np.zeros(shape=(self.x_size, self.y_size, 2*sliding_height, 
                            nb_stacks), dtype=np.float64)
            gen = self.generator(x_images,y_images)
            for i in range(len(x_images)):
                flow_x_file, flow_y_file = next(gen)
                img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                # Assign an image i to the jth stack in the kth position,
                # but also in the j+1th stack in the k+1th position and so on 
                # (for sliding window) 
                for s in list(reversed(range(min(sliding_height,i+1)))):
                    if i-s < nb_stacks:
                        flow[:,:,2*s,  i-s] = img_x
                        flow[:,:,2*s+1,i-s] = img_y
                del img_x,img_y
                gc.collect()
                
            # Subtract mean
            flow = flow - np.tile(flow_mean[...,np.newaxis], 
                    (1, 1, 1, flow.shape[3]))
            # Transpose for channel ordering (Tensorflow in this case)
            flow = np.transpose(flow, (3, 2, 0, 1)) 
            predictions = np.zeros((nb_stacks, self.num_features), 
                    dtype=np.float64)
            truth = np.zeros((nb_stacks, 1), dtype='int8')
            # Process each stack: do the feed-forward pass and store in the 
            # hdf5 file the output
            for i in range(nb_stacks):
                prediction = extractor_model.predict(np.expand_dims(
                                                                flow[i, ...],0))
                predictions[i, ...] = prediction
                truth[i] = label

            dataset_features[cont:cont+nb_stacks,:] = predictions
            dataset_labels[cont:cont+nb_stacks,:] = truth
            dataset_samples[number] = nb_stacks
            number+=1
            cont += nb_stacks
        h5features.close()
        h5labels.close()
        h5samples.close()
        h5num_classes.close()

if __name__ == '__main__':
    argp = argparse.ArgumentParser(description='Do feature extraction tasks')
    argp.add_argument("-data", dest='data_folder', type=str, nargs='?', 
            help='Usage: -data <path_to_your_data_folder>', required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-num_feat", dest='num_features', type=int, nargs='?',
            help='Usage: -num_feat <size_of_features_array>', required=True)
    argp.add_argument("-input_dim", dest='input_dim', type=int, nargs='+', 
            help='Usage: -input_dim <x_dimension> <y_dimension>', required=True)
    argp.add_argument("-model", dest='model', type=str, nargs='?',
            help='Usage: -model_name <path_to_your_stored_model>', 
            required=True)
    argp.add_argument("-id", dest='extract_id', type=str, nargs='?',
            help='Usage: -id <identifier_to_this_features>', required=True)
    args = argp.parse_args()

    fextractor = Fextractor(args.classes[0], args.classes[1], args.num_features,
                            args.input_dim[0], args.input_dim[1])
    fextractor.extract(args.id, args.model, args.data_folder)
