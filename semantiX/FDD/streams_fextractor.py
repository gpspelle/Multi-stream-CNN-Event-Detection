import keras
from keras.models import load_model
from keras import backend as K
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

    As the code present in multi-stream-vgg16.py, this code has a lot of auto-
    defined variables: num_features, x_size, y_size, sliding_height.

    Also, the files need to be organized in a proper way. Lets first define
    a few variables.

    -data is passed as argument and contains the path to our data. 

    In my case, I'm using a folder /mnt/Data/URFD/, so I pass:

    -data /mnt/Data/URFD

    Next, we have -class variable

    Inside /mnt/Data/URFD we have two folders: Falls and NotFalls. 

    /mnt/Data/URFD/Falls
    /mnt/Data/URFD/NotFalls

    so, 

    -class Falls NotFalls

    needs to be passed as argument. 

    Inside this folders /mnt/Data/URFD/Falls we have many other folders, 
    each of these folders has a video, the name of this folder, suppose
    video01 contains a video named, for example, video01.mp4.
    Illustrating, /mnt/Data/URFD/Falls/video01 has a file
    /mnt/Data/URFD/Falls/video01/video01.mp4. This video, that not necessarily
    needs to be in the .mp4 extension, is used by the Data_Extraction 
    pre-process stage to acquire informations like: optical flow, pose
    estimation, among others. 

    Now, if our video has n frames, inside this /mnt/Data/URFD/Falls/video01/
    we shall have:

    n frames of all streams
    n-1 frame for optical_flow (actually, only one OF algorithm is being used
    and it uses 2 frames to calculate one OF, so the last frame doesn't match)

    Now, we have two kind of streams, as already said in multi-stream-vgg16.py.
    STACK stream stacks sliding_height frames to evaluate.
    RGB streams evaluate every frame.

    It causes a decompensation.

    This decompensation is solved while extracting features and then everything
    shall be fine in this aspect. Every sliding_height stack has a start frame,
    right? We will compare the frame that start this stack to their matches
    from the other streams. Creating a one-to-one correlation within the stream.

    The output feature .h5 file is an array of arrays with num_features elements    each, so, (n, num_features). label .h5 file is an array of arrays with 1
    elements, so, (n, 1). This 1 element is the label of the class, so far,
    it's only possible 10 different classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    
'''


class Fextractor:

    def __init__(self, classes, id, ext):

        self.ext = ext
        self.folders = []
        
        self.classes = classes
        self.classes.sort()
        self.classes_dirs = []
        self.classes_videos = []

        self.class_value = []
        self.data_images = []
        self.data_images_1 = []
        
        # Some constants defined over a lot of classes.
        self.num_features = 4096    # Number of output of our CNN 
        self.x_size = 224           # X image size
        self.y_size = 224           # Y image size               224x224 pixels 
        
        self.id = id
        # Total amount of data with sliding window=num_images-sliding_height+1
        self.nb_total_data = 0

    def extract(self, stream, model, data_folder):

        print("### Model loading", flush=True)
        extractor_model = load_model(model)
        
        # This code produces 4 files for each stream.  

        # features_file contain arrays of size self.num_features. Each array is
        # the output of VGG16 to a data information and is composed of values       
        # in the range [0, 1].

        # Just to remember, data information is what this stream consider as an
        # input. RGB streams use a frame, and STACK streams use a stack of frames. 

        features_file = stream + '_features_' + self.id  + '.h5'
        labels_file = stream + '_labels_' + self.id  + '.h5'
        samples_file = stream + '_samples_' + self.id  + '.h5'
        num_file = stream + '_num_' + self.id  + '.h5'

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
        num_class = []

        # File to store the extracted features and datasets to store them
        # IMPORTANT NOTE: 'w' mode totally erases previous data
        print("### Creating h5 files", flush=True)
        h5features = h5py.File(features_file,'w')
        h5labels = h5py.File(labels_file,'w')
        h5samples = h5py.File(samples_file, 'w')
        h5num_classes = h5py.File(num_file, 'w')

        '''
            Each stream need to have its file with correct names
        '''
        if stream == 'temporal':
            file_name = '/flow_x*.jpg'
            file_name_1 = '/flow_y*.jpg'
        elif stream == 'pose':
            file_name = '/pose_*.jpg'
        elif stream == 'spatial':
            file_name = '/frame_*.jpg'
        elif stream == 'ritmo':
            file_name = '/ritmo_*.jpg'
        elif stream == 'depth':
            file_name = '/depth_*.jpg'
        elif stream == 'saliency':
            file_name = '/saliency_*.png'
        else:
            print("INVALID STREAM ERROR")
            print("VALIDS STREAMS: {temporal, spatial, pose, ritmo, depth, saliency}") 
            exit(1)

        for c in range(len(self.classes)):

            num_class.append(0)
            #if self.classes[c] != 'Falls' and self.classes[c] != 'NotFalls':
            #    print("Sorry. Classes possibles are Falls and NotFalls, its \
            #        hardcoded and will be expanded really soon. Its being \
            #        used inside Extracting Features for, setting label value")
            #    exit(1)

            for dir in self.classes_dirs[c]: 
               
                self.data = glob.glob(data_folder + self.classes[c] + '/' + 
                                  dir + file_name)
                    
                # if file_name == 'flow_x_*.jpg'
                # then 
                #   we have len(self.data) equals to the amount of optical flows
                #   which is the amount of len(self.data) - 1 for every other
                #   file_name

                # if it is a temporal stream or any other STACK-like stream
                if stream == 'temporal':
                    if len(self.data) >= sliding_height:
                        # search which cam is being used in this dir
                        # dir is something like: chute01cam2 or chute01cam2_00
                        num_class[-1] += 1
                        self.folders.append(data_folder + self.classes[c] + '/' + dir)
                        dirs.append(dir)
                        self.class_value.append(self.classes[c])

                        # Removing last datas from all streams to match the
                        # amount of data present on temporal sream
                        self.nb_total_data += len(self.data) - sliding_height + 1
                else:
                    if len(self.data) - 1 >= sliding_height:
                        # search which cam is being used in this dir
                        # dir is something like: chute01cam2 or chute01cam2_00
                        num_class[-1] += 1
                        self.folders.append(data_folder + self.classes[c] + '/' + dir)
                        dirs.append(dir)
                        self.class_value.append(self.classes[c])

                        # Removing last datas from all streams to match the
                        # amount of data present on temporal sream
                        self.nb_total_data += len(self.data) - sliding_height

        dataset_features = h5features.create_dataset(features_key, 
                shape=(self.nb_total_data, self.num_features), dtype='float64')
        dataset_labels = h5labels.create_dataset(labels_key, 
                shape=(self.nb_total_data, 1), dtype='float64')  
        dataset_samples = h5samples.create_dataset(samples_key, 
                shape=(len(self.class_value), 1), dtype='int32')  
        dataset_num = h5num_classes.create_dataset(num_key, 
                shape=(len(self.classes), 1), dtype='int32')  
        
        for c in range(len(self.classes)):
            dataset_num[c] = num_class[c]

        number = 0
        cont = 0
        
        print("### Extracting Features", flush=True)
        for folder, dir, classe in zip(self.folders, dirs, self.class_value):
            self.update_progress(cont/self.nb_total_data)
        
            self.data_images = glob.glob(folder + file_name)
            self.data_images.sort()

            if stream == 'temporal':
                self.data_images_1 = glob.glob(folder + file_name_1)
                self.data_images_1.sort()
            else:
                # Removing unmatched frames from other streams
                self.data_images = self.data_images[:-sliding_height]

            label = self.classes.index(classe)

            # last -sliding_height + 1 OF frames dont get a stack
            if stream == 'temporal':
                nb_datas = len(self.data_images) - sliding_height + 1
            else:
                # for other streams, data_images already is matched with 
                # temporal
                nb_datas = len(self.data_images)

            amount_datas = 100 
            fraction_datas = nb_datas // amount_datas

            iterr = iter(self.data_images)
            image_c = 0
            for fraction in range(fraction_datas):

                if stream == 'temporal':
                    flow = np.zeros(shape=(self.x_size, self.y_size, 2*sliding_height, 
                                    amount_datas), dtype=np.float64)

                    for i in range(amount_datas + sliding_height -1):
                        flow_x_file = self.data_images[image_c]
                        flow_y_file = self.data_images_1[image_c]

                        image_c += 1

                        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)

                        # Assign an image i to the jth stack in the kth position,
                        # but also in the j+1th stack in the k+1th position and so 
                        # on (for sliding window) 
                        for s in list(reversed(range(min(sliding_height,i+1)))):
                            if i-s < amount_datas:
                                flow[:,:,2*s,  i-s] = img_x
                                flow[:,:,2*s+1,i-s] = img_y
                        del img_x,img_y
                        gc.collect()

                    # Restore last images from previous fraction to start next 
                    # fraction    
                    image_c = image_c - sliding_height + 1
                        
                    # Subtract mean
                    flow = flow - np.tile(flow_mean[...,np.newaxis], 
                            (1, 1, 1, flow.shape[3]))
                    # Transpose for channel ordering (Tensorflow in this case)
                    flow = np.transpose(flow, (3, 0, 1, 2)) 
                    predictions = np.zeros((amount_datas, self.num_features), 
                            dtype=np.float64)
                    truth = np.zeros((amount_datas, 1), dtype='int8')
                    # Process each stack: do the feed-forward pass and store in the 
                    # hdf5 file the output
                    for i in range(amount_datas):
                        prediction = extractor_model.predict(
                                                    np.expand_dims(flow[i, ...], 0))
                        predictions[i, ...] = prediction
                        #truth[i] = self.get_media_optflow(label_values, i+(fraction*amount_datas), sliding_height)
                        truth[i] = label
                else:
                    predictions = np.zeros((amount_datas, self.num_features), 
                            dtype=np.float64)
                    truth = np.zeros((amount_datas, 1), dtype='int8')
                    # Process each stack: do the feed-forward pass and store in the 
                    # hdf5 file the output
                    for i in range(amount_datas):
                        frame = next(iterr)
                        frame = cv2.imread(frame)
                        predictions[i, ...] = extractor_model.predict(np.expand_dims(frame, 0))
                        #truth[i] = label_values[i+fraction*amount_frames]
                        truth[i] = label

                
                dataset_features[cont:cont+amount_datas,:] = predictions
                dataset_labels[cont:cont+amount_datas,:] = truth
                cont += amount_datas

            amount_datas = nb_datas % amount_datas
            predictions = np.zeros((amount_datas, self.num_features), 
                    dtype=np.float64)
            truth = np.zeros((amount_datas, 1), dtype='int8')

            if stream == 'temporal':
                flow = np.zeros(shape=(self.x_size, self.y_size, 2*sliding_height, 
                                amount_datas), dtype=np.float64)

                for i in range(amount_datas + sliding_height - 1):
                    flow_x_file = self.data_images[image_c]
                    flow_y_file = self.data_images_1[image_c]

                    image_c += 1

                    img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                    img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                    # Assign an image i to the jth stack in the kth position,
                    # but also in the j+1th stack in the k+1th position and so on 
                    # (for sliding window) 
                    for s in list(reversed(range(min(sliding_height,i+1)))):
                        if i-s < amount_datas:
                            flow[:,:,2*s,  i-s] = img_x
                            flow[:,:,2*s+1,i-s] = img_y
                    del img_x,img_y
                    gc.collect()
                    
                # Subtract mean
                flow = flow - np.tile(flow_mean[...,np.newaxis], 
                        (1, 1, 1, flow.shape[3]))
                # Transpose for channel ordering (Tensorflow in this case)
                flow = np.transpose(flow, (3, 0, 1, 2)) 
                # Process each stack: do the feed-forward pass and store in the 
                # hdf5 file the output
                for i in range(amount_datas):
                    prediction = extractor_model.predict(np.expand_dims(flow[i, ...],
                                                                                 0))
                    predictions[i, ...] = prediction
                    # this 100 value is related to initial amount_datas
                    truth[i] = label
            else:
                # Process each stack: do the feed-forward pass and store in the 
                # hdf5 file the output
                for i in range(amount_datas):
                    frame = next(iterr)
                    frame = cv2.imread(frame)
                    predictions[i, ...] = extractor_model.predict(np.expand_dims(frame, 0))
                    # this 100 value is related to initial amount_frames
                    truth[i] = label

            dataset_features[cont:cont+amount_datas,:] = predictions
            dataset_labels[cont:cont+amount_datas,:] = truth
            cont += amount_datas
            dataset_samples[number] = nb_datas
            number+=1

        h5features.close()
        h5labels.close()
        h5samples.close()
        h5num_classes.close()

    def update_progress(self, workdone):
            print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)

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
                                   '/' + f + self.ext)

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
    argp.add_argument("-streams", dest='streams', type=str, nargs='+',
            help='So far, spatial, temporal, pose and its combinations \
                  Usage: -streams spatial temporal',
            required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-id", dest='id', type=str, nargs=1,
            help='Usage: -id <identifier_to_this_features>', required=True)
    argp.add_argument("-ext", dest='ext', type=str, nargs=1, 
            help='Usage: -ext <file_extension> .mp4 | .avi | ...', required=True)
    
    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)


    for stream in args.streams:
        print("STREAM: " + stream)
        fextractor = Fextractor(args.classes, args.id[0], args.ext[0])
        fextractor.get_dirs(args.data_folder[0])
        fextractor.extract(stream, 'VGG16_' + stream, args.data_folder[0])
        K.clear_session()

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: impressao dupla de help se -h ou --help eh passado
'''
