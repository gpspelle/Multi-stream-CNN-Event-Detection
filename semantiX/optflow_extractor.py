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

''' Documentation: class Optflow_extractor
    
    This class has a few methods:

    extract
    extract_optflow

    The only method that should be called outside of this class is:

    extract: simply calls to extract_optflow for it's multiple classes

    extract_optflow: extracts opticalflows from videos using Farnebacks's
    algorithm and stores in the same folder that the video is.
'''


class Optflow_extractor:

    def __init__(self, class0, class1, x_size, y_size):
        self.class0 = class0
        self.class1 = class1
        self.fall_dirs = []
        self.not_fall_dirs = []
        self.fall_videos = []
        self.not_fall_videos = []
        self.classes = []
        self.x_size = x_size
        self.y_size = y_size
        # Total amount of stacks with sliding window=num_images-sliding_height+1

    def extract(self, extract_id, model, data_folder):

        self.get_dirs(data_folder)

        # Extracting Fall optical flow
        self.extract_optflow(data_folder, self.fall_videos, self.fall_dirs,
                             self.class0)

        # Extracting NotFall optical flow
        self.extract_optflow(data_folder, self.not_fall_videos,
                             self.not_fall_dirs, self.class1)

    def extract_optflow(self, data_folder, videos, dirs, class_):

        for (video, dir) in zip(videos, dirs): 
            counter = 1
            cap = cv2.VideoCapture(video)
            success, frame1 = cap.read()
            try:
                prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                print("Inside every folder in dataset it's expected a valid" +
                "(non-empty) video with name equal to the folder + .mp4." +
                "In your case, inside %s it's expected a %s video" 
                % (data_folder + class_ + dir, video)
                , file=sys.stderr)
                exit(1)
            hsv = np.zeros_like(frame1)
            hsv[...,1] = 255
            path = data_folder + class_ +  '/' + dir
            while True:
                success, frame2 = cap.read()
                if success == False:
                    break
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 
                        0.702, 5, 10, 2, 7, 1.5, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                # todo: (ALERT) because of a unknown reason cartToPolar is 
                # returning -inf for some mag positions and than normalize
                # gets all 0...
                for i in range(len(mag)):
                    for j in range(len(mag[i])):
                        if math.isnan(mag[i][j]) or math.isinf(mag[i][j]):
                            mag[i][j] = 0
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                '''
                    todo: this isn't fine and only will work for urfd data set
                '''
                if self.x_size != 224 or self.y_size != 224:
                    print("-input_dim 224 224 are obrigatory so far. sorry.",
                           file=sys.stderr)
                    exit(1)

                cv2.imwrite(path + '/' + 'flow_x_' + str(counter).zfill(5) + 
                        '.jpg', hsv[..., 0])
                cv2.imwrite(path + '/' + 'flow_y_' + str(counter).zfill(5) +
                        '.jpg', hsv[..., 2])
                cv2.imwrite(path + '/' + 'flow_z_' + str(counter).zfill(5) + 
                        '.jpg', bgr)
                counter += 1
                prvs = next
            cap.release()
            cv2.destroyAllWindows()

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

    optflow_extractor = Optflow_extractor(args.classes[0], args.classes[1], 
                args.input_dim[0], args.input_dim[1])
    optflow_extractor.extract(args.id[0], args.cnn_arch[0], args.data_folder[0])

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: impressao dupla de help se -h ou --help eh passado
'''