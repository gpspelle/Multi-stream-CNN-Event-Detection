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


''' Documentation: class Frame_extractor
    
    This class has a few methods:

    extract
    extract_frame

    extract: calls to extract_frame for it's multiple classes

    extract_frame: extracts all video frames and saves them individually.
'''

class Frame_extractor:

    def __init__(self, classes, extension):
        self.classes = classes 
        self.classes_dirs = []
        self.classes_videos = []
        self.fall_dirs = []
        self.class_value = []
        self.extension = extension

    def get_dirs(self, data_folder):

        for c in self.classes:
            self.classes_dirs.append([f for f in os.listdir(data_folder + c) 
                        if os.path.isdir(os.path.join(data_folder, c, f))])
            self.classes_dirs[-1].sort()

            self.classes_videos.append([])
            for f in self.classes_dirs[-1]:
                self.classes_videos[-1].append(data_folder + c+ '/' + f +
                                   '/' + f + '.' + self.extension)

            self.classes_videos[-1].sort()


    def extract(self, data_folder):

        self.get_dirs(data_folder)
    
        for i in range(len(self.classes)):
            # Extracting video frames
            self.extract_frame(data_folder, self.classes_videos[i], self.classes_dirs[i], self.classes[i])

    def extract_frame(self, data_folder, videos, dirs, class_):

        for (video, dir) in zip(videos, dirs): 
            print (":::Video:::")
            print (video) 

            path = data_folder + class_ +  '/' + dir 
            cap = cv2.VideoCapture(video)

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print( length )
            #width1 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            #height1 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
            #print (width1)
            #print (height1)
            count = 0
            for fi in range (0,length):
                cap.set(1,fi);
                ret, img = cap.read()

                save_path = path + '/depth_' + str(count).zfill(5) + '.jpg'
                print (save_path)
                cv2.imwrite(save_path, img)

                count = count + 1

            cap.release()
            #cv2.destroyAllWindows()
                             

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

    argp.add_argument("-extension", dest='extension', type=str, nargs='+',
            help='Usage: -extension <video_extension_type>',
            required=True)

   
    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    fr_extractor = Frame_extractor(args.classes, args.extension[0])
    fr_extractor.extract(args.data_folder[0])
    print ("done")



