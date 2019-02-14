#import keras
#from resnet152 import Scale
#from keras.models import load_model
#from keras.utils.generic_utils import CustomObjectScope 
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

''' Documentation: class Visual_Rythm_extractor
    
    This class has a few methods:

    extract
    extract_visual_rythm

    The only method that should be called outside of this class is:

    extract: simply calls to extract_visual_rythm for it's multiple classes

    extract_visual_rythm: extracts vertical and horizontal visual rythm from the video.
'''


class Visual_Rythm_extractor:

    def __init__(self, classes, mean, extension):
        self.mean = mean 
        self.extension = extension
        self.classes = classes 
        self.classes_dirs = []
        self.classes_videos = []
        self.fall_dirs = []
        self.class_value = []


    def extract(self, data_folder, window):

        self.get_dirs(data_folder)
    
        for i in range(len(self.classes)):
            # Extracting visual rythm
            self.extract_visual_rythm(data_folder, self.classes_videos[i], 
                    self.classes_dirs[i], self.classes[i], window)

    def extract_visual_rythm(self, data_folder, videos, dirs, class_, window):

        for (video, dir) in zip(videos, dirs): 
            print (video) 
            #print(dir)
            path = data_folder + class_ +  '/' + dir 
            cap = cv2.VideoCapture(video)
            #sucess, frame1 = cap.read( )
            #print (sucess)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print( length )
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
            #print (width)
            #print (height)
            hor_vr = np.array([]).reshape(0,int(width),3)
            ver_vr = np.array([]).reshape(0,int(height),3)
            #window = 10 
            count = 0
            for fi in range (0,(length+1)-window):
                #print (fi)
                foward = True
                self.vr(video, width, height, hor_vr, ver_vr, fi, window, path, foward, count)
                count = count + 1
            print ("---------------------------------###---------------------------------------")
            for fi in range (length-1,(length)-window, -1):
                #print (fi)
                foward =  False
                self.vr (video, width, height, hor_vr, ver_vr, fi, window, path, foward, count)
                count = count + 1
                

    def vr (self, video, width, height, hor_vr, ver_vr, fi, window, path, foward, count):
        cap = cv2.VideoCapture(video)
        
        if foward == True:
            ff = fi + window
            for fr in range(fi,ff):
                print (fr)
                #print (fr+10)
                cap.set(1,fr);
                ret, img = cap.read()
                #print (ret)
                if ret == True:
                    #print ('TRUE')
                    #frames+=1
                    if(self.mean):
                        #print ('if')
                        hor = np.mean(img, axis=0)
                        ver = np.mean(img, axis=1)
                    else:
                        #print ('else')
                        hor = img[int(height/2),:]
                        ver = img[:,int(width/2)]
                    hor_vr = np.vstack([hor_vr,[hor]])
                    ver_vr = np.vstack([ver_vr,[ver]])
                    #print (hor_vr)
                else:
                    #print ('breaked')
                    break
        else:
            ff = (fi + 1) - window
            #print (ff)
            for fr in range(fi,ff-1,-1):
                print (fr)
                #print (fr-10)
                cap.set(1,fr);
                ret, img = cap.read()
                #print (ret)
                if ret == True:
                    #print ('TRUE')
                    #frames+=1
                    if(self.mean):
                        #print ('if')
                        hor = np.mean(img, axis=0)
                        ver = np.mean(img, axis=1)
                    else:
                        #print ('else')
                        hor = img[int(height/2),:]
                        ver = img[:,int(width/2)]
                    hor_vr = np.vstack([hor_vr,[hor]])
                    ver_vr = np.vstack([ver_vr,[ver]])
                    #print (hor_vr)
                else:
                    #print ('breaked')
                    break

        hor_vr = hor_vr.astype(np.uint8)
        ver_vr = ver_vr.astype(np.uint8)
        print (hor_vr.shape)
        print (ver_vr.shape)
        print (':::::path:::::::')
        ph = path + '/ritmo_0000'+str(count)+'_h.jpg'
        pv = path + '/ritmo_0000'+str(count)+'_v.jpg'
        print (ph)
        print (pv)
        cv2.imwrite(ph, hor_vr)
        cv2.imwrite(pv, ver_vr)
        try:
            self.rescale_VR(ph, pv, path)
        except cv2.error as e:
            print ("Check dimensions")
        cap.release()
        cv2.destroyAllWindows()

    def rescale_VR(self, ph, pv, path): 

        im1 = cv2.imread(ph)
        im2 = cv2.imread(pv)

        img1 = cv2.resize(im1,(224,224))
        img2 = cv2.resize(im2,(224,224))
        #for i in range (1,11,2):
        #    ph1 = path + '/ritmo_0000'+ str(i) + '.jpg'
        #    pv1 = path + '/ritmo_0000'+ str(i + 1) + '.jpg'
        #    cv2.imwrite(ph1, img1)
        #    cv2.imwrite(pv1, img2)
        cv2.imwrite(ph, img1)
        cv2.imwrite(pv, img2)

            
        

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

    argp.add_argument("-mean", dest='input_mean', type=int, nargs=1, 
            help='Usage: -mean <vr_mean> ', required=False)

    argp.add_argument("-extension", dest='extension', type=str, nargs='+',
            help='Usage: -extension <video_extension_type>',
            required=True)

    argp.add_argument("-window", dest='window', type=int, nargs=1,
            help='Usage: -window <sliding_window_size>',
            required=True)
    
    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    vr_extractor = Visual_Rythm_extractor(args.classes, args.input_mean[0], args.extension[0])
    vr_extractor.extract(args.data_folder[0], args.window[0])
    print ("done")



