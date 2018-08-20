import sys
import argparse
import h5py
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.engine import InputLayer

''' This code is based on Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, 
    I. (2017). "Vision-Based Fall Detection with Convolutional Neural Networks"
    Wireless Communications and Mobile Computing, 2017.
    Also, new features were added by Gabriel Pellegrino Silva working in 
    Semantix. 
'''

''' Documentation: script multi-stream-resnet50 

    This code use Kera's applications in deep learning to create and load
    weights for the possibilities of CNN's that are implemented here.
    
    Temporal: input are a 10+10 stack of optical flows (mag, ord)
    Pose: imput are RGB drawed skeletons using tf-pose-estimation
    Spatial: input are RGB images

'''

if __name__ == '__main__':
    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)
    argp = argparse.ArgumentParser(description='Do architecture tasks')
    argp.add_argument("-streams", dest='streams', type=str, nargs='+',
            help='So far, spatial, temporal, pose and its combinations \
                  Usage: -streams spatial temporal',
            required=True)
    #argp.add_argument("-num_feat", dest='num_features', type=int, nargs=1,
    #        help='Usage: -num_feat <size_of_features_array>', required=True)
    #argp.add_argument("-input_dim", dest='input_dim', type=int, nargs=2, 
    #        help='Usage: -input_dim <x_dimension> <y_dimension>', required=True)
    argp.add_argument("-weight", dest='weight', type=str, nargs='+', 
            help='Usage: -weight <path_to_your_weight_file>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)
    
    sliding_height = 10
    x_dim = 224
    y_dim = 224

    if 'temporal' in args.streams:
        
        # Build this CNN hard-coded
        print("Saving your temporal CNN as RESNET50_temporal")
        model = ResNet50(weights=None, include_top=False)
        new_input = InputLayer(input_shape=(x_dim, y_dim, 2*sliding_height), name='input_1')
        model.layers[0] = new_input
        model.load_weights(args.weight[0], by_name=True)
        model.save('RESNET50_temporal')

    if 'pose' in args.streams:
        model_1 = ResNet50(include_top=False)
        model_1.load_weights(args.weight[0], by_name=True)
        
        print("Saving your pose-estimation CNN as RESNET50_pose")
        model_1.save('RESNET50_pose')
    
    if 'spatial' in args.streams:
        model_2 = ResNet50(include_top=False)
        model_2.load_weights(args.weight[0], by_name=True)

        print("Saving your spatial CNN as RESNET50_spatial")
        model_2.save('RESNET50_spatial')

