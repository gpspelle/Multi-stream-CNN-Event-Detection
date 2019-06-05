import sys
import argparse
import h5py
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras import backend as K
from keras.applications.vgg16 import VGG16

''' This code is based on Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, 
    I. (2017). "Vision-Based Fall Detection with Convolutional Neural Networks"
    Wireless Communications and Mobile Computing, 2017.
    Also, new features were added by Gabriel Pellegrino Silva working in 
    Semantix. 
'''

''' Documentation: script multi-stream-vgg16 


    FOR USING: in order to use it the recommended procedure is to call the
    desired .py without parameters, for example:

    $ multi-stream-vgg16.py

    Check the necessary parameters and theirs usage informations. Fill the
    parameters as you need them.

    This code use Kera's applications in deep learning to create and load
    weights for the possibilities of CNN's that are implemented here.
    
    Temporal: input are a 10+10 stack of optical flows (mag, ord)
    Pose: imput are RGB drawed skeletons using tf-pose-estimation
    Spatial: input are RGB images
    Pose: input are pose skeletons images
    Depth: input are depth images
    Ritmo: input is a transformation of a video into an image
    Saliency: input is the result of the weights of an inception-v3 prediction 
    over a frame  

    Some important values are harcoded: number of outputs of VGG16 
    (num_features), the size of the sliding window for stack streams
    (sliding window), the x and y dimension of the input video.

    There are two kinds of VGG16 being created with this code, those which use
    a stack approach and others using RGB approach. 

    Stack approach uses weights_vgg16_STACK.h5 input, that is an pre-trained
    VGG16 in imagenet + UCF101. UCF101 is a human action dataset.
    
    RGB approach uses weights_vgg16_RGB.h5 input, that is an pre-trained
    VGG16 in imagenet.

    FUTURE DEVELOPMENT: in order to add a new stream, there are a few examples
    of cases.

    First of all you need to identify the shape of your new input stream.
    If your stream has one data sample for every frame of your video, then it's
    a RGB stream. But, if your stream has one data sample for a collection of
    sliding_height frames, it is, you need a stack of size sliding_height to
    have an input information, then you have a new STACK stream.

    An example of each: spatial stream use raw RGB data from the video. Whereas, 
    temporal stream stacks the content of 10 frames to create an input
    information. 
    
    If you have a nice abstraction of this ocurring in a finite
    input video, you can see that the last (sliding_heights - 1) frames can't 
    make a stack. Besides, note that temporal is not using raw frames to stack,
    it's using optical flow, which is using 2 frames to create one optical flow.
    Therefore, the last frame can't make an optical flow for it. And that makes 
    temporal to cut the last sliding_height frames. But That's a discussion that
    matters only for the next steps, in specific, streams_fextractor.py.

'''

if __name__ == '__main__':
    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)
    print("First weight file is STACK and second is RGB", file=sys.stderr)
    argp = argparse.ArgumentParser(description='Do architecture tasks')
    argp.add_argument("-streams", dest='streams', type=str, nargs='+',
            help='So far, spatial, temporal, pose and its combinations \
                  Usage: -streams spatial temporal',
            required=True)
    argp.add_argument("-weight", dest='weight', type=str, nargs='+', 
            help='Usage: -weight <path_to_your_weight_file>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)
    
    num_features = 4096     # Number of outputs of our CNN
    sliding_height = 10     # Number of elements inside a window
    x_dim = 224             # X input size
    y_dim = 224             # Y input size

    weights = dict()

    # Filling a dict with corresponding values of transfer learning files
    for w in args.weight:
        if 'RGB' in w:
            weights['RGB'] = w
        else:
            weights['STACK'] = w

    # Check the existence of every possible stream in input argument
    if 'temporal' in args.streams:
        
        stack_input = (x_dim, y_dim, 2*sliding_height)
        tensor_stack_input = Input(shape=stack_input)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(tensor_stack_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(num_features, activation='relu', name='fc6')(x)

        model = Model(tensor_stack_input, x, name='VGG16_temporal')

        layers_name = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']

        h5 = h5py.File(weights['STACK'])
             
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        # Copy the weights stored in the weights file to the feature extractor part of the VGG16
        for layer in layers_name:
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (3,2,1,0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            K.set_value(layer_dict[layer].kernel, w2)
            K.set_value(layer_dict[layer].bias, b2)

        # Copy the weights of the first fully-connected layer (fc6)
        layer = 'fc6'
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1,0))
        b2 = np.asarray(b2)
        K.set_value(layer_dict[layer].kernel, w2)
        K.set_value(layer_dict[layer].bias, b2)


        print("Saving your temporal CNN as VGG16_temporal")
        model.save('VGG16_temporal')

    if 'pose' in args.streams:
        model = VGG16(include_top=False)
        
        layers_name = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']

        h5 = h5py.File(weights['RGB'])
             
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        # Copy the weights stored in the weights file to the feature extractor part of the VGG16
        for layer in layers_name:
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (3,2,1,0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            K.set_value(layer_dict[layer].kernel, w2)
            K.set_value(layer_dict[layer].bias, b2)

        input = Input(shape=(x_dim, y_dim, 3),
                        name='pose_input')

        output_vgg16 = model(input)

        x = Flatten(name='flatten')(output_vgg16)
        x = Dense(num_features, activation='relu', name='fc6')(x)

        model = Model(inputs=input, outputs=x, name='VGG16_pose')
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # Copy the weights of the first fully-connected layer (fc6)
        layer = 'fc6'
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1,0))
        b2 = np.asarray(b2)
        K.set_value(layer_dict[layer].kernel, w2)
        K.set_value(layer_dict[layer].bias, b2)

        # This simple keras function can't be used because of the transformations
        # we need to apply and the current format of the data stored in h5
        #model.load_weights(args.weight[0], by_name=True)

        print("Saving your pose-estimation CNN as VGG16_pose")
        model.save('VGG16_pose')

    if 'depth' in args.streams:
        model = VGG16(include_top=False)

        layers_name = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']

        h5 = h5py.File(weights['RGB'])
             
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        # Copy the weights stored in the weights file to the feature extractor part of the VGG16
        for layer in layers_name:
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (3,2,1,0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            K.set_value(layer_dict[layer].kernel, w2)
            K.set_value(layer_dict[layer].bias, b2)
        
        input = Input(shape=(x_dim, y_dim, 3),
                        name='depth_input')

        output_vgg16 = model(input)

        x = Flatten(name='flatten')(output_vgg16)
        x = Dense(num_features, activation='relu', name='fc6')(x)

        model = Model(inputs=input, outputs=x, name='VGG16_depth')
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # Copy the weights of the first fully-connected layer (fc6)
        layer = 'fc6'
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1,0))
        b2 = np.asarray(b2)
        K.set_value(layer_dict[layer].kernel, w2)
        K.set_value(layer_dict[layer].bias, b2)

        # This simple keras function can't be used because of the transformations
        # we need to apply and the current format of the data stored in h5
        #model.load_weights(args.weight[0], by_name=True)

        print("Saving your spatial CNN as VGG16_depth")
        model.save('VGG16_depth')
    
    if 'spatial' in args.streams:
        model = VGG16(include_top=False)

        layers_name = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']

        h5 = h5py.File(weights['RGB'])
             
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        # Copy the weights stored in the weights file to the feature extractor part of the VGG16
        for layer in layers_name:
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (3,2,1,0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            K.set_value(layer_dict[layer].kernel, w2)
            K.set_value(layer_dict[layer].bias, b2)
        
        input = Input(shape=(x_dim, y_dim, 3),
                            name='spatial_input')

        output_vgg16 = model(input)

        x = Flatten(name='flatten')(output_vgg16)
        x = Dense(num_features, activation='relu', name='fc6')(x)

        model = Model(inputs=input, outputs=x, name='VGG16_spatial')
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # Copy the weights of the first fully-connected layer (fc6)
        layer = 'fc6'
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1,0))
        b2 = np.asarray(b2)
        K.set_value(layer_dict[layer].kernel, w2)
        K.set_value(layer_dict[layer].bias, b2)

        # This simple keras function can't be used because of the transformations
        # we need to apply and the current format of the data stored in h5
        #model.load_weights(args.weight[0], by_name=True)

        print("Saving your spatial CNN as VGG16_spatial")
        model.save('VGG16_spatial')

    if 'ritmo' in args.streams:
        model = VGG16(include_top=False)

        layers_name = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']

        h5 = h5py.File(weights['RGB'])
             
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        # Copy the weights stored in the weights file to the feature extractor part of the VGG16
        for layer in layers_name:
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (3,2,1,0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            K.set_value(layer_dict[layer].kernel, w2)
            K.set_value(layer_dict[layer].bias, b2)
        
        input = Input(shape=(x_dim, y_dim, 3),
                        name='ritmo_input')

        output_vgg16 = model(input)

        x = Flatten(name='flatten')(output_vgg16)
        x = Dense(num_features, activation='relu', name='fc6')(x)

        model = Model(inputs=input, outputs=x, name='VGG16_ritmo')
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # Copy the weights of the first fully-connected layer (fc6)
        layer = 'fc6'
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1,0))
        b2 = np.asarray(b2)
        K.set_value(layer_dict[layer].kernel, w2)
        K.set_value(layer_dict[layer].bias, b2)

        # This simple keras function can't be used because of the transformations
        # we need to apply and the current format of the data stored in h5
        #model.load_weights(args.weight[0], by_name=True)

        print("Saving your spatial CNN as VGG_ritmo")
        model.save('VGG16_ritmo')

    if 'saliency' in args.streams:
        model = VGG16(include_top=False)

        layers_name = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']

        h5 = h5py.File(weights['RGB'])
             
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        # Copy the weights stored in the weights file to the feature extractor part of the VGG16
        for layer in layers_name:
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (3,2,1,0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            K.set_value(layer_dict[layer].kernel, w2)
            K.set_value(layer_dict[layer].bias, b2)
        
        input = Input(shape=(x_dim, y_dim, 3),
                        name='ritmo_input')

        output_vgg16 = model(input)

        x = Flatten(name='flatten')(output_vgg16)
        x = Dense(num_features, activation='relu', name='fc6')(x)

        model = Model(inputs=input, outputs=x, name='VGG16_ritmo')
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # Copy the weights of the first fully-connected layer (fc6)
        layer = 'fc6'
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1,0))
        b2 = np.asarray(b2)
        K.set_value(layer_dict[layer].kernel, w2)
        K.set_value(layer_dict[layer].bias, b2)

        # This simple keras function can't be used because of the transformations
        # we need to apply and the current format of the data stored in h5
        #model.load_weights(args.weight[0], by_name=True)

        print("Saving your spatial CNN as VGG_saliency")
        model.save('VGG16_saliency')
