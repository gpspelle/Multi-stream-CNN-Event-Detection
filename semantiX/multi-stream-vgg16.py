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
    
    num_features = 4096
    sliding_height = 10
    x_dim = 224
    y_dim = 224

    weights = dict()

    for w in args.weight:
        if 'RGB' in w:
            weights['RGB'] = w
        else:
            weights['STACK'] = w
    
    if 'temporal' in args.streams:
        
        # All this must be done instead of calling VGG16 as the others, because
        # of the input format needs a channel 3, and for temporal we're using
        # 2 * sliding_height as channel. 


        # TODO: CAN BE CHANGED, SEE multi-stream-resnet50.py

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
                        name='pose_input')

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
                        name='pose_input')

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


