import sys
import argparse
import h5py
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Activation, \
                         Dense, Dropout, ZeroPadding2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


''' This code is based on Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, 
    I. (2017). "Vision-Based Fall Detection with Convolutional Neural Networks"
    Wireless Communications and Mobile Computing, 2017.
    Also, new features were added by Gabriel Pellegrino Silva working in 
    Semantix. 
'''

''' Documentation: script multi-stream-vgg16 
    
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
    argp.add_argument("-num_feat", dest='num_features', type=int, nargs=1,
            help='Usage: -num_feat <size_of_features_array>', required=True)
    argp.add_argument("-input_dim", dest='input_dim', type=int, nargs=2, 
            help='Usage: -input_dim <x_dimension> <y_dimension>', required=True)
    argp.add_argument("-weight", dest='weight', type=str, nargs=1, 
            help='Usage: -weight <path_to_your_weight_file>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)
    
    sliding_height = 10
    
    if 'temporal' in args.streams:
        model = VGG16(include_top=False, input_shape=(args.input_dim[0], 
                        args.input_dim[1], 2*sliding_height))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(args.num_features[0], name='fc6', 
                  kernel_initializer='glorot_uniform'))

        model.load_weights(args.weight[0], by_name=True)
        model = Model(inputs=model.input, outputs=top_model(model.output))
        print("Saving your temporal CNN as VGG16_temporal")
        top_model.save('VGG16_temporal')

    if 'pose' in args.streams:
        model = VGG16(include_top=False, 
                        input_shape=(args.input_dim[0], args.input_dim[1], 3))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(args.num_features[0], name='fc6', 
                  kernel_initializer='glorot_uniform'))

        model.load_weights(args.weight[0], by_name=True)
        model = Model(inputs=model.input, outputs=top_model(model.output))
        print("Saving your pose-estimation CNN as VGG16_pose")
        model.save('VGG16_pose')
    
    if 'spatial' in args.streams:
        model = VGG16(include_top=False, 
                        input_shape=(args.input_dim[0], args.input_dim[1], 3))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(args.num_features[0], name='fc6', 
                  kernel_initializer='glorot_uniform'))

        model.load_weights(args.weight[0], by_name=True)
        model = Model(inputs=model.input, outputs=top_model(model.output))
        print("Saving your spatial CNN as VGG16_spatial")
        model.save('VGG16_spatial')

'''
    todo: criar excecoes para facilitar o uso
'''
'''
    todo: parametros deveriam ser opcionais e nao obrigatorios.
    Alternativas: tornar os parametros como opcao se voce quiser alterar o
    default? Mas ai terei de mudar o tratamento do caso em que nenhum parametro
    eh passado, pois dai nenhum eh requerido mais. Trade-offs de design
'''
