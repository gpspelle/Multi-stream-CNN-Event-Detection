import sys
import argparse
import h5py
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Activation, \
                         Dense, Dropout, ZeroPadding2D
from keras import backend as K
K.set_image_dim_ordering('th')

''' This code is based on Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, 
    I. (2017). "Vision-Based Fall Detection with Convolutional Neural Networks"
    Wireless Communications and Mobile Computing, 2017.
    Also, new features were added by Gabriel Pellegrino Silva working in 
    Semantix. 
'''

''' Documentation: class Architech
    
    This class has only one method:

    weight_init

    The methods that should be called outside of this class are:

    weight_init: receives as parameter a file containing weights of a trained
    CNN
'''

class Architech: 

    def __init__(self, arch, num_features, x_size, y_size):
        '''
        Input: TODO
        Output: TODO
        '''

        self.arch_name = arch 
        self.model = None
        self.layers_name = []
        self.num_features = num_features
        self.x_size = x_size
        self.y_size = y_size

        if self.arch_name == 'VGG16':
            self.layers_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 
                    'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 
                    'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 
                    'fc8']
            self.model = Sequential()

            self.model.add(ZeroPadding2D((1, 1), input_shape=(20, self.x_size, 
                           self.y_size)))
            self.model.add(Convolution2D(64, (3, 3), activation='relu', 
                           name='conv1_1'))
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(64, (3, 3), activation='relu', 
                           name='conv1_2'))
            self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(128, (3, 3), activation='relu',
                      name='conv2_1'))
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(128, (3, 3), activation='relu', 
                      name='conv2_2'))
            self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(256, (3, 3), activation='relu',
                      name='conv3_1'))
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(256, (3, 3), activation='relu',
                      name='conv3_2'))
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(256, (3, 3), activation='relu', 
                      name='conv3_3'))
            self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', 
                      name='conv4_1'))
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', 
                      name='conv4_2'))
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', 
                      name='conv4_3'))
            self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', 
                      name='conv5_1'))
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', 
                      name='conv5_2'))
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', 
                      name='conv5_3'))
            self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

            self.model.add(Flatten())
            self.model.add(Dense(self.num_features, name='fc6', 
                kernel_initializer='glorot_uniform'))
        else:

            '''
                execcao de nome invalido
            '''

            print("***********************************************************",
                    file=sys.stderr)
            print("You've entered an invalid name for your network. Also, it's",
                    file=sys.stderr)
            print("case sensitive. Supported CNN's are: VGG16", file=sys.stderr)
            print("***********************************************************",
                    file=sys.stderr)

    def weight_init(self, weights_file):
        '''
        Input:
        * weights_file: path to a hdf5 file containing weights and biases to a
        trained network
        
        Output: TODO
        '''

        h5 = h5py.File(weights_file)
        
        layer_dict = dict([(layer.name, layer) 
                            for layer in self.model.layers])

        # Copy the weights stored in the 'weights_file' file to the feature 
        # extractor part of the CNN architecture
        for layer in self.layers_name[:-3]:
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (3,2,1,0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            K.set_value(layer_dict[layer].kernel, w2)
            K.set_value(layer_dict[layer].bias, b2)
          
        # Copy the weights of the first fully-connected layer (fc6)
        layer = self.layers_name[-3]
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1,0))
        b2 = np.asarray(b2)
        K.set_value(layer_dict[layer].kernel, w2)
        K.set_value(layer_dict[layer].bias, b2)

if __name__ == '__main__':
    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)
    argp = argparse.ArgumentParser(description='Do architecture tasks')
    argp.add_argument("-num_feat", dest='num_features', type=int, nargs=1,
            help='Usage: -num_feat <size_of_features_array>', required=True)
    argp.add_argument("-input_dim", dest='input_dim', type=int, nargs=2, 
            help='Usage: -input_dim <x_dimension> <y_dimension>', required=True)
    argp.add_argument("-cnn_arch", dest='cnn_arch', type=str, nargs=1,
            help='Usage: -cnn_arch <name_of_desired_architecture>', 
            required=True)
    argp.add_argument("-weight", dest='weight', type=str, nargs=1, 
            help='Usage: -weight <path_to_your_weight_file>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)
    
    arch = Architech(args.cnn_arch[0], args.num_features[0], args.input_dim[0], 
                     args.input_dim[1])
        
    arch.weight_init(args.weight[0])

    arch.model.save(args.cnn_arch[0])

'''
    todo: criar excecoes para facilitar o uso
'''
'''
    todo: parametros deveriam ser opcionais e nao obrigatorios.
    Alternativas: tornar os parametros como opcao se voce quiser alterar o
    default? Mas ai terei de mudar o tratamento do caso em que nenhum parametro
    eh passado, pois dai nenhum eh requerido mais. Trade-offs de design
'''
