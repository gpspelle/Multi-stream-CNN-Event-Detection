import sys
import h5py
import argparse
import numpy as np
import copy
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, \
        ZeroPadding2D, Flatten, Activation, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K

import sys
sys.setrecursionlimit(3000)

''' This code is based on Deep Residual Learning for Image Recognition.
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    arXiv:1512.03385
    Also, new features were added by Gabriel Pellegrino Silva working in 
    Semantix. 
'''

''' Documentation: class Resnet152 
    
    This class has only one method:

    weight_init

    The methods that should be called outside of this class are:

    weight_init: receives as parameter a file containing weights of a trained
    CNN
'''

class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` 
            argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` 
            argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero',
                 gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%
                     self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        return K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta,
               broadcast_shape)

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main 
        path.
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_base = 'res' + str(stage) + block + '_branch'
    bn_base = 'bn' + str(stage) + block + '_branch'
    scale_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_base + '2a',
            use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_base + '2a')(x)
    x = Activation('relu', name=conv_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_base + '2b', 
                                                             use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_base + '2b')(x)
    x = Activation('relu', name=conv_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main 
        path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=
    (2,2) and the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_base = 'res' + str(stage) + block + '_branch'
    bn_base = 'bn' + str(stage) + block + '_branch'
    scale_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, 
            name=conv_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_base + '2a')(x)
    x = Activation('relu', name=conv_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_base + '2b')(x)
    x = Activation('relu', name=conv_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, 
            name=bn_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

class ResNet152:
    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    def __init__(self, num_features, x_size, y_size): 
        self.model = None

        eps = 1.1e-5

        sliding_height = 10

        # Handle Dimension Ordering for different backends
        global bn_axis
        bn_axis = 3
        img_input = Input(shape=(x_size, y_size, 2*sliding_height), 
                    name='data')
                
        x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
        x = Scale(axis=bn_axis, name='scale_conv1')(x)
        x = Activation('relu', name='conv1_relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        for i in range(1,8):
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        for i in range(1,36):
            x = identity_block(x, 3, [256,256,1024], stage=4, block='b'+str(i))

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_fc = Flatten()(x_fc)
        x_fc = Dense(num_features, activation='softmax', 
                kernel_initializer='glorot_uniform', name='fc1000')(x_fc)

        self.model = Model(img_input, x_fc)
    
    def weight_init(self, weights_file):
        self.model.load_weights(weights_file, by_name=True)

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
    argp.add_argument("-weight", dest='weight', type=str, nargs=1, 
            help='Usage: -weight <path_to_your_weight_file>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    '''
        todo: treinar no UCF101
    '''
    
    arch = ResNet152(args.num_features[0], args.input_dim[0], 
           args.input_dim[1])
        
    arch.weight_init(args.weight[0])

    print("Saving your CNN as RESNET152")
    arch.model.save('RESNET152')

