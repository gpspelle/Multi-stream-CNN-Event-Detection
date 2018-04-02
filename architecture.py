class architecture: 

    def __init__(self, name):
        self.name = name
        self.model = None
        self.layers_name = []

        if name == 'VGG16':
            self.model = Sequential()

            self.model.add(ZeroPadding2D((1, 1), input_shape=(20, 224, 224)))
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
            self.model.add(Dense(num_features, name='fc6', 
                kernel_initializer='glorot_uniform'))

            self.layers_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 
                    'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 
                    'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 
                    'fc8']

    def weight_init(weights_file):
        '''
        Input:
        * weights_file: path to a hdf5 file containing weights and biases to a
        trained network
        
        '''
        h5 = h5py.File(weights_file)
        
        layer_dict = dict([(self.layers_name, layer) 
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

    
