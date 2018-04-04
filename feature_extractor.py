class feature_extractor:

    def __init__(self, class0, class1, num_features):
        self.class0 = class0
        self.class1 = class1
        self.num_features = num_features
        self.folders = []
        self.classes = []
        # Total amount of stacks, with sliding window = num_images-L+1
        self.nb_total_stacks = 0

    def subtracte_mean_file(flow, mean):
        return flow - np.tile(flow_mean[...,np.newaxis],
                              (1, 1, 1, flow.shape[3]))

    def open_mean_file(name_file, file_key):
        return sio.loadmat(name_file)[file_key]

    def generator(list1, list2):
        '''
        Auxiliar generator: returns the ith element of both given list with 
        each call to next() 
        '''
        for x,y in zip(list1,list2):
            yield x, y

    def extract(extractor_model, features_file, labels_file,
                        samples_file, num_file, features_key, labels_key,
                        samples_key, num_key, data_folder, sliding_height):
        '''
        Function to load the optical flow stacks, do a feed-forward through 
        the feature extractor (VGG16) and store the output feature vectors in 
        the file 'features_file' and the labels in 'labels_file'.
        Input:
        * extractor_model: model VGG16 until the fc6 layer.
        * features_file: path to the hdf5 file where the extracted features are
        going to be stored
        * labels_file: path to the hdf5 file where the labels of the features
        are going to be stored
        * features_key: name of the key for the hdf5 file to store the features
        * labels_key: name of the key for the hdf5 file to store the labels
        # todo: finish comments
        '''

        flow_mean = open_mean_file(mean_file, 'image_mean')

        # Fill the folders and classes arrays with all the paths to the data
        fall_videos = [f for f in os.listdir(data_folder + self.class0) 
                       if os.path.isdir(os.path.join(data_folder + 
                           self.class0, f))]
        fall_videos.sort()
        for fall_video in fall_videos:
            x_images = glob.glob(data_folder + self.class0 + '/' + 
                                 fall_video + '/flow_x*.jpg')
            if int(len(x_images)) >= 10:
                folders.append(data_folder + self.class0 + '/' + fall_video)
                classes.append(0)

        not_fall_videos = [f for f in os.listdir(data_folder + self.class1) 
                    if os.path.isdir(os.path.join(data_folder + 
                        self.class1, f))]
        not_fall_videos.sort()
        for not_fall_video in not_fall_videos:
            x_images = glob.glob(data_folder + self.class1 + '/' +
                                 not_fall_video + '/flow_x*.jpg')
            if int(len(x_images)) >= 10:
                folders.append(data_folder + self.class1 + '/' + not_fall_video)
                classes.append(1)

        for folder in folders:
            x_images = glob.glob(folder + '/flow_x*.jpg')
            nb_total_stacks += len(x_images)-L+1
        
        # File to store the extracted features and datasets to store them
        # IMPORTANT NOTE: 'w' mode totally erases previous data
        h5features = h5py.File(features_file,'w')
        h5labels = h5py.File(labels_file,'w')
        h5samples = h5py.File(samples_file, 'w')
        h5num_classes = h5py.File(num_file, 'w')

        dataset_features = h5features.create_dataset(features_key, 
                shape=(nb_total_stacks, self.num_features), dtype='float64')
        dataset_labels = h5labels.create_dataset(labels_key, 
                shape=(nb_total_stacks, 1), dtype='float64')  
        dataset_samples = h5samples.create_dataset(samples_key, 
                shape=(len(fall_videos) + len(not_fall_videos), 1), 
                dtype='int32')  
        dataset_num = h5num_classes.create_dataset(num_key, shape=(2, 1), 
                dtype='int32')  
        
        dataset_num[0] = len(fall_videos)
        dataset_num[1] = len(not_fall_videos)

        cont = 0
        number = 0
        
        for folder, label in zip(folders, classes):
            x_images = glob.glob(folder + '/flow_x*.jpg')
            x_images.sort()
            y_images = glob.glob(folder + '/flow_y*.jpg')
            y_images.sort()
            nb_stacks = len(x_images)-sliding_height+1
            # Here nb_stacks optical flow stacks will be stored
            flow = np.zeros(shape=(x_size, y_size, 2*sliding_height, nb_stacks),
                                    dtype=np.float64)
            gen = self.generator(x_images,y_images)
            for i in range(len(x_images)):
                flow_x_file, flow_y_file = next(gen)
                img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                # Assign an image i to the jth stack in the kth position,
                # but also in the j+1th stack in the k+1th position and so on 
                # (for sliding window) 
                for s in list(reversed(range(min(sliding_height,i+1)))):
                    if i-s < nb_stacks:
                        flow[:,:,2*s,  i-s] = img_x
                        flow[:,:,2*s+1,i-s] = img_y
                del img_x,img_y
                gc.collect()
                
            # Subtract mean
            flow = subtracte_mean_file(flow, flow_mean)
            # Transpose for channel ordering (Tensorflow in this case)
            flow = np.transpose(flow, (3, 2, 0, 1)) 
            predictions = np.zeros((nb_stacks, self.num_features), 
                    dtype=np.float64)
            truth = np.zeros((nb_stacks, 1), dtype='int8')
            # Process each stack: do the feed-forward pass and store in the 
            # hdf5 file the output
            for i in range(nb_stacks):
                prediction = extractor_model.predict(np.expand_dims(
                                                                flow[i, ...],0))
                predictions[i, ...] = prediction
                truth[i] = label

            dataset_features[cont:cont+nb_stacks,:] = predictions
            dataset_labels[cont:cont+nb_stacks,:] = truth
            dataset_samples[number] = nb_stacks
            number+=1
            cont += nb_stacks
        h5features.close()
        h5labels.close()
        h5samples.close()
        h5num_classes.close()
