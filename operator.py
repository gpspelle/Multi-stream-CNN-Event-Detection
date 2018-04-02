class operator:

    def __init__(self, name):
        self.kf_falls = None
        self.kf_nofalls = None
        self.falls = None
        self.no_falls = None

    def pre_training_cross(features_file, labels_file, features_key, 
                           labels_key, n_splits):
        self.pre_training(features_file, labels_file, features_key, labels_key)
        
        # Use a 'n_splits' fold cross-validation
        self.kf_falls = KFold(n_splits=n_splits)
        self.kf_nofalls = KFold(n_splits=n_splits)


    def pre_training(features_file, labels_file, features_key, labels_key):
        h5features = h5py.File(features_file, 'r')
        h5labels = h5py.File(labels_file, 'r')
        
        # all_features will contain all the feature vectors extracted from 
        # optical flow images
        all_features = h5features[features_key]
        all_labels = np.asarray(h5labels[labels_key])
        
        # Falls are related to 0 and not falls to 1
        self.falls = np.asarray(np.where(all_labels==0)[0])
        self.no_falls = np.asarray(np.where(all_labels==1)[0])
        self.falls.sort()
        self.no_falls.sort()
        

    def cross_training(features_file, labels_file, features_key, labels_key,
                       n_splits):

        self.pre_training_cross(features_file, labels_file, features_key, 
                                labels_key, n_splits)
        sensitivities = []
        specificities = []
        fars = []
        mdrs = []
        accuracies = []
       
        first = 0

        # CROSS-VALIDATION: Stratified partition of the dataset into 
        # train/test setes
        for (train_falls, test_falls), (train_nofalls, test_nofalls) 
            in zip(self.kf_falls.split(self.all_features[self.falls, ...]), 
                    self.kf_nofalls.split(
                        self.all_features[self.no_falls, ...])):
            train_falls = np.asarray(train_falls)
            test_falls = np.asarray(test_falls)
            train_nofalls = np.asarray(train_nofalls)
            test_nofalls = np.asarray(test_nofalls)

            # todo: change this X, _y, X2 and _y2 variables name
            X = np.concatenate((self.all_features[train_falls, ...], 
                self.all_features[train_nofalls, ...]))
            _y = np.concatenate((self.all_labels[train_falls, ...],
                self.all_labels[train_nofalls, ...]))
            X2 = np.concatenate((self.all_features[test_index_falls, ...],
                self.all_features[test_index_nofalls, ...]))
            _y2 = np.concatenate((self.all_labels[test_falls, ...], 
                self.all_labels[test_nofalls, ...]))   
            
            # todo: Is this working? 
            # Balance the number of positive and negative samples so that there
            # is the same amount of each of them
            all0 = np.asarray(np.where(_y==0)[0])
            all1 = np.asarray(np.where(_y==1)[0])  
            if len(all0) < len(all1):
                all1 = np.random.choice(all1, len(all0), replace=False)
            else:
                all0 = np.random.choice(all0, len(all1), replace=False)
            allin = np.concatenate((all0.flatten(),all1.flatten()))
            allin.sort()
            X = X[allin,...]
            _y = _y[allin]

            # ==================== CLASSIFIER ========================
            extracted_features = Input(shape=(num_features,), dtype='float32',
                                       name='input')
            if batch_norm:
                x = BatchNormalization(axis=-1, momentum=0.99, 
                                       epsilon=0.001)(extracted_features)
                x = Activation('relu')(x)
            else:
                x = ELU(alpha=1.0)(extracted_features)
           
            x = Dropout(0.9)(x)
            x = Dense(4096, name='fc2', kernel_initializer='glorot_uniform')(x)
            if batch_norm:
                x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
                x = Activation('relu')(x)
            else:
                x = ELU(alpha=1.0)(x)
            x = Dropout(0.8)(x)
            x = Dense(1, name='predictions', 
                      kernel_initializer='glorot_uniform')(x)
            x = Activation('sigmoid')(x)
            
            classifier = Model(input=extracted_features, output=x, 
                               name='classifier')
            classifier.compile(optimizer=adam, loss='binary_crossentropy',
                               metrics=['accuracy'])
            
            # ==================== TRAINING ========================     
            # weighting of each class: only the fall class gets a different
            # weight
            class_weight = {0: weight_0, 1: 1}
            # Batch training
            if mini_batch_size == 0:
                history = classifier.fit(X,_y, validation_data=(X2,_y2), 
                        batch_size=X.shape[0], epochs=epochs, shuffle='batch',
                        class_weight=class_weight)
            else:
                history = classifier.fit(X,_y, validation_data=(X2,_y2), 
                        batch_size=mini_batch_size, nb_epoch=epochs, 
                        shuffle='batch', class_weight=class_weight)
            plot_training_info(exp, ['accuracy', 'loss'], save_plots, 
                               history.history)

            # Store only the first classifier
            if first == 0:
                classifier.save('urfd_classifier.h5')
                first = 1

            # ==================== EVALUATION ========================        
            if compute_metrics:
               predicted = classifier.predict(np.asarray(X2))
               evaluate(predicted, X2, _y2, sensitivities, specificities, fars,
                       mdrs, accuracies)
               check_videos(_y2, predicted, samples_key, training_samples_file,
                       num_key, training_samples_file) 
        
        print('5-FOLD CROSS-VALIDATION RESULTS ===================')
        print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities), 
                                                    np.std(sensitivities)))
        print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities),
                                                    np.std(specificities)))
        print("FAR: %.2f%% (+/- %.2f%%)" % (np.mean(fars), np.std(fars)))
        print("MDR: %.2f%% (+/- %.2f%%)" % (np.mean(mdrs), np.std(mdrs)))
        print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies), 
                                                 np.std(accuracies)))

    def training:
        
