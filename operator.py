class operator:

    def __init__(self, name):
        self.kf_falls = None
        self.kf_nofalls = None
        self.falls = None
        self.no_falls = None
        self.classifier = None

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
    n_splits, num_features, weight_0, epochs, compute_metrics):

        self.pre_training_cross(features_file, labels_file, features_key, 
                                labels_key, n_splits)
        sensitivities = []
        specificities = []
        fars = []
        mdrs = []
        accuracies = []
       
        first = 0

        # CROSS-VALIDATION: Stratified partition of the dataset into 
        # train/test sets
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

            set_classifier(num_features) 

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

    def training(features_file, labels_file, features_key, labels_key, 
    num_features, weight_0, epochs, compute_metrics)
        sensitivities = []
        specificities = []
        fars = []
        mdrs = []
        accuracies = []

        pre_training(features_file, labels_file, features_key, labels_key)

        # todo: change this X, _y
        X = np.concatenate((self.all_features[self.falls, ...], 
            self.all_features[self.no_falls, ...]))
        _y = np.concatenate((self.all_labels[self.falls, ...],
            self.all_labels[self.no_falls, ...]))
        
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

        set_classifier(num_features)

        # ==================== TRAINING ========================     
        # weighting of each class: only the fall class gets a different
        # weight
        class_weight = {0: weight_0, 1: 1}
        # Batch training
        if mini_batch_size == 0:
            history = classifier.fit(X,_y, validation_split=0.15, 
                    batch_size=X.shape[0], epochs=epochs, shuffle='batch',
                    class_weight=class_weight)
        else:
            history = classifier.fit(X,_y, validation_split=0.15, 
                    batch_size=mini_batch_size, nb_epoch=epochs, 
                    shuffle='batch', class_weight=class_weight)
        plot_training_info(exp, ['accuracy', 'loss'], save_plots, 
                           history.history)

        classifier.save('urfd_classifier.h5')

        # ==================== EVALUATION ========================        
        if compute_metrics:
            predicted = classifier.predict(np.asarray(X2))
            evaluate(predicted, X2, _y2, sensitivities, specificities, fars,
                    mdrs, accuracies)
            check_videos(_y2, predicted, samples_key, training_samples_file,
                    num_key, training_samples_file) 

    def evaluate(predicted, X2, _y2, sensitivities, specificities, fars, mdrs, 
                                                                   accuracies):
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)
        # Compute metrics and print them
        cm = confusion_matrix(_y2, predicted,labels=[0,1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp/float(tp+fn)
        fpr = fp/float(fp+tn)
        fnr = fn/float(fn+tp)
        tnr = tn/float(tn+fp)
        precision = tp/float(tp+fp)
        recall = tp/float(tp+fn)
        specificity = tn/float(tn+fp)
        f1 = 2*float(precision*recall)/float(precision+recall)
        accuracy = accuracy_score(_y2, predicted)

        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))   
        print('Sensitivity/Recall: {}'.format(recall))
        print('Specificity: {}'.format(specificity))
        print('Precision: {}'.format(precision))
        print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))

        # Store the metrics for this epoch
        sensitivities.append(tp/float(tp+fn))
        specificities.append(tn/float(tn+fp))
        fars.append(fpr)
        mdrs.append(fnr)
        accuracies.append(accuracy)

    def check_videos(_y2, predicted, samples_key, samples_file, num_key, 
    num_file):

        h5samples = h5py.File(samples_file, 'r')
        h5num = h5py.File(num_file, 'r')

        all_samples = np.asarray(h5samples[samples_key])
        all_num = np.asarray(h5num[num_key])

        video = 1
        inic = 0
        misses = 0

        msage_fall = list("###### Fall videos ")
        msage_fall.append(str(all_num[0][0]))
        msage_fall.append(" ######")
        msage_not_fall = list("###### Not fall videos ")
        msage_not_fall.append(str(all_num[1][0]))
        msage_not_fall.append(" ######")

        for x in range(len(all_samples)):
            correct = 1

            if all_samples[x][0] == 0:
                continue

            if x == 0:
                print(''.join(msage_fall))
            elif x == all_num[1][0]:
                print(''.join(msage_not_fall))
                video = 1 

            for i in range(inic, inic + all_samples[x][0]):
                if i >= len(predicted):
                   break 
                elif predicted[i] != _y2[i]:
                    misses+=1
                    correct = 0

            if correct == 1:
               print("Hit video: " + str(video))
            else:
               print("Miss video: " + str(video))

            video += 1
            inic += all_samples[x][0]

    def set_classifier(num_features):
        extracted_features = Input(shape=(num_features,), dtype='float32',
                                   name='input')
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99, 
                                   epsilon=0.001)(extracted_features)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(extracted_features)
       
        x = Dropout(0.9)(x)
        x = Dense(num_features, name='fc2', 
                  kernel_initializer='glorot_uniform')(x)
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(x)
        x = Dropout(0.8)(x)
        x = Dense(1, name='predictions', 
                  kernel_initializer='glorot_uniform')(x)
        x = Activation('sigmoid')(x)
        
        self.classifier = Model(input=extracted_features, output=x, 
                           name='classifier')
        self.classifier.compile(optimizer=adam, loss='binary_crossentropy',
                           metrics=['accuracy'])


