from architech  import *
from operator import *
from fextractor import *

# CHANGE THESE VARIABLES
class0 = 'Falls'
class1 = 'NotFalls'
num_features = 4096
arch_name = 'VGG16'

#training_folder = '/home/ubuntu/gabriel/ssd_drive/UR_Fall_OF/'
training_folder = '/home/ubuntu/gabriel/ssd_drive/Fall_val/'

#evaluation_folder = '/home/ubuntu/gabriel/ssd_drive/UR_Fall_OF/'
evaluation_folder = '/home/ubuntu/gabriel/ssd_drive/Fall_val/'

mean_file = 'flow_mean.mat'
vgg_16_weights = 'weights.h5'
model_file = 'models/exp_'
weights_file = 'weights/exp_'

#training_features_file = 'features_urfd.h5'
#training_labels_file = 'labels_urfd.h5'
#training_samples_file = 'samples_urfd.h5'
#training_num_file = 'num_urfd.h5'

training_features_file = 'features_val.h5'
training_labels_file = 'labels_val.h5'
training_samples_file = 'samples_val.h5'
training_num_file = 'num_val.h5'

#evaluation_features_file = 'features_urfd.h5'
#evaluation_labels_file = 'labels_urfd.h5'
#evaluation_samples_file = 'samples_urfd.h5'
#evaluation_num_file = 'num_urfd.h5'

#evaluation_features_file = 'features_val.h5'
#evaluation_labels_file = 'labels_val.h5'
#evaluation_samples_file = 'samples_val.h5'
#evaluation_num_file = 'num_val.h5'

evaluation_features_file = 'fake_val.h5'
evaluation_labels_file = 'fakee_val.h5'
evaluation_samples_file = 'fakeee_val.h5'
evaluation_num_file = 'fakeeee_val.h5'

features_key = 'features'
labels_key = 'labels'
samples_key = 'samples'
num_key = 'num'

sliding_height = 10
batch_norm = True
learning_rate = 0.0001
mini_batch_size = 0
weight_0 = 1
epochs = 200

save_plots = True
extract_features_training = False
extract_features_evaluation = True

cross_training = False

do_training = False 
do_evaluation = True 
compute_metrics = True
threshold = 0.5


extractor = Fextractor(class0, class1, num_features)
arch = Architech(arch_name, num_features)
#operator = Operator()

if do_training:

    if extract_features_training:
        extractor.extract(arch.model, training_features_file, 
        training_labels_file, training_samples_file, training_num_file, 
        features_key, labels_key, samples_key, num_key, training_folder, 
        sliding_height, mean_file)

    if cross_training:
        operator.cross_training(training_features_file, training_labels_file, 
        features_key, labels_key, n_splits, num_features, weight_0, epochs, 
        compute_metrics)
    else:
        operator.training(training_features_file, training_labels_file, 
        features_key, labels_key, num_features, weight_0, epochs, 
        compute_metrics)

if do_evaluation:

    if extract_features_evaluation:
        extractor.extract(arch.model, evaluation_features_file, 
        evaluation_labels_file, evaluation_samples_file, evaluation_num_file,
        features_key, labels_key, samples_key, num_key, evaluation_folder,
        sliding_height, mean_file)

    operator.prepare_evaluate(evaluation_features_file, evaluation_labels_file,
    features_key, labels_key)
