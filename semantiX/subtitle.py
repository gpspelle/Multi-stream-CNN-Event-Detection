''' Documentation: class Subtitle
    
    This class has a few methods:

    pre_result
    result
    check_videos

    The methods that should be called outside of this class are:

    result: show the results of a prediction based on a feed forward on the
    classifier of this worker.

'''

class Subtitle:

    def __init__(self, data, class0, class1, threshold, extract_id):
       
        self.features_key = 'features' 
        self.labels_key = 'labels'
        self.samples_key = 'samples'
        self.num_key = 'num'

        self.features_file = "features_" + extract_id + ".h5"
        self.labels_file = "labels_" + extract_id + ".h5"
        self.samples_file = "samples_" + extract_id + ".h5"
        self.num_file = "num_" + extract_id + ".h5"
    
        self.data = data
        self.class0 = class0
        self.class1 = class1
        self.threshold = threshold

        self.result = Result(class0, class1, threshold, extract_id)

    def create_subtitle(self):

        # todo: change X and Y variable names
        X, Y, predicted = self.result.pre_result()

        for i in range(len(predicted)):
            if predicted[i] < self.threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)
       
        h5samples = h5py.File(self.samples_file, 'r')
        h5num = h5py.File(self.num_file, 'r')

        all_samples = np.asarray(h5samples[self.samples_key])
        all_num = np.asarray(h5num[self.num_key])

        
if __name__ == '__main__':

    '''
        todo: verify if all these parameters are really required
    '''

    print("***********************************************************",
            file=sys.stderr)
    print("             SEMANTIX - UNICAMP DATALAB 2018", file=sys.stderr)
    print("***********************************************************",
            file=sys.stderr)

    argp = argparse.ArgumentParser(description='Do subtitle tasks')
    argp.add_argument("-data", dest='data_folder', type=str, nargs=1, 
            help='Usage: -data <path_to_your_data_folder>', required=True)
    argp.add_argument("-thresh", dest='thresh', type=float, nargs=1,
            help='Usage: -thresh <x> (0<=x<=1)', required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-id", dest='id', type=str, nargs=1,
        help='Usage: -id <identifier_to_this_features>', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    subt = Subtitle(args.data[0], args.classes[0], args.classes[1], args.thresh[0], 
                args.id[0])

    subt.create_subtitle()

'''
    todo: criar excecoes para facilitar o uso
'''

'''
    todo: nomes diferentes para classificadores
'''
