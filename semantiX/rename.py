import numpy as np
import h5py

replace = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 
           'block3_conv1', 'block3_conv2', 'block3_conv3',
           'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1',
           'block5_conv2', 'block5_conv3', '0', '0', '0', '0', '0', '0',
           'block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 
           'block5_pool', '1']

h5 = h5py.File('weights_vgg16.h5', 'r+')
for i in h5:
    l = list(h5[i])
    for j in range(len(l)):
        if replace[j] == '0':
            continue
        elif replace[j] == '1':
            break

        h5[i][replace[j]] = h5[i][l[j]]
        del h5[i][l[j]]

h5.close()
