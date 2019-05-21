import numpy as np
import h5py

h5 = h5py.File('weights_new.h5', 'r+')

new_data = np.zeros( (64, 3, 3, 3), float ) 
for i in range(64): #64
    for j in range(3): #3
        
        for l in range(3): #3
            avg = 0.0
            for k in range(20): #20
                avg += h5['data']['block1_conv1']['0'][i][k][j][l]

            avg /= 20 

            for k in range(3):
                new_data[i][k][j][l] = avg

del h5['data']['block1_conv1']['0']

dset = h5.create_dataset('data/block1_conv1/0', data=new_data)
h5.close()
