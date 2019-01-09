import os

#os.system('python3 temporal_fextractor.py -data /mnt/hotstorage/Data/URFD-TEST/ -class Falls NotFalls -cnn_arch VGG16_temporal -id URFD-TEST')
#os.system('python3 spatial_fextractor.py -data /mnt/hotstorage/Data/URFD-TEST/ -class Falls NotFalls -cnn_arch VGG16_spatial -id URFD-TEST')
#os.system('python3 pose_fextractor.py -data /mnt/hotstorage/Data/URFD-TEST/ -class Falls NotFalls -cnn_arch VGG16_pose -id URFD-TEST')

os.system('python3 temporal_fextractor.py -data /mnt/hotstorage/Data/URFD-TRAIN/ -class Falls NotFalls -cnn_arch VGG16_temporal -id flip-URFD-TRAIN')
os.system('python3 spatial_fextractor.py -data /mnt/hotstorage/Data/URFD-TRAIN/ -class Falls NotFalls -cnn_arch VGG16_spatial -id flip-URFD-TRAIN')
os.system('python3 pose_fextractor.py -data /mnt/hotstorage/Data/URFD-TRAIN/ -class Falls NotFalls -cnn_arch VGG16_pose -id flip-URFD-TRAIN')

