#!/bin/sh

# User inputs
DS_DIR=$1
CHECK_PATH=$2

# Validation
if [ -z $DS_DIR ] 
then
  echo "DS_DIR not set <arg #1>!"; exit
fi
if [ -z $CHECK_PATH ] 
then
  echo "CHECK_PATH not set <arg #2>!"; exit
fi

# Loop through all files in DS_DIR
for class_dir in $(ls $DS_DIR); do
  video_dir_path="$DS_DIR/$class_dir"
  
  # Loop through all videos
  for video_dir in $(ls $video_dir_path); do
    video_file_path="$video_dir_path/$video_dir"
    
    # Loop through all frames 
    for video_file in $(ls "$video_file_path/frame"*); do
      #echo "python3 monodepth_simple.py --image-path $video_file --checkpoint_path $CHECK_PATH"
      echo "$video_file"
      python3 monodepth_simple.py --image_path $video_file --checkpoint_path $CHECK_PATH
    done
  done
done


