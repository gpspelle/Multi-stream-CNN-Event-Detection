import os
import sys
import argparse

if __name__ == '__main__':
    argp = argparse.ArgumentParser(description='auxiliar script for get_skeleton_video')
    argp.add_argument("-data", dest='data_folder', type=str, nargs=1, 
            help='Usage: -data <path_to_your_data_folder>', required=True)
    argp.add_argument("-class", dest='classes', type=str, nargs='+', 
            help='Usage: -class <class0_name> <class1_name>..<n-th_class_name>',
            required=True)
    argp.add_argument("-ext", dest='ext', type=str, nargs=1, 
            help='Usage: -ext <file_extension> .mp4 | .avi | ...', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    classes_dirs = []
    classes_videos = []

    for c in args.classes[0]:
        classes_dirs.append([f for f in os.listdir(args.data_folder[0] + c) 
                    if os.path.isdir(os.path.join(args.data_folder[0], c, f))])
        classes_dirs[-1].sort()

        classes_videos.append([])
        for f in classes_dirs[-1]:
            classes_videos[-1].append(args.data_folder[0] + c + '/' + f +
                               '/' + f + args.ext[0])

        classes_videos[-1].sort()

    for c in args.classes[0]:
        for video in classes_videos[c]:
            print(video)
            os.command('python3 get_skeleton_video.py -video ' + video)

