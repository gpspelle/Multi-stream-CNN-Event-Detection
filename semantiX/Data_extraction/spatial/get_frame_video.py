import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get frame Video')
    parser.add_argument('-video', type=str, default='', required=True)

    try:
        args = argp.parse_args()
    except:
        argp.print_help(sys.stderr)
        exit(1)

    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    i = 0
    while cap.isOpened():
        i+=1
        ret_val, image = cap.read()

        if ret_val == False:
            break

        cv2.imwrite(video + '/frame_' + str(i).zfill(5) + '.jpg', image)

    cv2.destroyAllWindows()
logger.debug('finished+')
