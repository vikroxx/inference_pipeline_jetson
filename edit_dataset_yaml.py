import json
import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_dir', type=str, default='/home/vikrant/Desktop/python_pipelines/yolov5')
    parser.add_argument('--image_dir', type=str, default='/home/vikrant/Desktop/08_09/images')
    args = parser.parse_args()
    yolo_dir = args.yolo_dir
    image_dir = args.image_dir

    try:
        yaml_file = open(os.path.join(yolo_dir, 'dataset.yaml'), 'r').readlines()
    except Exception as e:
        print('Failed to open dataset.yaml due to error {}!!'.format(e))
        sys.exit()

    yaml_file[2] = 'val : {}\n'.format(image_dir)

    with open(os.path.join(yolo_dir, 'dataset.yaml'), 'w') as f:
        f.writelines(yaml_file)
