import os
import shutil
import pickle
import argparse
from tqdm import tqdm
from glob import glob


def main():
    parser = argparse.ArgumentParser(description='food output split folder')
    parser.add_argument('--original_crop_dir', type=str, default="images", help='path of the image directory')
    parser.add_argument('--bottle_image_dir', type=str, default="images",
                        help='path of the image directory containg images with output text written over it')
    parser.add_argument('--final_dir', type=str, default="images",
                        help='path of the image directory containg images with output text written over it')

    args = parser.parse_args()
    original_crop_dir = args.original_crop_dir
    bottle_image_dir = args.bottle_image_dir
    final_dir = args.final_dir

    print('Creating separate directory for bottle only crop outs........!')
    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    images = os.listdir(bottle_image_dir)
    for image in tqdm(images):
        shutil.copy(os.path.join(original_crop_dir, image), os.path.join(final_dir, image))


if __name__ == '__main__':
    main()
