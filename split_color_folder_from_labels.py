import os
import shutil
import pickle
import argparse
from tqdm import tqdm
from glob import glob
import sys

def main():
    parser = argparse.ArgumentParser(description='bottle output split folder')
    parser.add_argument('--label_dir', type=str, default="images", help='path of the label directory')
    parser.add_argument('--image_dir', type=str, default="images", help='path of the image directory')

    args = parser.parse_args()
    image_dir = args.image_dir+'_split/bottle/'
    label_dir = args.label_dir
    final_dir = args.image_dir + '_split_2'

    directory_dict = {'0': 'brown_and_darks', '1': 'clear_light_blue', '2': 'dark_blue', '3': 'green', '4': 'opaque',
                      '5': 'other', '6': 'sleeve_clear'}

    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    directories = list(directory_dict.values())

    for dir_ in directories:
        if not os.path.isdir(os.path.join(final_dir, dir_)):
            os.mkdir(os.path.join(final_dir, dir_))

    labels = glob(label_dir + '/*.p')
    print('Copying output images in final corresponding directories......!')
    for label in tqdm(labels):
        img_file = os.path.basename(label).split('.')[0] + '.jpg'
        label_content = pickle.load(open( label, 'rb'))
        label = list(label_content.values())[0][0][0]
        if str(label) !='1': 
            shutil.copy(os.path.join(image_dir, img_file), os.path.join(final_dir, directory_dict[str(label)], img_file))


if __name__ == '__main__':
    main()
