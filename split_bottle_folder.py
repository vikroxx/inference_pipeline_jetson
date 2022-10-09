import os
import shutil
import pickle
import argparse
from tqdm import tqdm
from glob import glob


def main():
    parser = argparse.ArgumentParser(description='Bottle output split folder')
    parser.add_argument('--image_and_label_dir', type=str, default="images", help='path of the image directory')

    args = parser.parse_args()
    image_dir = args.image_and_label_dir
    final_dir = image_dir + '_split'

    directory_dict = {'0': 'bottle', '1': 'no_bottle'}
    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    directories = list(directory_dict.values())

    for dir_ in directories:
        if not os.path.isdir(os.path.join(final_dir, dir_)):
            os.mkdir(os.path.join(final_dir, dir_))

    images = glob(image_dir + '/*.jpg')
    print('Moving output images in final corresponding directories......!')
    for image in tqdm(images):
        label_file = os.path.basename(image).split('.')[0] + '.p'
        label_file_content = pickle.load(open(os.path.join(image_dir, label_file), 'rb'))
        label = list(label_file_content.values())[0][0][0]
        shutil.move(image, os.path.join(final_dir, directory_dict[str(label)], os.path.basename(image)))


if __name__ == '__main__':
    main()
