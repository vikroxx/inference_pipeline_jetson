import os
import cv2
from tqdm import tqdm
from unique_objects import image_tuple_list_generator
import argparse


def concatenate_and_save_images(img_tuple_list, img_dir, final_dir, empty0, empty1):
    for img_tuple in tqdm(img_tuple_list):
        if 'empty' in img_tuple[0]:
            img1 = cv2.imread(empty1)
        else:
            img1 = cv2.imread(os.path.join(img_dir, img_tuple[0]))

        if 'empty' in img_tuple[1]:
            img2 = cv2.imread(empty0)
        else:
            img2 = cv2.imread(os.path.join(img_dir, img_tuple[1]))

        concat_img = cv2.hconcat([img1, img2])
        cv2.imwrite(os.path.join(final_dir, img_tuple[0].split('.')[0][:-2] + '.jpg'), concat_img)


def main():
    parser = argparse.ArgumentParser(description='food output split folder')
    parser.add_argument('--image_dir', type=str, default="images", help='path of the image directory')
    parser.add_argument('--final_dir', type=str, default="images", help='final dir path')
    parser.add_argument('--left_image_index', type=str)

    args = parser.parse_args()

    empty_0 = os.path.join('empty_conv', 'empty_0.jpg')
    empty_1 = os.path.join('empty_conv', 'empty_1.jpg')

    image_dir = args.image_dir
    final_dir = args.final_dir

    img_tuple_list = image_tuple_list_generator(image_dir, empty_0, empty_1, left_image_index=args.left_image_index)

    # Generate concatenated images
    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)
    concatenate_and_save_images(img_tuple_list, image_dir, final_dir, empty_0, empty_1)


if __name__ == '__main__':
    main()
