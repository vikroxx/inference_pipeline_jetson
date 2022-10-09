import json
import os
import sys
import time
from unique_objects import retained_labels, image_tuple_list_generator, unique_objects_from_pair, \
    unique_labels_after_retained_labels_filtering
import pickle
from tqdm import tqdm
import cv2
import argparse



def unique_objects_from_detections(img_dir, label_dir, final_dir, left_image_index):
    # pickle_file = '09_08_1_detections.p'

    # label_dir = '09_08_1_detections/labels/'

    # final_dir = 'final_labels_09_08_1'

    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    label_list = os.listdir(label_dir)

    img_width = 3024
    img_height = 1975

    empty_0 = os.path.join('empty_conv', 'empty_0.jpg')
    empty_1 = os.path.join('empty_conv', 'empty_1.jpg')

    # img_files = pickle.load(open(pickle_file, 'rb'))

    img_tuples = image_tuple_list_generator(img_dir, empty_0, empty_1, image_files=None,
                                            left_image_index=left_image_index)

    previous_index = ""
    previous_labels = []

    current_index = ""
    filename = ''

    # Final dictionary saved to pickle file with information like unique labels in left image and right image,
    # detection index with reference to original yolo detection and image files
    unique_objects_dict = ""

    for img_tuple in tqdm(img_tuples):

        current_index = img_tuple[0].split('.')[0].split('_')[1]
        label_dict = {}
        if 'empty' in img_tuple[0] or img_tuple[0].split('.')[0] + '.txt' not in label_list:
            label_dict[0] = None
        else:
            label_dict[0] = open(os.path.join(label_dir, img_tuple[0].split('.')[0] + '.txt')).readlines()
            current_index = img_tuple[0].split('.')[0].split('_')[1]
            filename = img_tuple[0].split('.')[0].split('_')[1] + '_' + img_tuple[0].split('.')[0].split('_')[2] + '_' + \
                       img_tuple[0].split('.')[0].split('_')[3] + '.p'

        if 'empty' in img_tuple[1] or img_tuple[1].split('.')[0] + '.txt' not in label_list:
            label_dict[1] = None

        else:
            label_dict[1] = open(os.path.join(label_dir, img_tuple[1].split('.')[0] + '.txt')).readlines()
            current_index = img_tuple[1].split('.')[0].split('_')[1]
            filename = img_tuple[1].split('.')[0].split('_')[1] + '_' + img_tuple[1].split('.')[0].split('_')[2] + '_' + \
                       img_tuple[1].split('.')[0].split('_')[3] + '.p'

        # Remove redundant objects from left and right images
        unique_objects = unique_objects_from_pair(img_tuple, label_dict[0], label_dict[1], img_height, img_width)

        # record['unique_objects'] = unique_objects

        unique_objects_dict = {'left': unique_objects[0], 'right': unique_objects[1]}

        # If objects from previous images are already counted, remove those repeated detections
        if isinstance(previous_index, int):
            if int(previous_index) == int(current_index) - 1:
                retained_left_labels, retained_right_labels = retained_labels(previous_labels[0], previous_labels[1],
                                                                              img_height, img_width)
                # record['retained_left_labels'] = retained_left_labels
                # record['retained_right_labels'] = retained_right_labels

                unique_labels_left = unique_labels_after_retained_labels_filtering(retained_left_labels,
                                                                                   unique_objects[0], img_height,
                                                                                   img_width)

                unique_labels_right = unique_labels_after_retained_labels_filtering(retained_right_labels,
                                                                                    unique_objects[1], img_height,
                                                                                    img_width)
                # record['unique_labels_left'] = unique_labels_left
                # record['unique_labels_right'] = unique_labels_right
                # record['previous_idx'] = previous_index
                # record['current_idx'] = current_index

                unique_objects_dict = {'left': unique_labels_left, 'right': unique_labels_right}

        if isinstance(unique_objects_dict, str):
            unique_objects_dict = {'left': unique_objects[0], 'right': unique_objects[1]}

        previous_index = int(current_index)
        previous_labels = unique_objects_dict['left'], unique_objects_dict['right']

        unique_objects_dict['left_file'] = img_tuple[0]
        unique_objects_dict['right_file'] = img_tuple[1]
        try:
            if unique_objects_dict['left']:
                unique_objects_dict['left_index'] = [label_dict[0].index(x + '\n') for x in unique_objects_dict['left']]
        except Exception as e:
            print('failed for original label :{}, unique objs :{} with exception : {}'.format(label_dict[0],
                                                                                              unique_objects_dict[
                                                                                                  'left'], e))
            print(img_tuple[0])
            # print(json.dumps(record, indent=2))

        try:
            if unique_objects_dict['right']:
                unique_objects_dict['right_index'] = [label_dict[1].index(x + '\n') for x in
                                                      unique_objects_dict['right']]
        except Exception as e:
            print('failed for original label :{}, unique objs :{} with exception : {}'.format(label_dict[1],
                                                                                              unique_objects_dict[
                                                                                                  'right'], e))
        if len(unique_objects_dict['left']) or len(unique_objects_dict['right']):
            with open(os.path.join(final_dir, filename), 'wb') as fp:
                pickle.dump(unique_objects_dict, fp)


def test1():
    labels = os.listdir('labels')
    for _label in labels:
        label = open(os.path.join('labels', _label), 'r').readlines()
        width = 3024
        height = 1975

        x, y = retained_labels(label, label, height, width)
        print('label file : {}\nvalue of x : {}\nlen of x : {}'.format(_label, x, len(x)))
        print('\n')


def test2():
    # pickle_file = '09_08_2_detections.p'
    # label_dir = '09_08_2_detections/labels/'

    # label_dir = 'labels/'
    image_dir = '/home/vikrant/Desktop/09_10_22/undistorted_test/'

    label_dir = '/home/vikrant/Desktop/09_10_22/yolo_output/exp/labels/'
    # image_dir = '/home/vikrant/Desktop/python_pipelines/multiprocess_resizing/09_08_3_undistorted/'

    final_dir = '/home/vikrant/Desktop/09_10_22/vertical_tracking_test/'
    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    final_img_dir = '/home/vikrant/Desktop/09_10_22/vertical_tracking_test_images/'
    if not os.path.isdir(final_img_dir):
        os.mkdir(final_img_dir)

    # print(label_dir, image_dir, final_dir, final_img_dir)
    label_list = os.listdir(label_dir)

    img_width = 3024
    img_height = 1975

    empty_0 = os.path.join('empty_conv', 'empty_0.jpg')
    empty_1 = os.path.join('empty_conv', 'empty_1.jpg')

    # img_files = pickle.load(open(pickle_file, 'rb'))

    img_tuples = image_tuple_list_generator(image_dir, empty_0, empty_1, image_files=None, left_image_index= '0')

    previous_index = ""
    previous_labels = []

    current_index = ""
    filename = ''
    # print(img_tuples)
    for img_tuple in tqdm(img_tuples):

        # print(img_tuple)
        # print(img_tuple[0].split('.')[0] + '.txt' in label_list)
        label_dict = {}
        if 'empty' in img_tuple[0] or img_tuple[0].split('.')[0] + '.txt' not in label_list:
            label_dict[0] = None
        else:
            label_dict[0] = open(os.path.join(label_dir, img_tuple[0].split('.')[0] + '.txt')).readlines()
            current_index = img_tuple[0].split('.')[0].split('_')[1]
            filename = img_tuple[0].split('.')[0].split('_')[1] + '_' + img_tuple[0].split('.')[0].split('_')[2] + '_' + \
                       img_tuple[0].split('.')[0].split('_')[3] + '.p'

        if 'empty' in img_tuple[1] or img_tuple[1].split('.')[0] + '.txt' not in label_list:
            label_dict[1] = None

        else:
            label_dict[1] = open(os.path.join(label_dir, img_tuple[1].split('.')[0] + '.txt')).readlines()
            current_index = img_tuple[1].split('.')[0].split('_')[1]
            # print(img_tuple[0])
            filename = img_tuple[1].split('.')[0].split('_')[1] + '_' + img_tuple[1].split('.')[0].split('_')[2] + '_' + \
                       img_tuple[1].split('.')[0].split('_')[3] + '.p'

        # print('current_index : {}\nprevious_index : {} \nprevious_labels :{} \n'.format(current_index, previous_index,
        #                                                                                 previous_labels))
        # print('yahi hoon main')
        # print(current_index)

        unique_objects = unique_objects_from_pair(img_tuple, label_dict[0], label_dict[1], img_height, img_width)
        unique_objects_dict = ""
        # print('Before vertical correction, len of tmp1 and tmp2 : {} and {}'.format(len(unique_objects[0]),
        #                                                                             len(unique_objects[1])))

        img1 = cv2.imread(os.path.join(image_dir, img_tuple[0]))
        img2 = cv2.imread(os.path.join(image_dir, img_tuple[1]))
        if isinstance(previous_index, int):
            if int(previous_index) == int(current_index) - 1:
                retained_left_labels, retained_right_labels = retained_labels(previous_labels[0], previous_labels[1],
                                                                              img_height, img_width)
                # print('previous labels left : {}'.format(json.dumps(previous_labels[0], indent=2)))
                # print('previous labels right : {}'.format(previous_labels[1]))

                # plot previous labels 0
                # plot retained labels
                if previous_labels[0] is not None:
                    label_3 = [x.strip() for x in previous_labels[0]]
                    for ele in label_3:
                        cls = ele.split(' ')[0]
                        xc = int(float(ele.split(' ')[1]) * img_width)
                        yc = int(float(ele.split(' ')[2]) * img_height)
                        w = int(float(ele.split(' ')[3]) * img_width)
                        h = int(float(ele.split(' ')[4]) * img_height)
                
                        x1 = int(round(xc - w / 2))
                        y1 = int(round(yc - h / 2))
                        x2 = int(round(xc + w / 2))
                        y2 = int(round(yc + h / 2))
                        cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 5)
                
                if retained_left_labels is not None:
                    label_3 = [x.strip() for x in retained_left_labels]
                    for ele in label_3:
                        cls = ele.split(' ')[0]
                        xc = int(float(ele.split(' ')[1]) * img_width)
                        yc = int(float(ele.split(' ')[2]) * img_height)
                        w = int(float(ele.split(' ')[3]) * img_width)
                        h = int(float(ele.split(' ')[4]) * img_height)
                
                        x1 = int(round(xc - w / 2))
                        y1 = int(round(yc - h / 2))
                        x2 = int(round(xc + w / 2))
                        y2 = int(round(yc + h / 2))
                        cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 5)

                # if unique_objects[0] is not None:
                #     label_3 = [x.strip() for x in unique_objects[0]]
                #     for ele in label_3:
                #         cls = ele.split(' ')[0]
                #         xc = int(float(ele.split(' ')[1]) * img_width)
                #         yc = int(float(ele.split(' ')[2]) * img_height)
                #         w = int(float(ele.split(' ')[3]) * img_width)
                #         h = int(float(ele.split(' ')[4]) * img_height)
                
                #         x1 = int(round(xc - w / 2))
                #         y1 = int(round(yc - h / 2))
                #         x2 = int(round(xc + w / 2))
                #         y2 = int(round(yc + h / 2))
                #         cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 0, 255), 5)


                if previous_labels[1] is not None:
                    label_3 = [x.strip() for x in previous_labels[1]]
                    for ele in label_3:
                        cls = ele.split(' ')[0]
                        xc = int(float(ele.split(' ')[1]) * img_width)
                        yc = int(float(ele.split(' ')[2]) * img_height)
                        w = int(float(ele.split(' ')[3]) * img_width)
                        h = int(float(ele.split(' ')[4]) * img_height)
                
                        x1 = int(round(xc - w / 2))
                        y1 = int(round(yc - h / 2))
                        x2 = int(round(xc + w / 2))
                        y2 = int(round(yc + h / 2))
                        # if x1 > img_width * 0.7:
                            # print(([x1,y1,x2,y2, xc, yc, w, h]))
                        cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 5)
                # print('retained right labels....')
                if retained_right_labels is not None:
                    label_3 = [x.strip() for x in retained_right_labels]
                    for ele in label_3:
                        cls = ele.split(' ')[0]
                        xc = int(float(ele.split(' ')[1]) * img_width)
                        yc = int(float(ele.split(' ')[2]) * img_height)
                        w = int(float(ele.split(' ')[3]) * img_width)
                        h = int(float(ele.split(' ')[4]) * img_height)
                
                        x1 = int(round(xc - w / 2))
                        y1 = int(round(yc - h / 2))
                        x2 = int(round(xc + w / 2))
                        y2 = int(round(yc + h / 2))
                        # if x1 > img_width * 0.7:
                            # print(([x1,y1,x2,y2, w, h]))
                        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 5)

                # if unique_objects[1] is not None:
                #     label_3 = [x.strip() for x in unique_objects[1]]
                #     for ele in label_3:
                #         cls = ele.split(' ')[0]
                #         xc = int(float(ele.split(' ')[1]) * img_width)
                #         yc = int(float(ele.split(' ')[2]) * img_height)
                #         w = int(float(ele.split(' ')[3]) * img_width)
                #         h = int(float(ele.split(' ')[4]) * img_height)
                
                #         x1 = int(round(xc - w / 2))
                #         y1 = int(round(yc - h / 2))
                #         x2 = int(round(xc + w / 2))
                #         y2 = int(round(yc + h / 2))
                #         cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 5)


                # print('len of previous label :{} and len of retained label :{}'.format(len(previous_labels[0]),
                #                                                                        len(retained_left_labels)))

                unique_labels_left = unique_labels_after_retained_labels_filtering(retained_left_labels,
                                                                                   unique_objects[0], img_height,
                                                                                   img_width)

                unique_labels_right = unique_labels_after_retained_labels_filtering(retained_right_labels,
                                                                                    unique_objects[1], img_height,
                                                                                    img_width)
                unique_objects_dict = {'left': unique_labels_left, 'right': unique_labels_right}
                # print('In here!')
                # print('After vertical correction, len of tmp1 and tmp2 : {} and {}'.format(len(unique_labels_left),
                #                                                                            len(unique_labels_right)))

        if isinstance(unique_objects_dict, str):
            unique_objects_dict = {'left': unique_objects[0], 'right': unique_objects[1]}
        # print(current_index)
        # print(previous_index)
        previous_index = int(current_index)
        previous_labels = unique_objects_dict['left'], unique_objects_dict['right']

        if unique_objects_dict['left'] is not None:
            label_3 = [x.strip() for x in unique_objects_dict['left']]
            for ele in label_3:
                cls = ele.split(' ')[0]
                xc = int(float(ele.split(' ')[1]) * img_width)
                yc = int(float(ele.split(' ')[2]) * img_height)
                w = int(float(ele.split(' ')[3]) * img_width)
                h = int(float(ele.split(' ')[4]) * img_height)

                x1 = int(round(xc - w / 2))
                y1 = int(round(yc - h / 2))
                x2 = int(round(xc + w / 2))
                y2 = int(round(yc + h / 2))
                cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 0, 255), 5)

            cv2.putText(img1, str(len(unique_objects_dict['left'])), (100, 100), 0, 4, [0, 255, 255],
                        thickness=3,
                        lineType=cv2.LINE_AA)
            cv2.putText(img1, img_tuple[0].split('.')[0].split('_')[2] + '_' + \
                        img_tuple[0].split('.')[0].split('_')[3], (100, 1800), 0, 3, [0, 255, 255],
                        thickness=3,
                        lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(final_img_dir, img_tuple[0]), img1)

        if unique_objects_dict['right'] is not None:
            label_3 = [x.strip() for x in unique_objects_dict['right']]
            for ele in label_3:
                cls = ele.split(' ')[0]
                xc = int(float(ele.split(' ')[1]) * img_width)
                yc = int(float(ele.split(' ')[2]) * img_height)
                w = int(float(ele.split(' ')[3]) * img_width)
                h = int(float(ele.split(' ')[4]) * img_height)

                x1 = int(round(xc - w / 2))
                y1 = int(round(yc - h / 2))
                x2 = int(round(xc + w / 2))
                y2 = int(round(yc + h / 2))
                cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 5)

            cv2.putText(img2, str(len(unique_objects_dict['right'])), (100, 100), 0, 4, [0, 255, 255],
                        thickness=3,
                        lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(final_img_dir, img_tuple[1]), img2)

        unique_objects_dict['left_file'] = img_tuple[0]
        unique_objects_dict['right_file'] = img_tuple[1]

        if unique_objects_dict['left']:
            unique_objects_dict['left_index'] = [label_dict[0].index(x + '\n') for x in unique_objects_dict['left']]

        if unique_objects_dict['right']:
            unique_objects_dict['right_index'] = [label_dict[1].index(x + '\n') for x in unique_objects_dict['right']]

        with open(os.path.join(final_dir, filename), 'wb') as fp:
            pickle.dump(unique_objects_dict, fp)


def main():
    parser = argparse.ArgumentParser(description='unique pickle generator from images and yolo labels')

    parser.add_argument('--label_dir', type=str, default="labels", help='path of the yolo label directory')
    parser.add_argument('--image_dir', type=str, default="images", help='path of the image directory')
    parser.add_argument('--final_dir', type=str, default="images", help='path of the image directory')
    parser.add_argument('--left_image_index', type=str, default='1', help='index of the left image')

    args = parser.parse_args()
    label_dir = args.label_dir
    image_dir = args.image_dir
    final_dir = args.final_dir
    left_index = args.left_image_index

    unique_objects_from_detections(image_dir, label_dir, final_dir, left_index)


if __name__ == '__main__':
    main()
