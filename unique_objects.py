import os
import time
import cv2
import numpy as np
from tqdm import tqdm
import json
from box_iou import return_box_iou_batch


def unique_labels_after_retained_labels_filtering(retained_label, tmp_label, img_height, img_width):
    # convert labels to normalised labels

    retained_label_norm = []
    tmp_label_norm = []

    # print('retained label :{} and tmp label : {}'.format(retained_label, tmp_label))

    if retained_label is not None:
        label_1 = [x.strip() for x in retained_label]
        for ele in label_1:
            xc = int(float(ele.split(' ')[1]) * img_width)
            yc = int(float(ele.split(' ')[2]) * img_height)
            w = int(float(ele.split(' ')[3]) * img_width)
            h = int(float(ele.split(' ')[4]) * img_height)

            x1 = int(round(xc - w / 2))
            y1 = int(round(yc - h / 2))
            x2 = int(round(xc + w / 2))
            y2 = int(round(yc + h / 2))
            retained_label_norm.append((x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height))

    if tmp_label is not None:
        label_1 = [x.strip() for x in tmp_label]
        for ele in label_1:
            xc = int(float(ele.split(' ')[1]) * img_width)
            yc = int(float(ele.split(' ')[2]) * img_height)
            w = int(float(ele.split(' ')[3]) * img_width)
            h = int(float(ele.split(' ')[4]) * img_height)

            x1 = int(round(xc - w / 2))
            y1 = int(round(yc - h / 2))
            x2 = int(round(xc + w / 2))
            y2 = int(round(yc + h / 2))
            tmp_label_norm.append((x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height))

    iou_status = return_box_iou_batch(retained_label_norm, tmp_label_norm, threshold=0.4)
    idx_removed = iou_status[1]
    # print('idx_removed after vertical correction :{}'.format(idx_removed))

    return list(np.delete(np.array(tmp_label), tuple(idx_removed)))


def retained_labels(tmp_1, tmp_2, img_height, img_width):
    # define distance in pixel an object centre moves between consecutive camera trigger
    #changed on 08_10_2022 
    # vertical_drift = 1270
    vertical_drift = 1130

    final_tmp_1 = []
    final_tmp_2 = []

    common_area_pixel = 900
    box_distance_from_top_bottom_allowance = 0

    if tmp_1 is not None:
        label_1 = [x.strip() for x in tmp_1]
        for box_n, ele in enumerate(label_1):
            xc = int(float(ele.split(' ')[1]) * img_width)
            yc = int(float(ele.split(' ')[2]) * img_height) + vertical_drift
            h = int(float(ele.split(' ')[4]) * img_height)
            # Excluded top and bottom margin objects in images
            if (yc - h / 2) > box_distance_from_top_bottom_allowance and (img_height - (
                    yc + h / 2)) > box_distance_from_top_bottom_allowance and xc < img_width - common_area_pixel / 2:
                # final_tmp_1.append(ele)
                final_tmp_1.append(' '.join(
                    ['0', ele.split(' ')[1], str(float(yc / img_height)), ele.split(' ')[3], ele.split(' ')[4]]))

    if tmp_2 is not None:
        label_1 = [x.strip() for x in tmp_2]
        for box_n, ele in enumerate(label_1):
            xc = int(float(ele.split(' ')[1]) * img_width)
            yc = int(float(ele.split(' ')[2]) * img_height) + vertical_drift
            h = int(float(ele.split(' ')[4]) * img_height)

            # Excluded top and bottom margin objects in images
            if (yc - h / 2) > box_distance_from_top_bottom_allowance and (img_height - (
                    yc + h / 2)) > box_distance_from_top_bottom_allowance and xc < img_width - common_area_pixel / 2:
                # final_tmp_2.append(ele)
                final_tmp_2.append(' '.join(
                    ['0', ele.split(' ')[1], str(float(yc / img_height)), ele.split(' ')[3], ele.split(' ')[4]]))

    return final_tmp_1, final_tmp_2


def image_tuple_list_generator(image_dir, empty_0, empty_1, image_files=None, left_image_index='1'):
    images = sorted(os.listdir(image_dir))

    if image_files:
        images = sorted(image_files)

    img_tuple_list = []

    j = 0
    last_index = len(images) - 1
    for index in tqdm(range(len(images))):

        if index < j:
            continue

        current_image = images[index]
        if index < last_index:
            next_image = images[index + 1]
            # print(current_image)
            # print(next_image)
            delta_t = int(next_image.split('_')[0]) - int(current_image.split('_')[0])
            if delta_t < 10 ** 7:
                if current_image.split('.')[0].split('_')[-1] == left_image_index:
                    img_tuple_list.append((current_image, next_image))
                else:
                    img_tuple_list.append((next_image, current_image))
                j = index + 2

            else:
                if current_image.split('.')[0].split('_')[-1] == left_image_index:
                    img_tuple_list.append((current_image, empty_0))
                else:
                    img_tuple_list.append((empty_1, current_image))
                j = index + 1

        if index == last_index:
            if current_image.split('.')[0].split('_')[-1] == '1':
                img_tuple_list.append((current_image, empty_0))
            else:
                img_tuple_list.append((empty_1, current_image))

    # print(img_tuple_list)
    return img_tuple_list


def unique_objects_from_pair(img_tuple, label1, label2, img_height, img_width):
    # Margins to leave on top and bottom for unique objects identification

    image = img_tuple[0]
    # img = cv2.imread('images_without_bb/' + image)
    final_dir = 'com_img/'
    cross_image_pixel_distance = 1570
    common_area_pixel = 900
    box_distance_from_side_allowance = 50
    box_distance_from_top_bottom_allowance = 120

    # Above values needs to be adjusted as per the conveyor speed and camera calibration
    # images from cam 1 are shifted by 50 pixel vertically in reference to images from cam 0
    lateral_shift = 50

    # image1, image2 ..... label1, label2 (inputs)
    # arguments ==> label1 , label 0 if non-empty

    # list of objects to keep in images
    tmp_1 = []
    tmp_2 = []

    # objects in common com and objects in img1 remapped to img2 ==> com_2_remapped
    com_1 = []
    com_2 = []
    com_2_remapped = []

    com_norm_1 = []
    com_norm_remapped_2 = []

    com_1_idx = []
    com_2_idx = []

    if label1 is not None:
        label_1 = [x.strip() for x in label1]
        for box_n, ele in enumerate(label_1):
            cls = ele.split(' ')[0]
            xc = int(float(ele.split(' ')[1]) * img_width)
            yc = int(float(ele.split(' ')[2]) * img_height)
            w = int(float(ele.split(' ')[3]) * img_width)
            h = int(float(ele.split(' ')[4]) * img_height)

            # Excluded top and bottom margin objects in images
            if (yc - h / 2) > box_distance_from_top_bottom_allowance and (img_height - (
                    yc + h / 2)) > box_distance_from_top_bottom_allowance and xc < img_width - common_area_pixel / 2:
                tmp_1.append(ele)

            # Common area object list

            if (yc - h / 2) > box_distance_from_top_bottom_allowance and (img_height - (
                    yc + h / 2)) > box_distance_from_top_bottom_allowance and \
                    img_width - cross_image_pixel_distance < xc < img_width - common_area_pixel / 2 and (img_width - (
                    xc + w / 2)) > 20 and (img_width - (xc + w / 2)) > box_distance_from_side_allowance:
                com_1.append(' '.join(
                    [cls, str(xc), str(yc), str(w),
                     str(h), str(w * h / 100)]))
                x1 = int(round(xc - w / 2))
                y1 = int(round(yc - h / 2))
                x2 = int(round(xc + w / 2))
                y2 = int(round(yc + h / 2))
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 5)
                com_norm_1.append((x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height))
                # cv2.putText(img, 'box_{}'.format(str(len(com_norm_1))), (x1 + 10, y1 + 40), 0, 1, [0, 255, 255],
                #             thickness=3,
                #             lineType=cv2.LINE_AA)
                # com_1_idx.append(box_n)
                com_1_idx.append(len(tmp_1) - 1)

    # print('com1 : ', com_1)
    # print(len(com_1))
    # print('com_norm_1 : ', com_norm_1)

    if label2 is not None:
        label_2 = [x.strip() for x in label2]
        for box_n, ele in enumerate(label_2):
            cls = ele.split(' ')[0]
            xc = int(float(ele.split(' ')[1]) * img_width)
            yc = int(float(ele.split(' ')[2]) * img_height)
            w = int(float(ele.split(' ')[3]) * img_width)
            h = int(float(ele.split(' ')[4]) * img_height)

            # Excluded top and bottom margin objects in images
            if (yc - h / 2) > box_distance_from_top_bottom_allowance and (img_height - (
                    yc + h / 2)) > box_distance_from_top_bottom_allowance and xc > common_area_pixel / 2:
                tmp_2.append(ele)

            # Common area object list
            if (yc - h / 2) > box_distance_from_top_bottom_allowance and (img_height - (
                    yc + h / 2)) > box_distance_from_top_bottom_allowance and cross_image_pixel_distance > xc > common_area_pixel / 2 and (
                    xc - w / 2) > box_distance_from_side_allowance:
                com_2.append(' '.join(
                    [cls, str(xc), str(yc), str(w),
                     str(h)]))
                com_2_remapped.append(' '.join(
                    [cls, str(xc - cross_image_pixel_distance + img_width), str(yc - lateral_shift), str(w),
                     str(h), str(w * h / 100)]))
                x1 = int(round(xc - cross_image_pixel_distance + img_width - w / 2))
                y1 = int(round(yc - lateral_shift - h / 2))
                x2 = int(round(xc - cross_image_pixel_distance + img_width + w / 2))
                y2 = int(round(yc - lateral_shift + h / 2))
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
                com_norm_remapped_2.append((x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height))
                # cv2.putText(img, 'box_{}'.format(str(len(com_norm_remapped_2))), (x1 + 10, y1 + 40), 0, 1,
                #             [255, 255, 0], thickness=3,
                #             lineType=cv2.LINE_AA)
                # com_2_idx.append(box_n)
                com_2_idx.append(len(tmp_2) - 1)

    # print('com2_remapped : ', com_2_remapped)
    # print('length of com2 : {}'.format(len(com_2_remapped)))
    # print('com2_remapped_norm : ', com_norm_remapped_2)
    # print('com_1_idx : {}'.format(com_1_idx))
    # print('com_2_idx : {}'.format(com_2_idx))
    # print('length of com2index : {}'.format(len(com_2_idx)))
    # print('tmp1 : {} '.format(tmp_1))
    # print('len of tmp1 : {}'.format(len(tmp_1)))
    # print('tmp2 : {} '.format(tmp_2))
    # print('len of tmp2 : {}'.format(len(tmp_2)))

    iou_status = return_box_iou_batch(com_norm_1, com_norm_remapped_2)
    # print(iou_status[0])

    idx_removed_from_com_2 = iou_status[1]
    # print(idx_removed_from_com_2)
    # if len(idx_removed_from_com_2):
    #     print('actual labels removed : {}'.format([com_2_idx[x] for x in idx_removed_from_com_2]))

    final_tmp2 = list(np.delete(np.array(tmp_2), tuple([com_2_idx[x] for x in idx_removed_from_com_2])))
    # print('initial and final length of tmp2 : {} and {}'.format(len(tmp_2), len(final_tmp2)))
    # if len(iou_status) != 0:
    #     for k, iou in enumerate(iou_status):
    #         cv2.putText(img, 'box {} : {} '.format(k + 1, iou), (50, 50 + k * 70), 0, 2, [255, 255, 0], thickness=3,
    #                     lineType=cv2.LINE_AA)
    # cv2.imwrite(final_dir + image, img)
    # print('\n')

    # unique objects from left camera and unique objects from right camera
    return tmp_1, final_tmp2


def main():
    empty_0 = os.path.join('empty_conv', 'empty_0.jpg')
    empty_1 = os.path.join('empty_conv', 'empty_1.jpg')

    image_dir = 'image_with_bb'
    # final_dir = 'final_dir_with_bb'
    label_dir = 'labels'

    img0 = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[0]))
    img_width = img0.shape[1]
    img_height = img0.shape[0]

    # print(img_width, img_height)
    img_tuple_list = image_tuple_list_generator(image_dir, empty_0, empty_1)
    label_list = os.listdir(label_dir)

    # unique objects from tuple
    for img_tuple in img_tuple_list[0:200]:
        print(img_tuple)
        label_dict = {}
        if 'empty' in img_tuple[0] or img_tuple[0].split('.')[0] + '.txt' not in label_list:
            label_dict[0] = None
        else:
            label_dict[0] = open(os.path.join(label_dir, img_tuple[0].split('.')[0] + '.txt')).readlines()

        if 'empty' in img_tuple[1] or img_tuple[1].split('.')[0] + '.txt' not in label_list:
            label_dict[1] = None
        else:
            label_dict[1] = open(os.path.join(label_dir, img_tuple[1].split('.')[0] + '.txt')).readlines()

        unique_object = unique_objects_from_pair(img_tuple, label_dict[0], label_dict[1], img_height, img_width)
        # print(unique_object)


if __name__ == '__main__':
    main()
