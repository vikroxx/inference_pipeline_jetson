import json
import os
import pickle
from tqdm import tqdm
import argparse


def generate_csv():
    label_dir = '/home/vikrant/Desktop/python_pipelines/image_tracking_and_merge/17_08/17_08_unique_pickle/'
    bottle_dir = '/home/vikrant/Desktop/python_pipelines/image_tracking_and_merge/17_08/classification/bottle_output'
    food_dir = '/home/vikrant/Desktop/python_pipelines/image_tracking_and_merge/17_08/classification/food_output'
    color_dir = '/home/vikrant/Desktop/python_pipelines/image_tracking_and_merge/17_08/classification/color_output'

    # bottle_labels = os.listdir(bottle_dir)
    # food_labels = os.listdir(food_dir)
    # color_labels = os.listdir(color_dir)

    final_list_dict = []

    bottle_output_dict = {0: 'Bottle', 1: 'Non Bottle'}
    color_output_dict = {0: 'Brown and Black', 1: 'Clear/Light Blue', 2: 'Dark Blue', 3: 'Green', 4: 'Opaque',
                         5: 'Other',
                         6: 'Sleeve-Clear'}
    food_output_dict = {0: 'Food', 1: 'Non Food'}

    for label_file in tqdm(sorted(os.listdir(label_dir))):
        label_content = pickle.load(open(os.path.join(label_dir, label_file), 'rb'))

        label_left = label_content['left']
        label_right = label_content['right']

        left_file = label_content['left_file']
        right_file = label_content['right_file']

        # Label left
        if len(label_left):
            label_1 = [x.strip() for x in label_left]
            for idx, ele in enumerate(label_1):
                file_name = left_file.split('.')[0] + '_' + str(
                    int(float(ele.split(' ')[1]) * 10 ** 6)) + "_" + str(
                    int(float(ele.split(' ')[2]) * 10 ** 6)) + "_" + str(idx) + '.p'

                date = left_file.split('_')[2].strip()
                time_ = left_file.split('_')[3].strip().replace('-', ':')
                time_stamp = date + ' ' + time_

                bottle_output = bottle_output_dict[pickle.load(open(os.path.join(bottle_dir, file_name), 'rb'))[
                    file_name.split('.')[0]][0][0]]

                if bottle_output == 'Non Bottle':
                    final_list_dict.append(
                        {'timestamp': time_stamp, 'bottle': bottle_output, 'color': 'NA', 'food': 'NA'})

                else:
                    color_output = color_output_dict[pickle.load(open(os.path.join(color_dir, file_name), 'rb'))[
                        file_name.split('.')[0]][0][0]]

                    if color_output == 'Clear/Light Blue' or color_output == 'Sleeve-Clear':
                        food_output = food_output_dict[pickle.load(open(os.path.join(food_dir, file_name), 'rb'))[
                            file_name.split('.')[0]][0][0]]
                        final_list_dict.append(
                            {'timestamp': time_stamp, 'bottle': bottle_output, 'color': color_output,
                             'food': food_output})
                    else:
                        final_list_dict.append(
                            {'timestamp': time_stamp, 'bottle': bottle_output, 'color': color_output, 'food': 'NA'})

        if len(label_right):
            label_1 = [x.strip() for x in label_right]
            for idx, ele in enumerate(label_1):
                file_name = right_file.split('.')[0] + '_' + str(
                    int(float(ele.split(' ')[1]) * 10 ** 6)) + "_" + str(
                    int(float(ele.split(' ')[2]) * 10 ** 6)) + "_" + str(idx) + '.p'

                date = right_file.split('_')[2].strip()
                time_ = right_file.split('_')[3].strip().replace('-', ':')
                time_stamp = date + ' ' + time_

                bottle_output = bottle_output_dict[pickle.load(open(os.path.join(bottle_dir, file_name), 'rb'))[
                    file_name.split('.')[0]][0][0]]

                if bottle_output == 'Non Bottle':
                    final_list_dict.append(
                        {'timestamp': time_stamp, 'bottle': bottle_output, 'color': 'NA', 'food': 'NA'})

                else:
                    color_output = color_output_dict[pickle.load(open(os.path.join(color_dir, file_name), 'rb'))[
                        file_name.split('.')[0]][0][0]]

                    if color_output == 'Clear/Light Blue' or color_output == 'Sleeve-Clear':
                        food_output = food_output_dict[pickle.load(open(os.path.join(food_dir, file_name), 'rb'))[
                            file_name.split('.')[0]][0][0]]
                        final_list_dict.append(
                            {'timestamp': time_stamp, 'bottle': bottle_output, 'color': color_output,
                             'food': food_output})
                    else:
                        final_list_dict.append(
                            {'timestamp': time_stamp, 'bottle': bottle_output, 'color': color_output, 'food': 'NA'})

    print(json.dumps(final_list_dict, indent=2))
    print(len(final_list_dict))

    file_pickle_name = 'final_pickle/' + '17_08_pickle' + '.p'
    with open(file_pickle_name, 'wb') as fp:
        pickle.dump(final_list_dict, fp)


def yolo_label_generator_from_pickle(label_dir):
    # label_dir = 'final_labels_09_08_3'

    converted_label_dir = label_dir + '_converted'

    if not os.path.isdir(converted_label_dir):
        os.mkdir(converted_label_dir)

    for label_file in tqdm(os.listdir(label_dir)):
        label_content = pickle.load(open(os.path.join(label_dir, label_file), 'rb'))
        label_left = label_content['left']
        label_right = label_content['right']

        left_file = label_content['left_file']
        right_file = label_content['right_file']

        if len(label_left):
            with open(os.path.join(converted_label_dir, left_file.split('.')[0] + '.txt'), 'w') as fp:
                fp.writelines('\n'.join(label_left))

        if len(label_right):
            with open(os.path.join(converted_label_dir, right_file.split('.')[0] + '.txt'), 'w') as fp:
                fp.writelines('\n'.join(label_right))


def main():
    parser = argparse.ArgumentParser(description='yolo generator from pickle unique')
    parser.add_argument('--label_dir', type=str, default="images", help='path of the label pickle directory')

    args = parser.parse_args()
    label_dir = args.label_dir
    yolo_label_generator_from_pickle(label_dir)


if __name__ == '__main__':
    main()
