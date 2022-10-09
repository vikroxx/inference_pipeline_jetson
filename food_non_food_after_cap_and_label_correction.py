import os
import shutil


def main():
    orig_dir = ['food_output_1424_cap_and_label_correction_split', 'food_output_1431_cap_and_label_correction_split']

    prediction_dir = ["food_output_1424_split_original", "food_output_1431_split_original"]

    for i in range(2):
        # Corrected non-food images for dir 1
        labels_corrected = os.listdir(os.path.join(prediction_dir[i], 'non_food_ring_output', 'detection', 'labels'))
        if len(labels_corrected):
            for label in labels_corrected:
                img = label.split('.')[0] + '.jpg'
                shutil.copy(os.path.join(prediction_dir[i], 'non_food_ring_output', 'detection', img),
                            os.path.join(orig_dir[i], 'food', img))
                os.remove(os.path.join(orig_dir[i], 'non_food', img))

        # Corrected food images for dir 1
        labels_corrected = os.listdir(os.path.join(prediction_dir[i], 'food_cap_output', 'detection', 'labels'))
        if len(labels_corrected):
            for label in labels_corrected:
                img = label.split('.')[0] + '.jpg'
                shutil.copy(os.path.join(prediction_dir[i], 'food_cap_output', 'detection', img),
                            os.path.join(orig_dir[i], 'non_food', img))
                os.remove(os.path.join(orig_dir[i], 'food', img))


if __name__ == '__main__':
    main()
