# Bash script to automatically undistort, run yolo, find unique objects, and then on the
# crop outs, run bottle classification, color classification and then food/non_food classification

# Edit the {parent directory} and {Image_dir} to start the program

PARENT_DIR="/media/vinglabs/vinglabs_hdd1/08_09_2/"
IMAGE_DIR="images/"

CURRENT_DIR=$PWD
UNDISTORTED_OUTPUT="undistorted/"
YOLO_OUTPUT="yolo_output/"
YOLO_LABELS="yolo_output/exp/labels/"
YOLO_CROPOUTS="yolo_crops"

YOLO_TO_PICKLE_UNIQUE="yolo_output_pickle"
CONVERTED="_converted"
LEFT_IMAGE_INDEX="1"
BOTTLE_CROP_DIR="bottle_crops"
MERGED_IMAGES_DIR="merged_images_for_tracking"

YOLO_SCRIPT_DIR="/media/vinglabs/dev/vikrant_pipeline/yolov5-master/"
UNDISTORTED_SCRIPT_DIR="/media/vinglabs/dev/vikrant_pipeline/undistortion/"
YOLO_CROP_DIR="/media/vinglabs/dev/vikrant_pipeline/yolo_crop/"

BOTTLE_NO_BOTTLE_SCRIPT_DIR="/media/vinglabs/dev/vikrant_pipeline/bottle_no_bottle_cans_classification/"
COLOR_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/color_classification/"
FOOD_NON_FOOD_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/food_classification/"

PARAMETER_JSON="classification-barebones/helpers/parameters.json"
DETECT_EXEC_DIR="classification-barebones/helpers/exec_files"


BOTTLE_OUTPUT="bottle_output"
#ONLY_BOTTLE_CROPOUTS="bottle_output_split_1429/bottle/"
COLOR_OUTPUT="color_output"
FOOD_NON_FOOD_OUTPUT_1="food_output_1431"
FOOD_NON_FOOD_OUTPUT_2="food_output_1424"


FOOD_NON_FOOD_WEIGHTS_1="best1431.pt"
FOOD_NON_FOOD_WEIGHTS_2="best1424.pt"


cd $UNDISTORTED_SCRIPT_DIR
#/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python undistort_from_pickle.py --orig_dir $PARENT_DIR$IMAGE_DIR --final_dir $PARENT_DIR$UNDISTORTED_OUTPUT

cd $YOLO_SCRIPT_DIR

#/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python detect.py --weights best.pt --data dataset.yaml --source $PARENT_DIR$UNDISTORTED_OUTPUT  --conf-thres 0.5 --iou-thres 0.5 --save-txt --project $PARENT_DIR$YOLO_OUTPUT --nosave

cd $CURRENT_DIR

#/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python test_vertical_tracking.py --image_dir $PARENT_DIR$UNDISTORTED_OUTPUT --label_dir $PARENT_DIR$YOLO_LABELS --final_dir $PARENT_DIR$YOLO_TO_PICKLE_UNIQUE --left_image_index $LEFT_IMAGE_INDEX

#/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python left_and_right_label_generator.py --label_dir $PARENT_DIR$YOLO_TO_PICKLE_UNIQUE



cd $YOLO_CROP_DIR

#/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python yolo_crop.py --image_dir $PARENT_DIR$UNDISTORTED_OUTPUT --label_dir $PARENT_DIR$YOLO_TO_PICKLE_UNIQUE$CONVERTED --classified_dir $PARENT_DIR$YOLO_CROPOUTS

# THIS CODE SNIPPET WILL DRAW BOXES ON UNDISTORTED IMAGES AS WELL AS CREATE PAIR TO VISUALISE TRACKING
cd $CURRENT_DIR
# Draw boxes
#/usr/bin/python draw_box.py --label_folder $PARENT_DIR$YOLO_TO_PICKLE_UNIQUE$CONVERTED  --raw_images_folder $PARENT_DIR$UNDISTORTED_OUTPUT  --save_images_folder $PARENT_DIR$UNDISTORTED_OUTPUT
# Concatenate images
#/usr/bin/python concatenate_image_tuple.py --image_dir $PARENT_DIR$UNDISTORTED_OUTPUT --final_dir $PARENT_DIR$MERGED_IMAGES_DIR --left_image_index $LEFT_IMAGE_INDEX


/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python edit_params.py --source $PARENT_DIR$YOLO_CROPOUTS --output $PARENT_DIR$BOTTLE_OUTPUT --json $BOTTLE_NO_BOTTLE_SCRIPT_DIR$PARAMETER_JSON
cd $BOTTLE_NO_BOTTLE_SCRIPT_DIR$DETECT_EXEC_DIR
/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python detect_exec.py
cd $CURRENT_DIR
/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python split_bottle_cans_folder_from_labels.py --label_dir $PARENT_DIR$BOTTLE_OUTPUT --image_dir $PARENT_DIR$YOLO_CROPOUTS


#Will run color and Food/Non food only on the bottle output images
#cd $CURRENT_DIR
#/usr/bin/python extract_bottle_cropouts.py --original_crop_dir $PARENT_DIR$YOLO_CROPOUTS  --bottle_image_dir  $PARENT_DIR$BOTTLE_OUTPUT"_split/bottle/" --final_dir $PARENT_DIR$BOTTLE_CROP_DIR

#cd $CURRENT_DIR
#/usr/bin/python edit_params.py --source $PARENT_DIR$BOTTLE_CROP_DIR --output $PARENT_DIR$COLOR_OUTPUT --json $COLOR_SCRIPT_DIR$PARAMETER_JSON
#cd $COLOR_SCRIPT_DIR$DETECT_EXEC_DIR
#/usr/bin/python detect_exec.py

# Running the first food_non_food model
#cd $CURRENT_DIR
#/usr/bin/python edit_params.py --source $PARENT_DIR$BOTTLE_CROP_DIR --output $PARENT_DIR$FOOD_NON_FOOD_OUTPUT_1 --json $FOOD_NON_FOOD_SCRIPT_DIR$PARAMETER_JSON --weights $FOOD_NON_FOOD_WEIGHTS_1
#cd $FOOD_NON_FOOD_SCRIPT_DIR$DETECT_EXEC_DIR
#/usr/bin/python detect_exec.py

# Running the second food non food model
#cd $CURRENT_DIR
#/usr/bin/python edit_params.py --source $PARENT_DIR$BOTTLE_CROP_DIR --output $PARENT_DIR$FOOD_NON_FOOD_OUTPUT_2 --json $FOOD_NON_FOOD_SCRIPT_DIR$PARAMETER_JSON --weights $FOOD_NON_FOOD_WEIGHTS_2
#cd $FOOD_NON_FOOD_SCRIPT_DIR$DETECT_EXEC_DIR
#/usr/bin/python detect_exec.py

#/usr/bin/python edit_params.py --source $PARENT_DIR$ONLY_BOTTLE_CROPOUTS --output $PARENT_DIR$FOOD_NON_FOOD_OUTPUT --json $FOOD_NON_FOOD_SCRIPT_DIR$PARAMETER_JSON



cd $CURRENT_DIR

#/usr/bin/python split_color_folder.py --image_and_label_dir $PARENT_DIR$COLOR_OUTPUT
#/usr/bin/python split_food_folder.py --image_and_label_dir $PARENT_DIR$FOOD_NON_FOOD_OUTPUT_1
#/usr/bin/python split_food_folder.py --image_and_label_dir $PARENT_DIR$FOOD_NON_FOOD_OUTPUT_2


# Create folders with original crop-outs for YOLO output
#/usr/bin/python replicate_with_original_crops.py --orig_dir $PARENT_DIR$FOOD_NON_FOOD_OUTPUT_1"_split" --crops_dir $PARENT_DIR$YOLO_CROPOUTS
#/usr/bin/python replicate_with_original_crops.py --orig_dir $PARENT_DIR$FOOD_NON_FOOD_OUTPUT_2"_split" --crops_dir $PARENT_DIR$YOLO_CROPOUTS


# YOLO neck ring
#cd $YOLO_SCRIPT_DIR
#/usr/bin/python detect.py --weights ring_weights/best_ring_5.pt  --source $PARENT_DIR"food_output_1431_split_original/non_food/" --imgsz 480 --project $PARENT_DIR"food_output_1431_split_original/non_food_ring_output" --iou-thres 0.5 --conf-thres 0.66 --data dataset.yaml --save-txt
#/usr/bin/python detect.py --weights ring_weights/best_ring_5.pt  --source $PARENT_DIR"food_output_1424_split_original/non_food/" --imgsz 480 --project $PARENT_DIR"food_output_1424_split_original/non_food_ring_output" --iou-thres 0.5 --conf-thres 0.66 --data dataset.yaml --save-txt

#cd $CURRENT_DIR


# YOLO bottle cap
#cd $YOLO_SCRIPT_DIR
#/usr/bin/python detect.py --weights cap_weights/caps_only_3.pt  --source $PARENT_DIR"food_output_1431_split_original/food" --imgsz 480 --project $PARENT_DIR"food_output_1431_split_original/food_cap_output" --iou-thres 0.5 --conf-thres 0.5 --data dataset.yaml --save-txt
#/usr/bin/python detect.py --weights cap_weights/caps_only_3.pt  --source $PARENT_DIR"food_output_1424_split_original/food" --imgsz 480 --project $PARENT_DIR"food_output_1424_split_original/food_cap_output" --iou-thres 0.5 --conf-thres 0.5 --data dataset.yaml --save-txt

#cd $CURRENT_DIR
#cp food_non_food_after_cap_and_label_correction.py $PARENT_DIR

#cd $PARENT_DIR
# Updated food/non food 1424 and 1431
#cp -r food_output_1424_split food_output_1424_cap_and_label_correction_split
#cp -r food_output_1431_split food_output_1431_cap_and_label_correction_split

#/usr/bin/python food_non_food_after_cap_and_label_correction.py
#rm food_non_food_after_cap_and_label_correction.py
#rm -r bottle_output color_output food_output_1424 food_output_1431 bottle_crops undistorted yolo_crops yolo_output yolo_output_pickle yolo_output_pickle_converted

#cd $CURRENT_DIR
