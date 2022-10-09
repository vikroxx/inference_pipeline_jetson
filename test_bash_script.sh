# Bash script to automatically undistort, run yolo, find unique objects, and then on the
# crop outs, run bottle classification, color classification and then food/non_food classification

# Edit the {parent directory} and {Image_dir} to start the program

PARENT_DIR="/home/vikrant/Desktop/virtual_drive/31_08/"
IMAGE_DIR="images/"

CURRENT_DIR=$PWD
UNDISTORTED_OUTPUT="undistorted/"
YOLO_OUTPUT="yolo_output/"
YOLO_LABELS="yolo_output/detection/labels/"
YOLO_CROPOUTS="yolo_crops/"
YOLO_TO_PICKLE_UNIQUE="yolo_output_pickle"
CONVERTED="_converted"

YOLO_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/yolov5"
UNDISTORTED_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/undistortion/"
YOLO_CROP_DIR="/home/vikrant/Desktop/python_pipelines/yolo_crop/"

BOTTLE_NO_BOTTLE_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/bottle_classification/"
COLOR_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/color_classification/"
FOOD_NON_FOOD_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/food_classification/"

PARAMETER_JSON="classification-barebones/helpers/parameters.json"
DETECT_EXEC_DIR="classification-barebones/helpers/exec_files"


BOTTLE_OUTPUT="bottle_output"
COLOR_OUTPUT="color_output"
FOOD_NON_FOOD_OUTPUT="food_output"

cd $UNDISTORTED_SCRIPT_DIR
/usr/bin/python undistort_from_pickle.py --orig_dir $PARENT_DIR$IMAGE_DIR --final_dir $PARENT_DIR$UNDISTORTED_OUTPUT

cd $YOLO_SCRIPT_DIR

/usr/bin/python detect.py --weights runs/train/obj_det_16_08/best.pt --data dataset.yaml --source $PARENT_DIR$UNDISTORTED_OUTPUT  --conf-thres 0.5 --iou-thres 0.5 --save-txt --project $PARENT_DIR$YOLO_OUTPUT --nosave

cd $CURRENT_DIR

/usr/bin/python test_vertical_tracking.py --image_dir $PARENT_DIR$UNDISTORTED_OUTPUT --label_dir $PARENT_DIR$YOLO_LABELS --final_dir $PARENT_DIR$YOLO_TO_PICKLE_UNIQUE

/usr/bin/python left_and_right_label_generator.py --label_dir $PARENT_DIR$YOLO_TO_PICKLE_UNIQUE 



cd $YOLO_CROP_DIR

/usr/bin/python yolo_crop.py --image_dir $PARENT_DIR$UNDISTORTED_OUTPUT --label_dir $PARENT_DIR$YOLO_TO_PICKLE_UNIQUE$CONVERTED --classified_dir $PARENT_DIR$YOLO_CROPOUTS


cd $CURRENT_DIR

/usr/bin/python edit_params.py --source $PARENT_DIR$YOLO_CROPOUTS --output $PARENT_DIR$BOTTLE_OUTPUT --json $BOTTLE_NO_BOTTLE_SCRIPT_DIR$PARAMETER_JSON
/usr/bin/python edit_params.py --source $PARENT_DIR$YOLO_CROPOUTS --output $PARENT_DIR$COLOR_OUTPUT --json $COLOR_SCRIPT_DIR$PARAMETER_JSON
/usr/bin/python edit_params.py --source $PARENT_DIR$YOLO_CROPOUTS --output $PARENT_DIR$FOOD_NON_FOOD_OUTPUT --json $FOOD_NON_FOOD_SCRIPT_DIR$PARAMETER_JSON

cd $BOTTLE_NO_BOTTLE_SCRIPT_DIR$DETECT_EXEC_DIR
/usr/bin/python detect_exec.py

cd $COLOR_SCRIPT_DIR$DETECT_EXEC_DIR
/usr/bin/python detect_exec.py

cd $FOOD_NON_FOOD_SCRIPT_DIR$DETECT_EXEC_DIR
/usr/bin/python detect_exec.py


cd $CURRENT_DIR

/usr/bin/python split_bottle_folder.py --image_and_label_dir $PARENT_DIR$BOTTLE_OUTPUT
/usr/bin/python split_color_folder.py --image_and_label_dir $PARENT_DIR$COLOR_OUTPUT
/usr/bin/python split_food_folder.py --image_and_label_dir $PARENT_DIR$FOOD_NON_FOOD_OUTPUT
