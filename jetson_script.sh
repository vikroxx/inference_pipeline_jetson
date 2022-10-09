# Bash script to automatically undistort, run yolo, find unique objects, and then on the
# crop outs, run bottle classification, color classification and then food/non_food classification

# Edit the {parent directory} and {Image_dir} to start the program

PARENT_DIR="/media/vinglabs/vinglabs_hdd1/12_09/alu_cans/"
IMAGE_DIR="images/"

CURRENT_DIR=$PWD
UNDISTORTED_OUTPUT="undistorted/"
YOLO_OUTPUT="yolo_output/"
YOLO_LABELS="yolo_output/exp/labels/"
YOLO_CROPOUTS="yolo_crops/"

YOLO_SCRIPT_DIR="/media/vinglabs/dev/vikrant_pipeline/yolov5-master/"
UNDISTORTED_SCRIPT_DIR="/media/vinglabs/dev/vikrant_pipeline/undistortion/"
YOLO_CROP_DIR="/media/vinglabs/dev/vikrant_pipeline/yolo_crop/"

BOTTLE_NO_BOTTLE_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/bottle_classification/"
COLOR_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/color_classification/"
FOOD_NON_FOOD_SCRIPT_DIR="/home/vikrant/Desktop/python_pipelines/food_classification/"

PARAMETER_JSON="classification-barebones/helpers/parameters.json"
DETECT_EXEC_DIR="classification-barebones/helpers/exec_files"


BOTTLE_OUTPUT="bottle_output"
COLOR_OUTPUT="color_output"
FOOD_NON_FOOD_OUTPUT="food_output"

cd $UNDISTORTED_SCRIPT_DIR
#/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python undistort_from_pickle.py --orig_dir $PARENT_DIR$IMAGE_DIR --final_dir $PARENT_DIR$UNDISTORTED_OUTPUT

cd $YOLO_SCRIPT_DIR

#/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python detect.py --weights best.pt --data dataset.yaml --source $PARENT_DIR$UNDISTORTED_OUTPUT  --conf-thres 0.5 --iou-thres 0.5 --save-txt --project $PARENT_DIR$YOLO_OUTPUT --nosave


cd $YOLO_CROP_DIR

/media/vinglabs/dev/virtual_envs_new/inference_env_trt/bin/python yolo_crop.py --image_dir $PARENT_DIR$UNDISTORTED_OUTPUT --label_dir $PARENT_DIR$YOLO_LABELS --classified_dir $PARENT_DIR$YOLO_CROPOUTS
