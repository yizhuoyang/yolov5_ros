# YOLOv5 ROS
This is a ROS interface for using YOLOv5 for real time object detection on a ROS image topic. It supports inference on multiple deep learning frameworks used in the [official YOLOv5 repository](https://github.com/ultralytics/yolov5).

## Training 
The training and inference codes is based on Yolov5: https://github.com/ultralytics/yolov5. 

### Dependencies
This package is built and tested on Ubuntu 20.04 LTS and ROS Noetic with Python 3.8.

* Clone the packages to ROS workspace and install requirement for YOLOv5 submodule:
```bash
cd <ros_workspace>/src
git clone https://github.com/mats-robotics/detection_msgs.git
git clone https://github.com/yizhuoyang/yolov5_ros.git
cd yolov5_ros/src/yolov5
pip install -r requirements.txt # install the requirements for yolov5
```
* Build the ROS package:
```bash
cd <ros_workspace>
catkin build yolov5_ros -DPYTHON_EXECUTABLE='path_to_your_python_interpreter' # build the ROS package
```
* Make the Python script executable 
```bash
cd <ros_workspace>/src/yolov5_ros/src
chmod +x detect.py
```

## Basic usage
Change the parameter for `input_image_topic` in launch/yolov5.launch to any ROS topic with message type of `sensor_msgs/Image` or `sensor_msgs/CompressedImage`. Other parameters can be modified or used as is.

* Launch the node:
```bash
roslaunch yolov5_ros yolov5.launch
```
*Note:
You can follow the yolo offical website to export the model to .engine file :https://docs.ultralytics.com/modes/export/#export-formats

After acceleartion the yolo network is able to achieve around 15hz speed on orin nx, but is recommanded to train a model with input size around 480*480, which will gaurentee the speed when more programs run at the same time

Specifically cd <ros_workspace>/src/yolov5_ros/src/yolov5 and run:
```bash
python export.py --weights /home/kemove/Downloads/yolov5s.pt --include engine --imgsz 480 --device 0 
```
where yolov5s can be downloaded from the official website: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt. Or can trian own model
## Note:
This repo is based on Ros_Yolov5: https://github.com/mats-robotics/yolov5_ros

The detection.py added the angle prediction of the pedestrians.

We have also developed a more advanced detection framework based on YOLOv11, which can also support yolov8,v9,v10, yolo-world etc. You can find it in the yolov11 branch but the readme is not up to date, but you can use it with the same manner.

## Reference
* Ros_Yolov5: https://github.com/mats-robotics/yolov5_ros
* YOLOv5 official repository: https://github.com/ultralytics/yolov5
* YOLOv3 ROS PyTorch: https://github.com/eriklindernoren/PyTorch-YOLOv3
* Darknet ROS: https://github.com/leggedrobotics/darknet_ros
