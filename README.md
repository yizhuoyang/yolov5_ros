# YOLOv5 ROS
This is a ROS interface for using YOLOv5 for real time object detection on a ROS image topic. It supports inference on multiple deep learning frameworks used in the [official YOLOv5 repository](https://github.com/ultralytics/yolov5).

## Installation

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
catkin build yolov5_ros # build the ROS package
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

## Note:
This repo is based on Ros_Yolov5: https://github.com/mats-robotics/yolov5_ros

The detection.py added the angle prediction of the pedestrians.

## Reference
* Ros_Yolov5: https://github.com/mats-robotics/yolov5_ros
* YOLOv5 official repository: https://github.com/ultralytics/yolov5
* YOLOv3 ROS PyTorch: https://github.com/eriklindernoren/PyTorch-YOLOv3
* Darknet ROS: https://github.com/leggedrobotics/darknet_ros
