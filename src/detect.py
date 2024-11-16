#!/usr/bin/env python3
import platform
print('ok')
print(platform.python_version())
import rospy
import cv2
import math
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import time
import sys
from rostopic import get_topic_type
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import Point
import std_msgs
# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.view_image = rospy.get_param("~view_image")
        # Initialize weights 
        weights = rospy.get_param("~weights")
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1,3,480,480))  # warmup   
        
        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.callback, queue_size=1
            )
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.callback, queue_size=1
            )

        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=1
        )
        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=1
            )
        
        # Initialize angle publisher
        self.angle_pub = rospy.Publisher(
            'angle_list', Float32MultiArray, queue_size=1
        )   

        self.center_points_pub = rospy.Publisher('center_points', Float32MultiArray, queue_size=1)
        # Initialize CV_Bridge
        self.bridge = CvBridge()
     

    def callback(self, data):
        """adapted from yolov5/detect.py"""
        # print(data.header)
        start = time.time()
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        im, im0 = self.preprocess(im)
        # print(im.shape)
        # print(img0.shape)
        # print(img.shape)

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255

        length = im0.shape[1]

        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        angle_array = []

        center_points = Float32MultiArray()
        center_points.layout.dim.append(std_msgs.msg.MultiArrayDimension())
        center_points.layout.dim[0].label = "center_points"
        center_points.layout.dim[0].size = 0
        center_points.layout.dim[0].stride = 0

        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                c = int(cls)
                if c==0:
                # Fill in bounding box message
                    bounding_box.Class = self.names[c]
                    bounding_box.probability = conf 
                    bounding_box.xmin = int(xyxy[0])
                    bounding_box.ymin = int(xyxy[1])
                    bounding_box.xmax = int(xyxy[2])
                    bounding_box.ymax = int(xyxy[3])
                    
                    r1 = (bounding_box.xmax-318)/952*2
                    r2 = (bounding_box.xmin-318)/952*2

                    angle1 = math.degrees(math.atan(r1))
                    angle2 = math.degrees(math.atan(r2))

                    # Calculate center point
                    center_x = (bounding_box.xmin + bounding_box.xmax) // 2
                    center_y =  bounding_box.ymax

                    # Add center point to the list
                    center_points.data.extend([bounding_box.xmin,bounding_box.ymin,bounding_box.xmax,bounding_box.ymax])
                    # p1 = bounding_box.xmax-length/2+0.00001
                    # p2 = bounding_box.xmin-length/2+0.00001
                    # h  = im.shape[-2]*2-bounding_box.ymax
                    # angle1 = math.degrees(math.atan(h/p1))
                    # angle2 = math.degrees(math.atan(h/p2))
                    # if angle1<0:
                    #     angle1 += 180
                    # if angle2<0:
                    #     angle2 += 180

                    angle_data = [angle1,angle2]
                    angle_array.append(angle_data)
                    bounding_boxes.bounding_boxes.append(bounding_box)

                    # Annotate the image
                    if self.publish_image or self.view_image:  # Add bbox to image
                        # integer class
                        label = f"{self.names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))       
            angle_array = np.array(angle_array,dtype='float32')
            angle_message = Float32MultiArray(data=angle_array.flatten())
            # print(angle_message)
                ### POPULATE THE DETECTION MESSAGE HERE
            center_points.layout.dim[0].size = len(center_points.data) // 2
            center_points.layout.dim[0].stride = len(center_points.data)

            # Stream results
            im0 = annotator.result()

        # Publish prediction
            # print(bounding_boxes)
            self.pred_pub.publish(bounding_boxes)
            # print('Publishing')
            self.angle_pub.publish(angle_message)
            self.center_points_pub.publish(center_points)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))

        end = time.time()
        used_time = (end-start)*1000
        # print("The time for processing is {} ms".format(used_time))
        

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0 


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    print('start')
    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()
    
    rospy.spin()
