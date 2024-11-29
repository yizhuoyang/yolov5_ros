import numpy as np
from ultralytics import YOLO
import rospy
import ast
from cv_bridge import CvBridge
from rostopic import get_topic_type
from std_msgs.msg import Header, Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage

class YoloDetector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes")
        self.line_thickness = rospy.get_param("~line_thickness")
        weights = rospy.get_param("~weights")
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.callback, queue_size=1
            )
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.callback, queue_size=1
            )
        self.detection_model = YOLO(weights)
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=10
            )

        self.bridge = CvBridge()
        self.angle_pub = rospy.Publisher(
            'angle_list', Float32MultiArray, queue_size=1
        )

    def callback(self, data):
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        array = im
        det_result = self.detection_model(array,classes=ast.literal_eval(self.classes),conf=self.conf_thres,imgsz=self.img_size,verbose=False)
        result_image = det_result[0].plot()[..., ::-1]

        r = det_result[0].boxes
        mask = r.cls == 0
        xyxy_filtered = r.xyxy[mask].cpu().numpy()
        if xyxy_filtered.shape[0] != 0:
            xmin = xyxy_filtered[:, 0]
            xmax = xyxy_filtered[:, 2]
            r1 = (xmax - 318) / 952 * 2
            r2 = (xmin - 318) / 952 * 2
            angle1 = np.degrees(np.arctan(r1))
            angle2 = np.degrees(np.arctan(r2))
            angle_array = np.column_stack((angle1, angle2))
            angle_array = np.array(angle_array, dtype='float32')
        else:
            angle_array = np.array([])

        angle_message = Float32MultiArray(data=angle_array.flatten())
        self.angle_pub.publish(angle_message)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))


if __name__ == "__main__":
    rospy.init_node("object_detection", anonymous=True)
    detector = YoloDetector()
    rospy.spin()
