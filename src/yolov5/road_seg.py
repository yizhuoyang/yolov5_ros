import numpy as np
import os
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image as Image2
from cv_bridge import CvBridge
import cv2
import rospy
from sensor_msgs.msg import Image as Image_type
from ros_numpy.image import image_to_numpy, numpy_to_image
# import pycuda.autoinit
from scipy.ndimage import zoom
# 前处理
def preprocess_road(image):
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    return np.moveaxis(data, 2, 0)

# 后处理
def postprocess(data):
    num_classes = 2
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette * i % 255 for i in range(num_classes)]).astype("uint8")
    img = Image2.fromarray(data.astype('uint8'), mode='P')
    img.putpalette(colors)
    return img

# 导入推理引擎engine
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger()
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, img):

    input_image = preprocess_road(img)
    image_width, image_height = img.shape[1], img.shape[0]

    with engine.create_execution_context() as context:
        ctx = cuda.Context.attach()
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
        bindings = []

        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_buffer.nbytes)
                bindings.append(int(input_memory))
                cuda.memcpy_htod(input_memory, input_buffer)

            else:
                output_buffer = np.empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()

        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        stream.synchronize()
        ctx.pop()
        # ctx.detach()

        # Release GPU memory
        input_memory.free()
        output_memory.free()


        # Return the output as needed
        return output_buffer.copy()



    # img = postprocess(np.reshape(output_buffer, (image_height, image_width))).convert('RGB')
    # # print(img)
    # return img

class SegRos(object):
    def __init__(self, predictor):
        self.predictor = predictor
        self.image_subscriber = rospy.Subscriber('/camera_D455/color/image_raw', Image_type, callback=self.image_callback, queue_size=1)
        self.image_publisher = rospy.Publisher('/image_publish', Image_type, queue_size=1)

    def image_callback(self, msg):
        image = image_to_numpy(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(640,384))

        result_image = infer(self.predictor, image)
        result_image = result_image.reshape((3,384,640))
        result_image = np.argmax(result_image, 0)
        # result_image = result_image[:,:, np.newaxis]
        result_image = result_image.astype(np.uint8)
        result_image = cv2.resize(result_image,(1280,720))
        result_image= postprocess(np.reshape( result_image, (720,1280))).convert('RGB')
        result_image = np.array(result_image)
        self.image_publisher.publish(numpy_to_image(result_image, encoding='bgr8'))

if __name__ == "__main__":
    TRT_LOGGER = trt.Logger()
    engine_file = "/home/kemove/delta_project/HybridNets/weights/hybridnets2_384x640.engine"
    engine = load_engine(engine_file)
    print('start inference')
    rospy.init_node("seg_node")
    print("you are here")
    yolox_ros = SegRos(predictor=engine)
    rospy.spin()
    print('end')
