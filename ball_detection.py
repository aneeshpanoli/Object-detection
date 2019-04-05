import numpy as np
import time
import os
import tensorflow as tf
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2


# requires, tf-gpu 1.12, cuda 9.0, cuDNN v7.1.4, driver 390


class BallDetection:
    def __init__(self):
        self.now = time.time()
        self.cwd = os.getcwd()
        self.num_classes = 1
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.path_to_ckpt = self.cwd + '/content/datalab/fine_tuned_model' + '/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        self.path_to_labels = self.cwd + '/content/datalab' + '/label_map.pbtxt'
        self.test_image_path = self.cwd + '/content/datalab/test_image/image1.png'
        # Size, in inches, of the output images.
        self.output_img_size = (12, 8)
        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, \
                                                                         max_num_classes=self.num_classes,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.detection_graph = tf.Graph()
        self.load_detection_graph()
        self.tf_session = tf.Session(graph=self.detection_graph)

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:  # find out what his is for
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the
                # image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # print(time.time() - self.now)

            # Run inference
            with self.tf_session.as_default():
                output_dict = self.tf_session.run(tensor_dict,
                                                  feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        # print(time.time() - self.now)
        return output_dict

    def load_detection_graph(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def visualize_detection(self, output_dict, image_np):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        # cv2.imshow('image', image_np)
        # cv2.waitKey(0)
        im = Image.fromarray(image_np)
        im.save(self.cwd + '/content/datalab/test_image/image_det.jpg')
        # cv2.destroyAllWindows()

        # print(time.time() - self.now)

    def set_up_object_detection_api(self):
        # print("starting object detection...")
        # self.now = time.time()
        test_image = Image.open(self.test_image_path)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.load_image_into_numpy_array(test_image)

        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np)

        # get actual dimensions of the test image
        width, height = test_image.size

        # find the center of the object detection box
        box_norm_coords = output_dict['detection_boxes'][0]
        # coord normalized output [y_min, x_min, y_max, x_max]
        # y = (y1+y2)/2
        # x = (x1+x2)/2
        # Visualization of the results of a detection.
        # self.visualize_detection(output_dict, image_np)
        # print(box_norm_coords)
        return [int((box_norm_coords[1] * width) + (box_norm_coords[3] * width)) / 2, \
                int((box_norm_coords[0] * height) + (box_norm_coords[2] * height)) / 2]

        # print(box_norm_coords[0] * height) # y1 = ymin * height
        # print(box_norm_coords[1] * width) # x1 = xmin * width
        # print(box_norm_coords[2] * height) # y2 = ymax * height
        # print(box_norm_coords[3] * width) # x2 = xmax * width


if __name__ == '__main__':
    run = BallDetection()
    while True:
        input("Press enter to continue")
        print(run.set_up_object_detection_api())
