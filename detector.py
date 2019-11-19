import numpy as np
import tensorflow as tf
import cv2
import time
import os
import json


class DetectorAPI:
    def __init__(self, config_path=None):
        load_relative_to_module = config_path is None
        script_path = os.path.dirname(os.path.realpath(__file__))

        if load_relative_to_module:
            config_path = os.path.join(script_path, 'config.json')

        with open(config_path, 'r') as f:
            config = json.load(f)
        self.path_to_ckpt = config['model_path']
        if load_relative_to_module:
            self.path_to_ckpt = os.path.join(script_path, self.path_to_ckpt)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def load_img(self, filepath):
        img = cv2.imread(filepath)
        return img

    def encode_img(self, img):
        retval, buffer = cv2.imencode('.jpg', img)
        return buffer

    def process(self, image, threshold=0.7, max_long=1280):
        im_height, im_width = image.shape[:2]

        if max((im_height, im_width)) > max_long:
            if im_height > im_width:
                new_w, new_h = int(max_long / im_height * im_width), max_long
            else:
                new_w, new_h = max_long, int(max_long / im_width * im_height)
            image = cv2.resize(image, (new_w, new_h))

        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection
        start_time = time.time()
        boxes, scores, classes, num = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time - start_time)

        result = sorted([(box, score) for box, score, cl in zip(boxes[0], scores[0], classes[0])
                         if cl == 1 and score > threshold],
                        key=lambda x: x[0][0] * x[0][1])
        box, score = None, None
        if result:
            box = result[0][0]
            box = [int(box[0] * im_height),
                   int(box[1] * im_width),
                   int(box[2] * im_height),
                   int(box[3] * im_width)]
            score = result[0][1]

        return box, score

    def img_with_box(self, img, box):
        img = img.copy()
        im_height, im_width = img.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2
        font_scale = 0.75

        def add_arrow_and_text(p1, p2):
            cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2)
            p = np.round(abs((p2[0] - p1[0]) / im_width * 100) + abs((p2[1] - p1[1]) / im_height * 100))
            txt = '%d%%' % p
            textsize = cv2.getTextSize(txt, font, font_scale, font_thickness)[0]
            cv2.putText(img, txt,
                        (p2[0] - textsize[0] // 2 + np.sign(p2[0] - p1[0]) * (textsize[0] // 2 + 2),
                         p2[1] + textsize[1] // 2 + np.sign(p2[1] - p1[1]) * (textsize[1] // 2 + 2)),
                        font, font_scale, (0, 0, 255), font_thickness)

        if box is not None:
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            y_m = (box[2] - box[0]) // 2 + box[0]
            x_m = (box[3] - box[1]) // 2 + box[1]
            add_arrow_and_text((0, y_m), (box[1], y_m))
            add_arrow_and_text((im_width, y_m), (box[3], y_m))
            add_arrow_and_text((x_m, 0), (x_m, box[0]))
            add_arrow_and_text((x_m, im_height), (x_m, box[2]))
        return img

    def crop(self, img, box, margin_w=10, margin_h=5):
        img = img.copy()
        im_height, im_width = img.shape[:2]
        margin_x = (box[3] - box[1]) * margin_w // (100 - 2 * margin_w)
        margin_y = (box[2] - box[0]) * margin_h // (100 - 2 * margin_h)
        img_cropped = img[max((0, box[0] - margin_y)):min((box[2] + margin_y, im_height)),
                      max((0, box[1] - margin_x)):min((box[3] + margin_x, im_width))]
        new_box = [box[0] - max((0, box[0] - margin_y)),
                   box[1] - max((0, box[1] - margin_x)),
                   box[2] - box[0] + box[0] - max((0, box[0] - margin_y)),
                   box[3] - box[1] + box[1] - max((0, box[1] - margin_x))]
        return img_cropped, new_box

    def normalize(self, img, box=None, max_long=720):
        img = img.copy()
        im_height, im_width = img.shape[:2]
        if im_height > im_width:
            new_w, new_h = int(max_long / im_height * im_width), max_long
        else:
            new_w, new_h = max_long, int(max_long / im_width * im_height)
        img = cv2.resize(img, (new_w, new_h))
        if box is None:
            return img
        else:
            box = [b * new_w // im_width for b in box]
            return img, box

    def close(self):
        self.sess.close()
        self.default_graph.close()


if __name__ == "__main__":
    odapi = DetectorAPI()
    data_path = 'data/test_1'

    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        img = cv2.imread(filepath)
        box, score = odapi.process(img)
        img_normalized, box_normalized = odapi.normalize(img, box)
        img_with_box = odapi.img_with_box(img_normalized, box_normalized)
        img_cropped, box_cropped = odapi.crop(img, box)
        img_cropped_normalized, box_cropped_normalized = odapi.normalize(img_cropped, box_cropped)
        img_cropped_with_box = odapi.img_with_box(img_cropped_normalized, box_cropped_normalized)

        # show loaded image
        cv2.imshow("frisson", img_normalized)
        cv2.waitKey(0)

        # show detected box
        cv2.imshow("frisson", img_with_box)
        cv2.waitKey(0)

        # show cropped with box
        cv2.imshow("frisson", img_cropped_with_box)
        cv2.waitKey(0)

        # show outcome
        cv2.imshow("frisson", img_cropped_normalized)
        cv2.waitKey(0)
