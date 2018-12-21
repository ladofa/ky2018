'''
SSD와 POSE 네트웍을 동시에 불러온 뒤
하나의 네트웍으로 동작할 수 있도록 연결

'''


import numpy as np
import tensorflow as tf
import cv2
import time

from pycocotools.coco import COCO
import pycocotools.cocoeval as cocoeval

import numpy as np
import datetime
from collections import defaultdict
import copy
import json
import os

colors = [
    (255, 255, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 128, 255),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (128, 128, 0)
]

labels = []
labels_file = open('coco_labels.txt')
for line in labels_file.readlines():
    line = line.replace('\n', '')
    if line == '':
        break
    labels.append(line)

def detect_and_draw(image):
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_rev = image[:, :, [2, 1, 0]]
    input_image_ssd = np.expand_dims(image_rev, 0)

    output_ssd = sess.run(tensor_dict_ssd, feed_dict={input_ssd:input_image_ssd})

    num_detections = int(output_ssd['num_detections'][0])
    scores = output_ssd['detection_scores'][0]
    classes = output_ssd['detection_classes'][0].astype(np.uint8)
    boxes = output_ssd['detection_boxes'][0]

    #(0, 1) 범위의 boxes를 픽셀 범위로 바꿔준다.
    rate_ori = [[image_height, image_width, image_height, image_width]]
    boxes = np.ndarray.astype(np.prod([boxes, rate_ori], 0), np.int32)
    dst = image.copy()

    font_family = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    line_type = cv2.LINE_AA
    font_thickness = 1    

    for i in range(num_detections):
        box = boxes[i]
        cat = int(classes[i])
        box_color = colors[cat % len(colors)]
        box_color_sum = box_color[0] + box_color[1] + box_color[2]
        if (box_color_sum > 255):
            font_color = (0, 0, 0)
        else:
            font_color = (255, 255, 255)
        if cat < len(labels):
            label_text = labels[cat]
        else:
            label_text = 'unknown'
        text = label_text + ' %.2f' % scores[i]
        
        #boundary
        cv2.rectangle(dst, (box[1], box[0]), (box[3], box[2]), box_color, 3)
        #background of font
        (text_width, text_height) = cv2.getTextSize(text, font_family, fontScale=font_scale, thickness=font_thickness)[0]
        cv2.rectangle(dst, (box[1], box[0] - text_height), (box[1] + text_width, box[0]), box_color, cv2.FILLED)
        #font
        cv2.putText(dst, text, (box[1], box[0]), font_family, font_scale, font_color, lineType=line_type, thickness=font_thickness)
        
    return dst

if __name__ == '__main__':

    sess = tf.Session()

   #pb 파일로부터 네트워크 불러오기
    with tf.gfile.GFile('ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict_ssd = {}
    for key in ['num_detections', 'detection_boxes', 'detection_classes', 'detection_scores']:
        tensor_name = key + ':0'
        tensor_dict_ssd[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    input_ssd = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    image = cv2.imread('test5.jpg')
    dst = detect_and_draw(image)
    cv2.imshow('dst', dst)
    key = cv2.waitKey(0)

