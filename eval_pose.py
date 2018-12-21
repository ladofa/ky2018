'''
SSD와 POSE 네트웍을 동시에 불러온 뒤
하나의 네트웍으로 동작할 수 있도록 연결

'''


import numpy as np
import tensorflow as tf
import cv2
import time


import numpy as np
import datetime
from collections import defaultdict
import copy
import json
import time
import itertools


pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

def heatmap2joints(heatmap, heatmap_flip):

    nr_skeleton = 17
    output_shape = (64, 48) #height, width
    data_shape = (256, 192) #height, width
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    res = heatmap
    res = res.transpose(0, 3, 1, 2)

    start_id = 0
    cls_skeleton = np.zeros((1, nr_skeleton, 3))
    crops = np.zeros((1, 4))
    details = []

    for i in range(0, res.shape[0]):
        fmp = heatmap_flip[0]
        fmp = cv2.flip(fmp, 1)
        fmp = list(fmp.transpose((2, 0, 1)))
        for (q, w) in symmetry:
            fmp[q], fmp[w] = fmp[w], fmp[q]
        fmp = np.array(fmp)
        res[i] += fmp
        res[i] /= 2

    for test_image_id in range(0, res.shape[0]):
        r0 = res[test_image_id - start_id].copy()
        r0 /= 255.
        r0 += 0.5
        for w in range(nr_skeleton):
            res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])
        border = 10
        dr = np.zeros((nr_skeleton, output_shape[0] + 2 * border, output_shape[1] + 2 * border))
        dr[:, border:-border, border:-border] = res[test_image_id - start_id][:nr_skeleton].copy()
        for w in range(nr_skeleton):
            dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)
        for w in range(nr_skeleton):
            lb = dr[w].argmax()
            y, x = np.unravel_index(lb, dr[w].shape)
            dr[w, y, x] = 0
            lb = dr[w].argmax()
            py, px = np.unravel_index(lb, dr[w].shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, output_shape[1] - 1))
            y = max(0, min(y, output_shape[0] - 1))
            cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)
            cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

    return cls_skeleton[0][:, :2]
     
def prepare_image_for_pose(image, box):
    ''' POSE에 넣을 이미지를 만들어준다. 
    단순 crop 뿐 아니라 리사이즈 후 가로 혹은 세로 비율에 맞춰서
    여백에 padding까지 넣어준다.
    
    '''

    input_image_rows = 256
    input_image_cols = 192

    input_width = box[3] - box[1]
    input_height = box[2] - box[0]

    #축소 비율
    rate_h = input_height / input_image_rows
    rate_w = input_width / input_image_cols

    if rate_h > rate_w:
        #리사이즈 비율, h, w 중 더 많이 축소되는 쪽을 선택
        rate = rate_h
        #리사이즈 크기
        rh = input_image_rows
        rw = int(input_width / rate)
        #패딩을 위한 마진
        margin_w = int( (input_image_cols - rw) / 2 )
        margin_h = 0
    else:
        rate = rate_w
        rh = int(input_height / rate)
        rw = input_image_cols
        margin_w = 0
        margin_h = int( (input_image_rows - rh) / 2 )

    #왼쪽 정렬?
    #margin_w = 0
    #margin_h = 0
        
    resized = cv2.resize(image[box[0]:box[2], box[1]:box[3], :], (rw, rh))
    input_image = np.zeros([input_image_rows, input_image_cols, 3])
    input_image[margin_h:margin_h+rh, margin_w:margin_w+rw, :] = resized

    #리사이즈 이미지를 리턴하고
    #POSE 결과를 다시 원래 크기로 복원할 수 있도록 각종 파라미터도 같이 리턴
    return input_image, rate, margin_h, margin_w


parts = [
    'nose', #0
    'l_eye', #1
    'r_eye', #2
    'l_ear', #3
    'r_ear', #4
    'l_shoulder', #5
    'r_shoulder', #6
    'l_elbow', #7
    'r_elbow', #8
    'l_wrist', #9
    'r_wrist', #10
    'l_hip', #11
    'r_hip', #12
    'l_knee', #13
    'r_knee', #14
    'l_ankle', #15
    'r_ankle' #16
]

#그리는 용도 외에는 쓸 일이 없다..
links = [
    #얼굴
    (parts.index('l_eye'),      parts.index('r_eye')),

    (parts.index('nose'),       parts.index('l_eye')),
    (parts.index('nose'),       parts.index('r_eye')),

    (parts.index('l_eye'),      parts.index('l_ear')),
    (parts.index('r_eye'),      parts.index('r_ear')),
    
    
    (parts.index('l_ear'),      parts.index('l_shoulder')),
    (parts.index('r_ear'),      parts.index('r_shoulder')),

    (parts.index('l_shoulder'), parts.index('r_shoulder')),

    (parts.index('l_shoulder'), parts.index('l_elbow')),
    (parts.index('r_shoulder'), parts.index('r_elbow')),

    (parts.index('l_elbow'),    parts.index('l_wrist')),#10
    (parts.index('r_elbow'),    parts.index('r_wrist')),
    
     
    (parts.index('l_shoulder'), parts.index('l_hip')),
    (parts.index('r_shoulder'), parts.index('r_hip')),

    (parts.index('l_hip'),      parts.index('r_hip')),
    
    (parts.index('l_hip'),      parts.index('l_knee')),
    (parts.index('r_hip'),      parts.index('r_knee')),

    (parts.index('l_knee'),     parts.index('l_ankle')),
    (parts.index('r_knee'),     parts.index('r_ankle')),
]

links_color = [
    #얼굴
    (0, 255, 255),#0
    (0, 255, 192), #1
    (0, 192, 255),#2
    (0, 204, 128),#3
    (0, 128, 255),#4

    (0, 255, 64),#5
    (0, 64, 255),#6

    (128, 255, 255),#7

    (0, 255, 0),#8
    (0, 0, 255),#9

    (0, 255, 128),#10
    (128, 0, 255),#11
    (128, 255, 128),#12
    (128, 128, 255),#13

    (255, 128, 128),#14

    (255, 255, 128),#15
    (255, 128, 255),#16
    (255, 255, 0),#17
    (255, 0, 255),#18
]

def draw_body_parts(npimg, body_parts, valids = None):
    #npimg = npimg.astype(np.uint8)
    #선을 그린다.
    for idx, link in enumerate(links):
        #두 지점을 연결해야 한다.
        bp0 = body_parts[link[0]]
        bp1 = body_parts[link[1]]

        #valids가 input으로 들어왔으면 체크를 해준다.
        if valids is not None:
            if not valids[link[0]] or not valids[link[1]]:
                continue
        
        #링크의 두 지점 모두 존재해야 한다.
        if bp0[0] <= 0 or bp1[0] <= 0:
            continue
        
        cv2.line(npimg, (int(bp0[0]), int(bp0[1])), (int(bp1[0]), int(bp1[1])), links_color[idx], 2)
    #점을 그린다.
    for idx, bp in enumerate(body_parts):
        if valids is not None:
            if not valids[idx]:
                continue

        if bp[0] <= 0:
            continue

        cv2.circle(npimg, (int(bp[0]), int(bp[1])), 2, (0, 0, 0), cv2.FILLED)
    return npimg

if __name__ == '__main__':

    test_subset = False
    showing_result = True
    use_gt_detection = False

    sess = tf.Session()

   #pb 파일로부터 네트워크 불러오기
    with tf.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='ssd')

    with tf.gfile.GFile('cpn_mb4.pb', 'rb') as fid:
        serialized_graph = fid.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='pose')

    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict_ssd = {}
    for key in ['num_detections', 'detection_boxes', 'detection_classes', 'detection_scores']:
        tensor_name = 'ssd/' + key + ':0'
        tensor_dict_ssd[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    tensor_dict_pose = {}
    for key in ['joints', 'valids', 'heatmap']:
        tensor_name = 'pose/' + key + ':0'
        tensor_dict_pose[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    input_ssd = tf.get_default_graph().get_tensor_by_name('ssd/image_tensor:0')
    input_pose = tf.get_default_graph().get_tensor_by_name('pose/input_image:0')


    image = cv2.imread('test2.jpg')

    
    #ssd 활용
    image_width = image.shape[1]
    image_height = image.shape[0]
    input_image_ssd = np.expand_dims(image, 0)
    
    output_ssd = sess.run(tensor_dict_ssd, feed_dict={input_ssd:input_image_ssd})

    num_detections = int(output_ssd['num_detections'][0])
    classes = output_ssd['detection_classes'][0].astype(np.uint8)
    boxes = output_ssd['detection_boxes'][0]

    #(0, 1) 범위의 boxes를 픽셀 범위로 바꿔준다.
    rate_ori = [[image_height, image_width, image_height, image_width]]
    boxes = np.ndarray.astype(np.prod([boxes, rate_ori], 0), np.int32)
        
    dst = image.copy()
       
    #각각의 디텍션에 대해서 처리
    for i in range(num_detections):
        #사람만 취급
        if classes[i] != 1:
            continue

        box = boxes[i]
        input_image, rate, margin_h, margin_w = prepare_image_for_pose(image, box)
        input_image_pose = np.expand_dims(input_image, 0)
        input_image_flip = cv2.flip(input_image, 1)
        input_image_pose_flip = np.expand_dims(input_image_flip, 0)

        output = sess.run(tensor_dict_pose, feed_dict={input_pose:input_image_pose})
        output_flip = sess.run(tensor_dict_pose, feed_dict={input_pose:input_image_pose_flip})
        joints_dt = output['joints']
        valids = output['valids']
        heatmap = output['heatmap']

        # for h in range(17):
        #     cv2.imshow('heatmap %d' % h, heatmap[0][:, :, h])

        joints_dt = joints_dt[0]
        # joints_dt = heatmap2joints(heatmap, output_flip['heatmap'])
        valids_dt = valids[0]

        #joints_dt를 원래 사이즈에 맞게 복원
        joints_dt = joints_dt - [[margin_w, margin_h]]
        joints_dt = joints_dt * rate
        #크롭 이전의 위치로 복원
        joints_dt = joints_dt + [[box[1], box[0]]]

        #joints_dt = np.ndarray.astype(joints_dt, np.int32)
        keypoints = np.concatenate([joints_dt, (np.ones([17, 1]) * 2)], 1)
        keypoints = np.reshape(keypoints, [17 * 3])
        keypoints = keypoints.tolist()

        ####
        draw_body_parts(dst, joints_dt, valids_dt)
            
    cv2.imshow('dt', dst)
    key = cv2.waitKey(0)


