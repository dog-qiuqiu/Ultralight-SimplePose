# -*- coding: UTF-8 -*-
# KEYPOINTS = {
#     0: "nose",
#     1: "left_eye",
#     2: "right_eye",
#     3: "left_ear",
#     4: "right_ear",
#     5: "left_shoulder",
#     6: "right_shoulder",
#     7: "left_elbow",
#     8: "right_elbow",
#     9: "left_wrist",
#     10: "right_wrist",
#     11: "left_hip",
#     12: "right_hip",
#     13: "left_knee",
#     14: "right_knee",
#     15: "left_ankle",
#     16: "right_ankle"
# }
import cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd

def demo(TEST_PATH, IMG_W, IMG_H, net, context):
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], 
                   [8, 10],[5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]
  
    img = cv2.imread(TEST_PATH)
    resize_img = cv2.resize(img,(IMG_W, IMG_H),interpolation=cv2.INTER_AREA)
    #数据预处理方式
    rgb_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
    data = rgb_img.transpose(2,0,1)
    data = data.reshape((1,3,IMG_H, IMG_W)).astype(np.float32)/255.
    data = mx.nd.array(data, ctx=context)
    mean = (0.485, 0.456, 0.406) 
    std  = (0.229, 0.224, 0.225)
    data = mx.nd.image.normalize(data, mean=mean, std=std)
    out = net(data)
    p = out[0].asnumpy()
    point_index = []
    for i in range(17):
        if np.max(p[i]) > 0.2:
            point_index.append(i)
    for l in joint_pairs:
        if l[0] in point_index and l[1] in point_index:
            y1,x1 = np.unravel_index(np.argmax(p[l[0]]),p[l[0]].shape)
            y2,x2 = np.unravel_index(np.argmax(p[l[1]]),p[l[1]].shape)
            cv2.line(resize_img,(x1*4, y1*4), (x2*4,y2*4), (255,255,0), 2)
            cv2.circle(resize_img, (x1*4,y1*4), 1, (0,255,255), 2)
            cv2.circle(resize_img, (x2*4,y2*4), 1, (0,255,255), 2)
    cv2.imwrite("demo.jpg",resize_img)

if __name__ == '__main__':
    #测试图片
    img_path = "test.jpg"
    IMG_W, IMG_H = 192, 256
    context = mx.gpu(0)
    json_path = 'model/Ultralight-Nano-SimplePose.json'
    params_path = "model/Ultralight-Nano-SimplePose.params"

    net = gluon.SymbolBlock.imports(json_path, ['data'], params_path)
    net.collect_params().reset_ctx(ctx = context)
    demo(img_path, IMG_W, IMG_H, net, context)
