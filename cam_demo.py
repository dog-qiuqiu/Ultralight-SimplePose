from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints


def keypoint_detection(img, detector, pose_net, ctx):
    x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(img, short=480, max_size=1024)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(scaled_img, class_IDs, scores, bounding_boxs,
                                                       output_shape=(256, 192), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        scale = 1.0 * img.shape[0] / scaled_img.shape[0]
        img = cv_plot_keypoints(img.asnumpy(), pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                box_thresh=1, keypoint_thresh=0.1, scale=scale)
    return img

if __name__ == '__main__':
    json_path = "Ultralight-Nano-SimplePose.json"
    params_path = "Ultralight-Nano-SimplePose.params"
    
    ctx = mx.gpu(0)
    detector_name = "yolo3_mobilenet1.0_coco"
    detector = get_model(detector_name, pretrained=True, ctx=ctx)
    detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
    net = gluon.SymbolBlock.imports(json_path, ['data'], params_path)
    net.collect_params().reset_ctx(ctx=ctx)

    cap = cv2.VideoCapture(0)
    time.sleep(1)  ### letting the camera autofocus

    while True:
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        img = keypoint_detection(frame, detector, net, ctx=ctx)
        cv_plot_image(img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
