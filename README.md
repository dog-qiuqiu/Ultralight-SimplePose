# Ultralight-SimplePose
![image](https://github.com/dog-qiuqiu/Ultralight-SimplePose/blob/master/data/demo.gif)

* Support NCNN mobile terminal deployment
* Based on MXNET(>=1.5.1) GLUON(>=0.7.0) framework
* Top-down strategy: The input image is the person ROI detected by the object detector
* Lightweight mobile terminal human body posture key point model(COCO 17 person_keypoints)
* Detector:https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3
# Model 
#### Mobile inference frameworks benchmark (4*ARM_CPU)
Network|COCO AP(0.5)|Resolution|Inference time (NCNN/Kirin 990)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:
[Ultralight-Nano-SimplePose](https://github.com/dog-qiuqiu/Ultralight-SimplePose/tree/master/model)|71.0%|W:192 H:256|~5.4ms|&BFlops|2.3MB
# Demo
* zzzz
# How To Train
## Download the coco2017 dataset
* http://images.cocodataset.org/zips/train2017.zip
* http://images.cocodataset.org/annotations/annotations_trainval2017.zip
* http://images.cocodataset.org/zips/val2017.zip
* Unzip the downloaded dataset zip file to the coco directory
## Train
```
python train_simple_pose.py
```
# Ncnn Deploy
* zzzz
# Thanks
* SimplePose Paper:https://arxiv.org/abs/1804.06208
* https://github.com/Tencent/ncnn
* https://gluon-cv.mxnet.io/
