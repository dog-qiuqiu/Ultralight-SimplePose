# Ultralight-SimplePose
![image](https://github.com/dog-qiuqiu/Ultralight-SimplePose/blob/master/data/demo.gif)

* Support NCNN mobile terminal deployment
* Based on MXNET(>=1.5.1) GLUON(>=0.7.0) framework
* Top-down strategy: The input image is the person ROI detected by the object detector
* Lightweight mobile terminal human body posture key point model(COCO 17 person_keypoints)
* Detector:https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3
# Model 
#### Mobile inference frameworks benchmark (4*ARM_CPU)
Network|Resolution|Inference time (NCNN/Kirin 990)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:
[Ultralight-Nano-SimplePose](https://github.com/dog-qiuqiu/Ultralight-SimplePose/tree/master/model)|W:192 H:256|~5.4ms|0.224BFlops|2.3MB
### COCO2017 val keypoints metrics evaluate
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.518
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.816
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.558
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.498
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.837
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.604

```
# Install
```
pip install mxnet-cu101 gluoncv
pip install opencv-python cython pycocotools
```
* Install mxnet according to your own cuda version
# Demo
### Test picture
```
python img_demo.py
```
![image](https://github.com/dog-qiuqiu/Ultralight-SimplePose/blob/master/data/Figure_1-1.jpg)
### Test camera stream
```
python cam_demo
```
# How To Train
### Download the coco2017 dataset
* http://images.cocodataset.org/zips/train2017.zip
* http://images.cocodataset.org/annotations/annotations_trainval2017.zip
* http://images.cocodataset.org/zips/val2017.zip
* Unzip the downloaded dataset zip file to the coco directory
* 交流qq群:1062122604
### Train
```
python train_simple_pose.py
```
# Ncnn Deploy
* Dependent library: Opencv Ncnn
* Read the camera video stream test by default, if you test the picture, please modify the code
## Install ncnn
```
$ git clone https://github.com/Tencent/ncnn.git
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build
$ make -j4
$ make install
```
## Run ncnn sample
```
$ cp -rf ncnn/build/install/include ./Ultralight-SimplePose/ncnnsample/
$ cp -rf ncnn/build/install/lib ./Ultralight-SimplePose/ncnnsample/
$ g++ -o ncnnpose ncnnpose.cpp -I include/ncnn/ lib/libncnn.a `pkg-config --libs --cflags opencv` -fopenmp
$ ./ncnnpose
```
## Ncnn Picture test results
![image](https://github.com/dog-qiuqiu/Ultralight-SimplePose/blob/master/data/ncnndemo.png)
# Thanks
* SimplePose Paper:https://arxiv.org/abs/1804.06208
* https://github.com/Tencent/ncnn
* https://gluon-cv.mxnet.io/
