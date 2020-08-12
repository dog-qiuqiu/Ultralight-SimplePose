# -*- coding: UTF-8 -*-
from __future__ import division

import cv2
import os,time, shutil
import time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultTrainTransform
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultValTransform
from gluoncv.utils.metrics import HeatmapAccuracy

import posenet
import mxnet_demo

#数据集加载
def data_loader(PATH, BATCH_SIZE, IMG_W, IMG_H, HEATMAP_W, HEATMAP_H, num_workers):
    train_dataset = mscoco.keypoints.COCOKeyPoints(PATH,
                                                splits=('person_keypoints_train2017'))
    val_dataset = mscoco.keypoints.COCOKeyPoints(PATH,
                                                splits=('person_keypoints_val2017'))
    #mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    transform_train = SimplePoseDefaultTrainTransform(num_joints=train_dataset.num_joints,
                                                      joint_pairs=train_dataset.joint_pairs,
                                                      image_size=(IMG_H, IMG_W), heatmap_size=(HEATMAP_H, HEATMAP_W),
                                                      scale_factor=0.30, rotation_factor=40, random_flip=True)

    transform_val = SimplePoseDefaultTrainTransform(num_joints=val_dataset.num_joints,
                                                      joint_pairs=val_dataset.joint_pairs,
                                                      image_size=(IMG_H, IMG_W), heatmap_size=(HEATMAP_H, HEATMAP_W),
                                                      scale_factor=0.30, rotation_factor=40, random_flip=True)
    train_data = gluon.data.DataLoader(
        train_dataset.transform(transform_train),
        batch_size=BATCH_SIZE, shuffle=True, last_batch='discard', num_workers= num_workers)

    val_data = gluon.data.DataLoader(
        val_dataset.transform(transform_val),
        batch_size=BATCH_SIZE, shuffle=False, last_batch='discard', num_workers= num_workers)

    return train_data, val_data

#验证集上验证评价指标
def test(net, val_data, context):
    metric = HeatmapAccuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=[context], batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=[context], batch_axis=0)
        weight = gluon.utils.split_and_load(batch[2], ctx_list=[context], batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

#训练函数
def train():
    #数据集文件夹路径
    PATH = 'coco'
    TEST_PATH = "test.jpg"

    #batch size
    BATCH_SIZE = 128
    #输入图像尺寸
    IMG_W, IMG_H = 192, 256
    #特征图尺寸
    HEATMAP_W, HEATMAP_H = 48, 64

    #学习率
    lr = 0.001
    lr_factor = 0.1
    EPOCH = 140
    lr_steps = [90,120]

    #指定训练的GPU
    num_workers=128 #占用内存64gb 值越大数据读取越快但内存占用越多
    context = mx.gpu(0)

    #响应点回归采用L2loss
    L = gluon.loss.L2Loss()
    #构建网络结构 MobileNetV2
    net = posenet.mobilenetv2_05(context, IMG_W, IMG_H)
    #构建评价指标
    metric = HeatmapAccuracy()
    #加载数据
    train_data, val_data = data_loader(PATH, BATCH_SIZE, IMG_W, IMG_H, HEATMAP_W, HEATMAP_H, num_workers)
    #优化器采用ADAM
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    #迭代训练
    lr_counter = 0
    for epoch in range(EPOCH):
        metric.reset()
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1
        for i, batch in enumerate(train_data):
            tic = time.time()
            data = gluon.utils.split_and_load(batch[0], ctx_list=[context], batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[context], batch_axis=0)
            weight = gluon.utils.split_and_load(batch[2], ctx_list=[context], batch_axis=0)
            with ag.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y, w) for yhat, y, w in zip(outputs, label, weight)]
            for l in loss:
                l.backward()
            trainer.step(BATCH_SIZE)
            train_loss = sum([l.mean().asscalar() for l in loss]) / len(loss)
            print('[Epoch %d] batch_num: %d | learn_rate: %.5f | loss: %.8f | time: %.1f' %
                (epoch, i, trainer.learning_rate, train_loss, time.time() - tic))
        _, val_acc = test(net, val_data, context)
        mxnet_demo.demo(TEST_PATH, IMG_W, IMG_H, net, context)
        print("============================Val acc: %.5f============================"%val_acc)
        net.export('model/Ultralight-Nano-SimplePose_%.5f'%val_acc, epoch=epoch)

if __name__ == '__main__':
    train()
