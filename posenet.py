# -*- coding: UTF-8 -*-
import mxnet as mx
import gluoncv as gcv
from mxnet.gluon import nn
from mxnet import gluon, nd
from gluoncv.model_zoo import get_model

#构建网络结构
def mobilenetv2_05(context, IMG_W, IMG_H):
    base_model = get_model('mobilenetv2_0.5', pretrained=True, ctx=context,norm_layer=gcv.nn.BatchNormCudnnOff)
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        #添加主干网络
        net.add(base_model.features[:-4])

        net.add(nn.Conv2D(128, 1, strides=1, padding=0, use_bias=False))
        net.add(gcv.nn.BatchNormCudnnOff(scale=True))
        net.add(nn.Activation('relu'))

        #构建upsample 模块
        net.add(nn.Conv2DTranspose(128, 4, strides=2, padding=1,groups=128, in_channels=128, use_bias=False))
        net.add(gcv.nn.BatchNormCudnnOff(scale=True))
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(128, 1, strides=1, padding=0, use_bias=False))
        net.add(gcv.nn.BatchNormCudnnOff(scale=True))
        net.add(nn.Activation('relu'))

        net.add(nn.Conv2DTranspose(128, 4, strides=2, padding=1,groups=128, in_channels=128, use_bias=False))
        net.add(gcv.nn.BatchNormCudnnOff(scale=True))
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(128, 1, strides=1, padding=0, use_bias=False))
        net.add(gcv.nn.BatchNormCudnnOff(scale=True))
        net.add(nn.Activation('relu'))

        net.add(nn.Conv2DTranspose(128, 4, strides=2, padding=1,groups=128, in_channels=128, use_bias=False))
        net.add(gcv.nn.BatchNormCudnnOff(scale=True))
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(128, 1, strides=1, padding=0, use_bias=False))
        net.add(gcv.nn.BatchNormCudnnOff(scale=True))
        net.add(nn.Activation('relu'))

        net.add(nn.Conv2D(17, 1, strides=1, padding=0, use_bias=False))

    net.initialize(ctx=context)

    x = mx.nd.ones((1, 3, IMG_H, IMG_W), ctx=context)
    net.summary(x)
    net.hybridize(static_alloc=True, static_shape=True)
    return net
