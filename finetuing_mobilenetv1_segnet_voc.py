
import keras
import tensorflow as tf

import os
import cv2
import random
import itertools
import numpy as np
from keras import backend as K
from keras.models import *
from keras.layers import *

import training
import voc_reader

keras.backend.set_image_data_format('channels_last')


IMAGE_ORDERING = 'channels_last'


def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad',
                      data_format=IMAGE_ORDERING)(inputs)
    x = Conv2D(filters, kernel, data_format=IMAGE_ORDERING,
               padding='valid',
               use_bias=False,
               strides=strides,
               name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING,
                      name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), data_format=IMAGE_ORDERING,
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1), data_format=IMAGE_ORDERING,
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                           name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def get_mobilenet_encoder(input_height=224, input_width=224):
    # todo add more alpha and stuff
    assert (K.image_data_format() ==
            'channels_last'), "Currently only channels last mode is supported"
    assert (IMAGE_ORDERING ==
            'channels_last'), "Currently only channels last mode is supported"
    assert (input_height == 224), \
        "For mobilenet , 224 input_height is supported "
    assert (input_width == 224), "For mobilenet , 224 width is supported "
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3
    img_input = Input(shape=(input_height, input_width, 3))
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    f1 = x
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    f2 = x
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    f3 = x
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    f4 = x
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    f5 = x
    model_name = 'mobilenet_1_0_224_tf_no_top.h5'
    BASE_WEIGHT_PATH = '/home/zhu/procedure/onnx/mobilenet_segnet/all/model/'
    weight_path = BASE_WEIGHT_PATH + model_name
    weights_path = keras.utils.get_file(model_name, weight_path)
    Model(img_input, x).load_weights(weights_path)
    return img_input, [f1, f2, f3, f4, f5]


def my_segmentation(feat, img_input, n_classes = 21):
    x = feat
    x = keras.layers.Conv2D(512, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(256, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(n_classes, (3,3), strides=(1, 1), padding='same', use_bias=False)(x)
    ####
    o_shape = keras.models.Model(img_input, x).output_shape
    i_shape = keras.models.Model(img_input, x).input_shape
    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    x = (keras.layers.Reshape((output_height*output_width,-1)))(x)
    x = keras.layers.Activation('softmax')(x)
    model = keras.models.Model(img_input, x)
    return model



img_input, levels = get_mobilenet_encoder(input_height=224, input_width=224)
feat = levels[3]
model = my_segmentation(feat, img_input, n_classes = 21)
model.summary()


sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer= 'adadelta',
              metrics=['acc']
              )


voc = voc_reader.voc_reader(224,224,8,8)
batch_size = 8
steps_per_epoch = 1464
validation_steps =  1449

training.train_model(model,"/home/zhu/procedure/onnx/mobilenet_segnet/all/te.h5",training.train_generator_data(voc),training.val_generator_data(voc),steps_per_epoch,validation_steps)

model.save('/home/zhu/procedure/onnx/mobilenet_segnet/all/model/finetuing_mobilenetv1_segnet.h5')



