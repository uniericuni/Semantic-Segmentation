# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import tensorflow as tf
import numpy as np

from config import *
from six.moves import urllib

# Basic model parameters.
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                            """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                        tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        # print(shape)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _bias_with_weight_decay(name, shape, val, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
            name,
            shape,
            tf.constant_initializer(val))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def module_wrap(image, in_ch):
    IN_CH = {0:in_ch}
    SCOPE_NUM = {0:1}
    OUT_CH = {0:3}
    BUFF = {0:image}

    def module(conv_num, out_ch):
        # conv
        for i in range(0,conv_num):  
            scope_name = 'conv'+str(SCOPE_NUM[0])+str(i)
            with tf.variable_scope(scope_name) as scope:
                kernel = _variable_with_weight_decay('weights',
                                                        shape=[3, 3, IN_CH[0], out_ch],
                                                        stddev=5e-2,
                                                        wd=1*WEIGHT_DECAY)
                IN_CH[0] = out_ch
                conv = tf.nn.conv2d(BUFF[0], kernel, [1, 1, 1, 1], padding='SAME')
                bias = _bias_with_weight_decay( 'biases',
                                                shape=[out_ch],
                                                val=0.0,
                                                wd=0*WEIGHT_DECAY)
                pre_activation = tf.nn.bias_add(conv, bias)
                BUFF[0] = tf.nn.relu(pre_activation, name=scope.name)
                _activation_summary(BUFF[0])
 
        # pool
        scope_name = 'pool'+str(SCOPE_NUM[0])
        BUFF[0] = tf.nn.max_pool(BUFF[0], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name=scope_name)
        SCOPE_NUM[0] += 1
        IN_CH[0] = out_ch
        return BUFF[0]

    return module

def computeShape(shape, bottom, stride):
    if shape is None:
        in_shape = tf.shape(bottom)
        h = ((in_shape[1]-1) * stride) + 1
        w = ((in_shape[2]-1) * stride) + 1
        new_shape = [in_shape[0], h, w, NUM_CLASSES]
    else:
        kernel_size = 0
        new_shape = [shape[0], shape[1]+kernel_size, shape[2]+kernel_size, NUM_CLASSES]

    return tf.stack(new_shape)

def inference(images, labels):
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    
    module = module_wrap(images, 3)

    # 2 conv + 1 pool
    pool1 = module(2, 64)

    # 2 conv + 1 pool
    pool2 = module(2, 128)

    # 3 conv + 1 pool
    pool3 = module(3, 256)

    # 3 conv + 1 pool
    pool4 = module(3, 512)

    # 3 conv + 1 pool
    pool5 = module(3, 512)

    # fully connected 6
    with tf.variable_scope('fc6') as scope:
        kernel = _variable_with_weight_decay(   'weights',
                                                shape=[7, 7, 512, 4096],
                                                stddev=5e-2,
                                                wd=1*WEIGHT_DECAY)
        conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _bias_with_weight_decay(   'biases',
                                            shape=[4096],
                                            val=0.0,
                                            wd=0*WEIGHT_DECAY)
        pre_activation = tf.nn.bias_add(conv, biases)
        fc6 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(fc6)

    # dropout 6
    drop6 = tf.nn.dropout(fc6, 0.5)

    # fully connected 7
    with tf.variable_scope('fc7') as scope:
        kernel = _variable_with_weight_decay(   'weights',
                                                shape=[1, 1, 4096, 4096],
                                                stddev=5e-2,
                                                wd=1*WEIGHT_DECAY)
        conv = tf.nn.conv2d(drop6, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _bias_with_weight_decay(   'biases',
                                            shape=[4096],
                                            val=0.0,
                                            wd=0*WEIGHT_DECAY)
        pre_activation = tf.nn.bias_add(conv, biases)
        fc7 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(fc7)

    # dropout 7
    drop7 = tf.nn.dropout(fc7, 0.5)

    # score fr
    with tf.variable_scope('score_fr') as scope:
        kernel = _variable_with_weight_decay(   'weights',
                                                shape=[1, 1, 4096, 21],
                                                stddev=5e-2,
                                                wd=1*WEIGHT_DECAY)
        conv = tf.nn.conv2d(drop7, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _bias_with_weight_decay(   'biases',
                                            shape=[21],
                                            val=0.0,
                                            wd=0*WEIGHT_DECAY)
        pre_activation = tf.nn.bias_add(conv, biases)
        score_fr = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(score_fr)

    # upscore2
    with tf.variable_scope('upscore2') as scope:
        kernel = _variable_with_weight_decay(   'weights',
                                                shape=[13, 13, NUM_CLASSES, NUM_CLASSES],
                                                stddev=5e-2,
                                                wd=False)
        output_shape = computeShape(tf.shape(pool4), score_fr, 2)
        upscore2 = tf.nn.conv2d_transpose(score_fr, kernel, output_shape, [1, 2, 2, 1], padding='VALID')
        _activation_summary(upscore2)

    # score_pool4
    with tf.variable_scope('score_pool4') as scope:
        kernel = _variable_with_weight_decay(   'weights',
                                                shape=[1, 1, 512, 21],
                                                stddev=5e-2,
                                                wd=1*WEIGHT_DECAY)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _bias_with_weight_decay(   'biases',
                                            shape=[21],
                                            val=0.0,
                                            wd=0*WEIGHT_DECAY)
        score_pool4 = tf.nn.bias_add(conv, biases)
        _activation_summary(score_pool4)

    # score_pool4c
    score_pool4c = score_pool4

    # fuse_pool4
    fuse_pool4 = tf.add(score_pool4c, upscore2)

    # upscore16
    with tf.variable_scope('upscore16') as scope:
        kernel = _variable_with_weight_decay(   'weights',
                                                shape=[32, 32, NUM_CLASSES, NUM_CLASSES],
                                                stddev=5e-2,
                                                wd=False)
        output_shape = computeShape(tf.shape(images), fuse_pool4, 16)
        upscore16 = tf.nn.conv2d_transpose(fuse_pool4, kernel, output_shape, [1, 16, 16, 1], padding='SAME')
        _activation_summary(upscore16)

    # softmax
    with tf.variable_scope('score') as scope:
        score = tf.nn.softmax(upscore16)

    return score
    '''
    # score
    with tf.variable_scope('score') as scope:
        score = tf.image.extract_glimpse(   upscore16,
                                            size = tf.shape(labels)[1:3],
                                            offsets = tf.zeros([1,2]),
                                            centered = True )
        _activation_summary(score)

    return tf.Print(score, [tf.shape(score)]);
    # output
    return upscore16
    '''
