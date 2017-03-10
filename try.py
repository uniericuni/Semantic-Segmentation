# Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import read_pascal
import pascal

from config import *

FLAG = None

def main(argv):

    # import data
    pascal_reader = read_pascal.PascalReader()

    # Create the model
    x = tf.placeholder(tf.float32) #shape=[batch size, dimemsionality] 
    y = pascal.inference(x)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    
    # Session Define
    #sess = tf.InteractiveSession()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    sess.run(init)

    # Training
    for _ in range(MAX_ITER):
        batch_xs, batch_ys = pascal_reader.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={x: batch_batch_xs, y_: batch_ys})

    # Testing
    # TODO: testing reader
    '''
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: pascal.test.images,
                                        y_: mnist.test.labels})
    '''

if __name__=='__main__':
    '''
    parser = argparse.ArgumentParser()
    # TODO: the following line
    parser.add_argument('--data_dir', type=str, default=route, help='Directory for storing data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
    '''
    tf.app.run(main=main, argv=[sys.argv[0]])
