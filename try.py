# Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf

FLAG = None
MAX_ITER = 10000
BATCH_SIZE = 20
LR = 1e-12
DIM

def main():

    # imort data
    # TODO: the following line
    pascal = read_pascal()

    # Create the model
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DIM]) #shape=[batch size, dimemsionality] 
    # W = tf.Variable(tf.zeros([dim, outDim]))
    # b = tf.Variable(tf.zeros([outDim]))
    # y = tf.matmul(x,W)+b

    # Define loss and optimizer
    # TODO: assign inference, logits=output of inference
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, DIM])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

    # Session Define
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Training
    for _ in range(MAX_ITER):
        batch_xs, batch_ys = pascal.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_batch_xs, y_: batch_ys}

    # Testing
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: pascal.test.images,
                                        y_: mnist.test.labels})

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # TODO: the following line
    parser.add_argument('--data_dir', type=str, default=route, help='Directory for storing data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
