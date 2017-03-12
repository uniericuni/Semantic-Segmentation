# Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys, os
import tensorflow as tf
import read_pascal
import pascal
import numpy as np

from config import *

FLAG = None

def main(argv):

    # read current file index
    file_index = 0
    if os.path.exists('./file_index'):
        f = open('./file_index')
        for line in f:
            file_index = int(line)
        f.close()
    f = open('./file_index', 'w+')
    
    # import data
    pascal_reader = read_pascal.PascalReader(file_index)

    # Create the model
    x = tf.placeholder(tf.float32) #shape=[batch size, dimemsionality] 
    y_ = tf.placeholder(tf.float32)
    y = pascal.inference(x, y_)

    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    
    # Session Define
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False))
    sess.run(init)
    saver = tf.train.Saver()
    if file_index != 0:
        saver.restore(sess, "./models/model%s.ckpt"%MODEL_INDEX)
        print("Model restored ...")

    # Training
    print('='*40)
    print('Training ...')
    loss = []
    for i in range(MAX_ITER):
        batch_xs, batch_ys, filename = pascal_reader.next_batch(BATCH_SIZE)
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        _, loss_val = sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        save_path = saver.save(sess, "./models/model%s.ckpt"%MODEL_INDEX)
        loss.append(loss_val)
        f.write(str(file_index+i+1))
        print('Iteration: %s'%str(i) + ' | Filename: %s'%filename + ' | Model saved in file: %s'%save_path)
    np.save('./models/trCrossEntropyLoss%s'%MODEL_INDEX, np.array(loss))
    
    # Testing
    print('='*40)
    print('Tresting ...')
    loss = []
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(MAX_ITER):
        batch_xs, batch_ys = pascal_reader.next_test()
        loss_val = (sess.run(accuracy, feed_dict={x: batch_xs,
                                       y_: batch_ys} ))
        loss.append(loss_val)
        print('Iteration: %s'%str(i) + ' | Error rate: %s'%str(loss_val))
    np.save('./models/tstAccuracy%s'%MODEL_INDEX, np.array(loss))
    print('='*40)

if __name__=='__main__':
    '''
    parser = argparse.ArgumentParser()
    # TODO: the following line
    parser.add_argument('--data_dir', type=str, default=route, help='Directory for storing data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
    '''

    # init model directory
    if not os.path.exists('./models'):
        os.makedirs('models')

    # run the main program
    tf.app.run(main=main, argv=[])
