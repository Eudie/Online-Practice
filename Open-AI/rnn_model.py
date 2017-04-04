#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Eudie
"""
Here I am trying to just replicate the tutorial given by sentdex on learning how to play OpenAI game.
First I will just replicate then look to improve upon that
"""

import numpy as np
import tensorflow as tf


LR = 1e-3

training_data = np.load('saved.npy')

lstm_layers = 2
input_size = 4
batch_size = 32
rnn_size = 64
no_of_epochs = 10
action_classes = 2
graph = tf.Graph()

x = np.array([i[0] for i in training_data])
test_batch_input = x[:24]


def get_batches(int_text, batch_size_, seq_length_):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size_: The size of batch
    :param seq_length_: The length of sequence
    :return: Batches as a Numpy array
    """
    total_batch = len(int_text) // (batch_size_ * seq_length_)
    len_to_consider = int(total_batch * batch_size_ * seq_length_)
    features_ = len(int_text[0])

    input_text = np.array(int_text[:len_to_consider])
    input_text = np.split(input_text, total_batch * batch_size_)

    output = np.empty((total_batch, batch_size_, seq_length_, features_))

    for i in range(batch_size_):
        for j in range(total_batch):
            output[j][i] = input_text[total_batch*i + j]

    return output

get_batches(test_batch_input, batch_size_=2, seq_length_=3)

quit()


with graph.as_default():
    inputs = tf.placeholder(tf.float32, [None, None, input_size])
    action = tf.placeholder(tf.float32, [None, None, action_classes])
    learning_rate = tf.placeholder(tf.float32)

    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)
    Outputs, FinalState = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)

    logit = tf.contrib.layers.fully_connected(Outputs, 2, activation_fn=None)
    prediction = tf.nn.softmax(logit)

    cross_entropy = -tf.reduce_sum(action * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(action, 1), tf.argmax(prediction, 1))
    error = tf.reduce_sum(tf.cast(mistakes, tf.float32))

    saver = tf.train.Saver(tf.all_variables())
    init = tf.initialize_all_variables()



with tf.Session(graph=graph) as sess:
    sess.run(init)

    for step in range(no_of_epochs):
        a
