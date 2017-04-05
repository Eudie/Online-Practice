#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Eudie
"""
Here I am trying to just replicate the tutorial given by sentdex on learning how to play OpenAI game.
First I will just replicate then look to improve upon that
"""

import numpy as np
import tensorflow as tf


training_data = np.load('saved.npy')

dropout_prob = 0.8
lstm_layers = 2
input_size = 4
batch_size = 256
rnn_size = 256
no_of_epochs = 750
action_classes = 2
graph = tf.Graph()
seq = 20
x = np.array([i[0] for i in training_data])
y = np.array([i[1] for i in training_data])
x_1 = x[:24]
y_1 = y[:24]


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

x_batch = get_batches(x, batch_size_=batch_size, seq_length_=seq)
y_batch = get_batches(y, batch_size_=batch_size, seq_length_=seq)
lr = 0.01
graph = tf.Graph()
with graph.as_default():
    inputs = tf.placeholder(tf.float32, [None, None, input_size], name='inputs')
    action = tf.placeholder(tf.float32, [None, None, action_classes], name='action')

    with tf.name_scope("rnn"):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.name_scope("RNN_forward"):
        Outputs, FinalState = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)

    with tf.name_scope("fully_connected"):
        logit = tf.contrib.layers.fully_connected(Outputs, action_classes, activation_fn=None)

    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(logit, name='prediction')
        tf.summary.histogram('predictions', prediction)

    ##################
    # logit_reshaped = tf.reshape(logit, [-1, 2])
    # y_reshaped = tf.reshape(action, [-1, 2])
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit_reshaped, labels=y_reshaped)
    # cost = tf.reduce_mean(loss)
    #
    # # Optimizer for training, using gradient clipping to control exploding gradients
    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    # train_op = tf.train.AdamOptimizer(learning_rate)
    # optimizer_1 = train_op.apply_gradients(zip(grads, tvars))
    # #################
    with tf.name_scope("loss"):
        cross_entropy = -tf.reduce_mean(action * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope("training"):
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate)
        minimize = optimizer.minimize(cross_entropy)

    with tf.name_scope("performance"):
        mistakes = tf.not_equal(tf.argmax(action, 2), tf.argmax(prediction, 2))
        error = tf.reduce_sum(tf.cast(mistakes, tf.float32))
        tf.summary.scalar('error', error)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()


with tf.Session(graph=graph) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('logs/rnn/rnn_256_750_0', sess.graph)

    for epoch_i in range(no_of_epochs):
        state = sess.run(initial_state)
        for i in range(len(x_batch)):
            iteration = epoch_i * len(x_batch) + i
            feed = {inputs: x_batch[i],
                    action: y_batch[i],
                    initial_state: state,
                    learning_rate: lr,
                    keep_prob: dropout_prob}

            summary, error_1, cn, _ = sess.run([merged, error, cross_entropy, minimize], feed_dict=feed)

            train_writer.add_summary(summary, iteration)

    save_path = saver.save(sess, "saved_models/rnn/rnn_256_750_0.ckpt")
