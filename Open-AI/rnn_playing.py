#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Eudie
"""
Here I am trying to just replicate the tutorial given by sentdex on learning how to play OpenAI game.
First I will just replicate then look to improve upon that
"""

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf


dropout_prob = 1
lstm_layers = 2
input_size = 4
batch_size = 1
rnn_size = 256
no_of_epochs = 750
action_classes = 2
graph = tf.Graph()
seq = 20


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


env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
scores = []
choices = []


with tf.Session(graph=graph) as sess:
    sess.run(init)
    saver.restore(sess, "saved_models/rnn/rnn_256_750_0.ckpt")

    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()

            if len(prev_obs) == 0:
                action = random.randrange(0, 2)
            else:
                pred = sess.run(prediction, feed_dict={inputs: [[prev_obs]], keep_prob: dropout_prob})
                action = np.argmax(pred[0])

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break

        scores.append(score)


print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
