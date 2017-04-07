#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Eudie
"""
Here I am trying to build my first text generator. I will train in on my texting history with bae.
Lets see weather algorithm can generate chat like us
or our relationship need few more years to develop and to get more data :P
I will be using Anna KaRNNa learning for this task
"""

import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
import pickle


dictionaries = pickle.load(open('saved_models/1/dictionaries.pkl', 'rb'))
vocab_to_int = dictionaries['vocab_to_int']
int_to_vocab = dictionaries['int_to_vocab']
vocab = dictionaries['vocab']


def build_rnn(num_classes, batch_size=50, num_steps=50, lstm_size=128, num_layers=2,
              learning_rate=0.001, grad_clip=5, sampling=False):
    # When we're using this network for sampling later, we'll be passing in
    # one character at a time, so providing an option for that
    if sampling == True:
        batch_size, num_steps = 1, 1

    tf.reset_default_graph()

    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # One-hot encoding the input and target characters
    x_one_hot = tf.one_hot(inputs, num_classes)
    y_one_hot = tf.one_hot(targets, num_classes)

    ### Build the RNN layers
    # Use a basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    ### Run the data through the RNN layers
    # Run each sequence step through the RNN and collect the outputs
    outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)
    final_state = state

    # Reshape output so it's a bunch of rows, one output row for each step for each batch
    seq_output = tf.concat(outputs, axis=1)
    output = tf.reshape(seq_output, [-1, lstm_size])

    # Now connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(num_classes))

    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and batch
    logits = tf.matmul(output, softmax_w) + softmax_b

    # Use softmax to get the probabilities for predicted characters
    preds = tf.nn.softmax(logits, name='predictions')

    # Reshape the targets to match the logits
    y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    cost = tf.reduce_mean(loss)

    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    # Export the nodes
    # NOTE: I'm using a namedtuple here because I think they are cool
    export_nodes = ['inputs', 'targets', 'initial_state', 'final_state',
                    'keep_prob', 'cost', 'preds', 'optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = build_rnn(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return ''.join(samples)

checkpoint = "saved_models/1/checkpoints/W_G_i2400_l512_v0.859.ckpt"

lstm_size = 512
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="Eudie:")

with open('results/3.txt', mode='w', encoding='utf-8') as file:
    file.write(samp)
