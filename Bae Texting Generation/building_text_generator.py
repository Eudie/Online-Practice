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
import pickle
import numpy as np
import tensorflow as tf

with open('data/WhatsApp.csv', encoding='utf-8') as f:
    text = f.read()

###############################################################################
# For the first time
###############################################################################
# vocab = set(text)
# vocab_to_int = {c: i for i, c in enumerate(vocab)}
# int_to_vocab = dict(enumerate(vocab))
# pickle.dump({'vocab_to_int': vocab_to_int, 'int_to_vocab': int_to_vocab, 'vocab': vocab},
#             open('saved_models/0/dictionaries.pkl', 'wb'))
###############################################################################


dictionaries = pickle.load(open('saved_models/0/dictionaries.pkl', 'rb'))
vocab_to_int = dictionaries['vocab_to_int']
int_to_vocab = dictionaries['int_to_vocab']
vocab = dictionaries['vocab']

chars = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


def split_data(chars, batch_size, num_steps, split_frac=0.9):
    """
    Split character data into training and validation sets, inputs and targets for each set.
    Arguments
    ---------
    chars: character array
    batch_size: Size of examples in each of batch
    num_steps: Number of sequence steps to keep in the input and pass to the network
    split_frac: Fraction of batches to keep in the training set
    Returns train_x, train_y, val_x, val_y
    """

    slice_size = batch_size * num_steps
    n_batches = int(len(chars) / slice_size)

    # Drop the last few characters to make only full batches
    x = chars[: n_batches * slice_size]
    y = chars[1: n_batches * slice_size + 1]

    # Split the data into batch_size slices, then stack them into a 2D matrix
    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))

    # Now x and y are arrays with dimensions batch_size x n_batches*num_steps

    # Split into training and validation sets, keep the first split_frac batches for training
    split_idx = int(n_batches * split_frac)
    train_x, train_y = x[:, :split_idx * num_steps], y[:, :split_idx * num_steps]
    val_x, val_y = x[:, split_idx * num_steps:], y[:, split_idx * num_steps:]

    return train_x, train_y, val_x, val_y

train_x, train_y, val_x, val_y = split_data(chars, 10, 50)


def get_batch(arrs, num_steps):
    batch_size, slice_size = arrs[0].shape

    n_batches = int(slice_size / num_steps)
    for b in range(n_batches):
        yield [x[:, b * num_steps: (b + 1) * num_steps] for x in arrs]


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

batch_size = 128
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.7

epochs = 50
# Save every N iterations
save_every_n = 200
train_x, train_y, val_x, val_y = split_data(chars, batch_size, num_steps)

model = build_rnn(len(vocab),
                  batch_size=batch_size,
                  num_steps=num_steps,
                  learning_rate=learning_rate,
                  lstm_size=lstm_size,
                  num_layers=num_layers)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Use the line below to load a checkpoint and resume training
    saver.restore(sess, 'saved_models/0/checkpoints/i860_l512_v1.427.ckpt')

    n_batches = int(train_x.shape[1] / num_steps)
    iterations = n_batches * epochs
    for e in range(epochs):

        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for b, (x, y) in enumerate(get_batch([train_x, train_y], num_steps), 1):
            iteration = e * n_batches + b
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.cost, model.final_state, model.optimizer],
                                                feed_dict=feed)
            loss += batch_loss
            end = time.time()
            print('Epoch {}/{} '.format(e + 1, epochs),
                  'Iteration {}/{}'.format(iteration, iterations),
                  'Training loss: {:.4f}'.format(loss / b),
                  '{:.4f} sec/batch'.format((end - start)))

            if (iteration % save_every_n == 0) or (iteration == iterations):
                # Check performance, notice dropout has been set to 1
                val_loss = []
                new_state = sess.run(model.initial_state)
                for x, y in get_batch([val_x, val_y], num_steps):
                    feed = {model.inputs: x,
                            model.targets: y,
                            model.keep_prob: 1.,
                            model.initial_state: new_state}
                    batch_loss, new_state = sess.run([model.cost, model.final_state], feed_dict=feed)
                    val_loss.append(batch_loss)

                print('Validation loss:', np.mean(val_loss),
                      'Saving checkpoint!')
                saver.save(sess, "saved_models/0/checkpoints/i2{}_l{}_v{:.3f}.ckpt".format(iteration, lstm_size, np.mean(val_loss)))




