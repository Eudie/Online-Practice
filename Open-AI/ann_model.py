#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Eudie
"""
Here I am trying to just replicate the tutorial given by sentdex on learning how to play OpenAI game.
First I will just replicate then look to improve upon that
"""

import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


LR = 1e-3

training_data = np.load('saved.npy')


def neural_network_model(input_size):
    """
    Function is to build NN based on the input size
    :param input_size: feature size of each observation
    :return: tensorflow model
    """
    network = input_data(shape=[None, input_size], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(data, model=False):
    """
    Training of NN model
    :param data: data on which model training will happen
    :param model: predefined tensorflow model
    :return: trained model
    """
    x = np.array([i[0] for i in data])
    y = [i[1] for i in data]
    if not model:
        model = neural_network_model(input_size=len(x[0]))

    model.fit({'input': x}, {'targets': y}, n_epoch=20, snapshot_step=500, show_metric=True, run_id='openai_learning')
    model.save('ann_model.tflearn')
    return model

ann_model = train_model(training_data)
