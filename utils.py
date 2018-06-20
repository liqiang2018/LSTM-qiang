#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np
import tensorflow as tf

def read_data(filename):

    '''with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data'''
    txt_data = []
    with tf.gfile.GFile(filename) as fid:
        lines = fid.readlines()
        for line in lines:
            for s in range(len(line.strip('\n'))):
                if s!="" :
                  txt_data.append(line[s])
        return txt_data

def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, dictionary,batch_size, num_steps):
    ##################
    # Your Code here
    data_len = len(vocabulary)
    batch_len = data_len // batch_size

    raw_x = [dictionary.get(w, 0) for w in vocabulary]
    raw_y = [dictionary.get(w, 0) for w in vocabulary[1:]]
    raw_y.append(5000 - 1)

    batch_len = data_len // batch_size
    data_x = np.zeros([batch_size, batch_len], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_len], dtype=np.int32)

    for i in range(batch_size):
        data_x[i] = raw_x[batch_len * i: batch_len * (i + 1)]
        data_y[i] = raw_y[batch_len * i: batch_len * (i + 1)]

    epoch_size = batch_len // num_steps
    print("epoch_size,",epoch_size)

    for i in range(epoch_size):
        x = data_x[:, num_steps * i: num_steps * (i + 1)]
        y = data_y[:, num_steps * i: num_steps * (i + 1)]
        yield (x, y)
    ##################


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
