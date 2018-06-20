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


def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    raw_data = list(vocabulary.keys())
    raw_data = np.array(raw_data, dtype=np.int32)  # raw
    raw_data = list(vocabulary.keys())
    data_len = len(raw_data)  # how many words in the data_set
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)  # batch_len 就是几个word的意思

    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
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
