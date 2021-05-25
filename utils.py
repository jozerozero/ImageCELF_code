from __future__ import print_function

# import gzip
# import cPickle as pkl
import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import pandas as pd
import random
import scipy.io as io


def load_office(source_name, target_name, data_folder, feature_type):
    source_path = os.path.join(data_folder, source_name+"_"+feature_type+".mat")
    target_path = os.path.join(data_folder, target_name+"_"+feature_type+".mat")
    source_data = loadmat(source_path)
    source_feature = source_data["fts"]
    source_label = source_data["labels"] - 1

    target_data = loadmat(target_path)
    target_feature = target_data["fts"]
    target_label = target_data["labels"] - 1

    print(len(source_label))
    print(len(target_label))
    print(source_data)

    source_label = np.squeeze(source_label, -1)
    target_label = np.squeeze(target_label, -1)

    return source_feature, source_label, target_feature, target_label


def load_resnet_office(root, source_name, target_name, data_folder):

    if data_folder == "Office-Home_resnet50":
        data_folder = os.path.join(root, data_folder)
        source_path = os.path.join(data_folder, source_name + "_" + source_name + ".csv")
        target_path = os.path.join(data_folder, source_name + "_" + target_name + ".csv")

        source_data = pd.read_csv(source_path, header=None).values
        source_feature = source_data[:, :-1]
        source_label = np.array(source_data[:, -1], dtype=np.int32)

        target_data = pd.read_csv(target_path, header=None).values
        target_feature = target_data[:, :-1]
        target_label = np.array(target_data[:, -1], dtype=np.int32)
    elif data_folder == "imageCLEF_resnet50":
    # else:
        data_folder = os.path.join(root, data_folder)
        source_path = os.path.join(data_folder, source_name + "_" + source_name + ".csv")
        target_path = os.path.join(data_folder, source_name + "_" + target_name + ".csv")

        source_data = pd.read_csv(source_path, header=None).values
        source_feature = source_data[:, :-1]
        source_label = np.array(source_data[:, -1], dtype=np.int32)

        target_data = pd.read_csv(target_path, header=None).values
        target_feature = target_data[:, :-1]
        target_label = np.array(target_data[:, -1], dtype=np.int32)
    elif data_folder == "office_home_mat":
        data_folder = os.path.join(root, data_folder)
        source_path = os.path.join(data_folder, "OfficeHome-%s-resnet50-noft.mat" % source_name)
        target_path = os.path.join(data_folder, "OfficeHome-%s-resnet50-noft.mat" % target_name)

        source_data = io.loadmat(source_path)
        source_feature = np.squeeze(source_data["resnet50_features"])
        source_label = np.squeeze(source_data["labels"])

        target_data = io.loadmat(target_path)
        target_feature = np.squeeze(target_data["resnet50_features"])
        target_label = np.squeeze(target_data["labels"])
    else:
        source_path = os.path.join(data_folder, source_name + "_" + source_name + ".npy")
        target_path = os.path.join(data_folder, source_name + "_" + target_name + ".npy")

        source_data = np.load(source_path, allow_pickle=True)
        source_feature = source_data[:, :-1]
        source_label = np.array(source_data[:, -1], dtype=np.int32)

        target_data = np.load(target_path, allow_pickle=True)
        target_feature = target_data[:, :-1]
        target_label = np.array(target_data[:, -1], dtype=np.int32)
        # print(":test")
        # exit()
        pass

    return source_feature, source_label, target_feature, target_label, target_feature, target_label


if __name__ == "__main__":

    data = np.load("office_home_extract/Ar_Ar.npy", allow_pickle=True)
    print(data)

def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def conv2d(x, W, padding="SAME"):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def balance_batch_generator(data, batch_size=130, shuffle=True):
    data_dict = {index: list() for index in range(65)}
    print(len(data[0]))
    print(len(data[1]))
    # exit()
    for index, feature in enumerate(data[0]):
        label = data[1][index]
        # print(label)
        # print(np.argmax(label))
        # exit()
        data_dict[np.argmax(label)].append([feature, label])
    # exit()
    while True:
        batch_feature = []
        batch_label = []
        for key, value in data_dict.items():
            for v in random.sample(value, 2):
                batch_feature.append(v[0])
                batch_label.append(v[1])

        yield np.asarray(batch_feature), np.asarray(batch_label)


def batch_generator (data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        # print(start, end, end - start)
        yield [d[start:end] for d in data]

