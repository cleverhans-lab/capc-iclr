from mnist.example import x_test as mnist_x_test
from mnist.example import y_test as mnist_y_test

from mnist_util import load_mnist_data
import tensorflow as tf
import numpy as np
import os


def get_data(start_batch, batch_size):
    (x_train, y_train, x_test, y_test) = load_mnist_data(
        start_batch, batch_size)

    is_test = False
    if is_test:
        data = mnist_x_test
        y_test = [mnist_y_test]
    else:
        data = x_test.flatten("C")
        # print('data (x_test): ', data)
        # print('y_test: ', y_test)
    data = data.reshape((-1, 28, 28, 1))
    return data, y_test


def get_dataset(dataset_name):
    if dataset_name in ['mnist', 'cifar10', 'cifar100']:
        ds = getattr(tf.keras.datasets, dataset_name)
    else:
        raise ValueError(f'ds: {dataset_name} not supported.')

    if dataset_name in ['mnist', 'cifar10', 'cifar100']:
        (x_train, y_train), (x_test, y_test) = ds.load_data()
        y_train = tf.compat.v1.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.compat.v1.keras.utils.to_categorical(y_test, num_classes=10)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 255.0
        x_test /= 255.0
    else:
        pass

    return (x_train, y_train), (x_test, y_test)


def load_data(data_dir):
    fs = os.listdir(data_dir)
    query = None
    labels = None
    noisy = None
    for f in fs:
        if f.find('samples') != -1 and f.find('raw-samples') == -1:
            query = os.path.join(data_dir, f)
        elif f.find('labels') != -1 and f.find('aggregated-labels') == -1:
            labels = os.path.join(data_dir, f)
        elif f.find('aggregated-labels') != -1:
            noisy = os.path.join(data_dir, f)
    if query is None:
        raise ValueError(f'Query file not found in data dir: {data_dir}')
    elif labels is None:
        raise ValueError(f'Labels file not found in data dir: {data_dir}')
    elif noisy is None:
        raise ValueError(f'Noisy labels file not found in data dir: {data_dir}')
    return np.load(query), np.load(labels), np.load(noisy)

