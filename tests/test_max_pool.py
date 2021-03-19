import tensorflow as tf
import unittest


class TestMethods(unittest.TestCase):

    def test_max_pool(self):
        num_classes = 10
        print('num_classes: ', num_classes)
        batch_size = 2
        y_output = tf.constant(value=[[.1, .3, .2, .0, .8, .4, .1, .1, .1, .2],
                                      [.1, .9, .2, .0, .1, .4, .1, .1, .1, .2]])
        input = tf.reshape(tensor=y_output, shape=[batch_size, num_classes, 1])
        print('input shape: ', input.shape)
        y_max = tf.nn.max_pool1d(input=input, ksize=[num_classes, 1, 1], strides=[1, 1, 1], padding='VALID', data_format='NWC')
        print('y_max: ', y_max)
        y_max = tf.reshape(tensor=y_max, shape=[batch_size])
        print('y_max: ', y_max)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    unittest.main()
