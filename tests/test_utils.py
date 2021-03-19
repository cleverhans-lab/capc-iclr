import unittest

import numpy as np
import tensorflow as tf

from utils.main_utils import array_str
from utils.main_utils import round_array


class TestMethods(unittest.TestCase):

    def test_round_array(self):
        a = np.array([0.28, 0.32])
        a = round_array(x=a, exp=4)
        print('rounded a: ', array_str(a))
        b = [4, 5]
        np.testing.assert_array_equal(x=a, y=b)

    def test_round_array2(self):
        a = np.array([[0.38, 0.42], [0.28, 0.32]])
        a = round_array(x=a, exp=4)
        print('rounded a: ', array_str(a))
        b = [[6, 6], [4, 5]]
        np.testing.assert_array_equal(x=a, y=b)

    def test_round_array3(self):
        a = np.array([[0.38, 0.42], [0.28, 0.32]])
        a = round_array(x=a, exp=100)
        print('rounded a: ', array_str(a))
        b = [[481707228086727178198246752256, 532413252095856328925366976512],
             [354942168063904266196074102784, 405648192073033416923194327040]]
        np.testing.assert_array_equal(x=a, y=b)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    unittest.main()
