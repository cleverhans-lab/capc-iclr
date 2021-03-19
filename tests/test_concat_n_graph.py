import tensorflow as tf
import unittest
import numpy as np
import ngraph_bridge

# The print below is to retain the ngraph_bridge import so that it is not
# discarded when we optimize the imports.
print('tf version for ngraph_bridge: ', ngraph_bridge.TF_VERSION)


class TestMethods(unittest.TestCase):

    def test_concat2(self):
        batch_size = 2
        y_output = tf.constant(
            value=[[.1, .3, .2, .0, .8, .4, .1, .1, .1, .2],
                   [.1, .9, .2, .0, .1, .4, .1, .1, .1, .2]])
        # y_max = y_output - 5.0
        y_max = np.random.uniform(-10, 10, (2, 10))

        print('y_output: ', y_output)
        print('y_max: ', y_max)

        y_out = tf.concat(
            [y_max, y_output], axis=0)
        print('y_out shape: ', y_out.shape[0])

        y_maxc = y_out[:batch_size]
        y_outputc = y_out[batch_size:]

        print('y_outputc: ', y_outputc)
        print('y_maxc: ', y_maxc)

        np.testing.assert_allclose(actual=y_maxc, desired=y_max)
        np.testing.assert_allclose(actual=y_outputc, desired=y_output)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    unittest.main()
