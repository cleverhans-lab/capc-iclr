# This import below is needed to run TensorFlow on top of the he-transformer.
import subprocess
import time

import ngraph_bridge
import numpy as np
import tensorflow as tf

from consts import out_server_name, out_final_name
# Add parent directory to path
from mnist_util import (
    load_mnist_data,
    server_argument_parser,
    server_config_from_flags,
    load_pb_file,
    print_nodes,
)
from get_r_star import get_rstar_server
from utils.main_utils import array_str
from utils.main_utils import round_array

# The print below is to retain the ngraph_bridge import so that it is not
# discarded when we optimize the imports.
print('tf version for ngraph_bridge: ', ngraph_bridge.TF_VERSION)


def max_pool(y_output, FLAGS):
    # directly use argmax from tf - this is not supported by he-transformer
    # y_max = tf.math.argmax(input=y_output, axis=1)

    # check max_pool from tf
    batch_size = FLAGS.batch_size
    # ksize = y_output.shape.as_list()[0]
    num_classes = FLAGS.num_classes
    print('num_classes: ', num_classes)
    input = tf.reshape(tensor=y_output, shape=[batch_size, num_classes, 1])
    y_max = tf.nn.max_pool1d(input=input, ksize=[num_classes, 1, 1],
                             strides=[1, 1, 1],
                             padding='VALID', data_format='NWC')
    # y_max = tf.reshape(tensor=y_max, shape=[num_classes])
    print('y_max: ', y_max)
    print('y_max: ', y_max.shape)
    return y_max


def run_server(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data(
        FLAGS.start_batch, FLAGS.batch_size)

    # Load saved model
    tf.import_graph_def(load_pb_file(FLAGS.model_file))

    print("loaded model")
    print_nodes()

    # Get input / output tensors
    x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
        FLAGS.input_node)
    y_output = tf.compat.v1.get_default_graph().get_tensor_by_name(
        FLAGS.output_node)
    # y_max = tf.nn.softmax(y_output)
    # y_max = y_output
    # y_max = approximate_softmax(x=y_output, nr_terms=2)
    # y_max = max_pool(y_output=y_output, FLAGS=FLAGS)
    print('r_star: ', FLAGS.rstar)
    print('is r_star [-1.0]', FLAGS.rstar == [-1.0])
    r_star = None
    if FLAGS.rstar is None:
        # for debugging: we don't want any r*
        y_max = y_output
    else:
        if FLAGS.rstar == [-1.0]:
            # Generate random r_star
            if y_test is not None:
                r_shape = y_test.shape
                batch_size = r_shape[0]
                num_classes = r_shape[1]
            else:
                batch_size, num_classes = FLAGS.batch_size, FLAGS.num_classes
            r_star = get_rstar_server(
                max_logit=FLAGS.max_logit,
                batch_size=batch_size,
                num_classes=num_classes,
                exp=FLAGS.rstar_exp,
            )
        else:
            r_star = np.array(FLAGS.rstar)
        # r - r* (subtract the random vector r* from logits)
        y_max = tf.subtract(
            y_output,
            tf.convert_to_tensor(r_star, dtype=tf.float32))

        if FLAGS.debug is True:
            print('y_max shape: ', y_max.shape)
            print('r_star shape: ', r_star.shape)
            y_max = tf.concat([y_max, r_star], axis=0)

    # Create configuration to encrypt input
    config = server_config_from_flags(FLAGS, x_input.name)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        start_time = time.time()
        y_hat = sess.run(y_max, feed_dict={x_input: x_test})
        # y_hat = y_max.eval(feed_dict={x_input: x_test})
        print('y_hat: ', y_hat)
        if FLAGS.debug is True:
            r_star = y_hat[FLAGS.batch_size:]
            y_hat = y_hat[:FLAGS.batch_size]
        print("logits (y_hat): ", array_str(y_hat))
        print("logits (y_hat) type: ", type(y_hat))
        print("logits (y_hat) shape: ", y_hat.shape)
        # print("change y_hat to one_hot encoding")
        y_pred = y_hat.argmax(axis=1)
        print("y_pred: ", y_pred)

        elasped_time = time.time() - start_time
        print("total time(s)", np.round(elasped_time, 3))
        party_id = int(FLAGS.port)

        if r_star is not None:
            print("doing 2pc")
            print('r_star (r*): ', array_str(r_star))
            if FLAGS.round_exp is not None:
                # r_star = (r_star * 2 ** FLAGS.round_exp).astype(np.int64)
                r_star = round_array(x=r_star, exp=FLAGS.round_exp)
                print('rounded r_star (r*): ', array_str(r_star))
            with open(f'{out_server_name}{party_id}.txt',
                      'w') as outfile:  # party id
                # assume batch size of 1 for now TODO: make this work for > 1 batch size
                for val in r_star.flatten():
                    outfile.write(f"{int(val)}" + '\n')
            time.sleep(1)
            process = subprocess.Popen(
                ['./gc-emp-test/bin/argmax_1', '1', '12345',
                 f'{out_server_name}{party_id}.txt',
                 f'{out_final_name}{party_id}.txt'])
            # time.sleep(15)
            process.wait()
        else:
            print('r_star is None in he_server.py!')

    if not FLAGS.enable_client:
        y_test_label = np.argmax(y_test, 1)

        if FLAGS.batch_size < 60:
            print("y_hat", np.round(y_hat, 2))

        y_pred = np.argmax(y_hat, 1)
        correct_prediction = np.equal(y_pred, y_test_label)
        error_count = np.size(correct_prediction) - np.sum(correct_prediction)
        test_accuracy = np.mean(correct_prediction)

        print("Error count", error_count, "of", FLAGS.batch_size, "elements.")
        print("Accuracy: %g " % test_accuracy)


if __name__ == "__main__":
    FLAGS, unparsed = server_argument_parser().parse_known_args()

    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)
    if FLAGS.encrypt_server_data and FLAGS.enable_client:
        raise Exception(
            "encrypt_server_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map"
        )
    if FLAGS.model_file == "":
        raise Exception("FLAGS.model_file must be set")

    run_server(FLAGS)
