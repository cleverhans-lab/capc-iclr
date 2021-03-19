# This import below is needed to run TensorFlow on top of the he-transformer.
import subprocess
import time

import ngraph_bridge
import numpy as np
import tensorflow as tf

from consts import out_server_name, out_final_name, argmax_times_name, \
    inference_no_network_times_name
# Add parent directory to path
from mnist_util import (
    load_mnist_data,
    server_argument_parser,
    server_config_from_flags,
    load_pb_file,
    print_nodes,
)
import models
from get_r_star import get_rstar_server
from utils.main_utils import array_str
from utils.time_utils import log_timing
from utils.main_utils import round_array
from utils import client_data

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


def run_server(FLAGS, query):
    # tf.import_graph_def(load_pb_file(FLAGS.model_file))
    tf.import_graph_def(
        load_pb_file("/home/dockuser/models/cryptonets-relu.pb"))
    # tf.import_graph_def(load_pb_file(f"/home/dockuser/models/{FLAGS.port}.pb"))
    print("loaded model")
    print_nodes()
    print(f"query: {query.shape}")

    # Get input / output tensors
    x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
        # FLAGS.input_node
        # "import/Placeholder:0"
        "import/input:0"
    )
    y_output = tf.compat.v1.get_default_graph().get_tensor_by_name(
        "import/output/BiasAdd:0"
        # FLAGS.output_node
        # "import/dense/BiasAdd:0"
    )

    # weight_check = tf.compat.v1.get_default_graph().get_tensor_by_name("import/conv2d_1/kernel:0")
    # print(weight_check)

    # Load saved model
    #
    # model = models.Local_Model(FLAGS.num_classes, FLAGS.dataset_name)  # load the model object
    # input_x = tf.compat.v1.placeholder(  # define input
    #     shape=(
    #         None,
    #         3,
    #         32,
    #         32,
    #     ), name="input", dtype=tf.float32)
    # init = tf.compat.v1.global_variables_initializer()
    # model.build((None, 3, 32, 32))
    # model.compile('adam', tf.keras.losses.CategoricalCrossentropy())

    # output_y = model.get_out(input_x)  # input through layers
    # print("loaded model")
    # print_nodes()
    # print(input_x)
    # print(output_y)
    # Get input / output tensors
    # x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
    #     FLAGS.input_node)
    # y_output = tf.compat.v1.get_default_graph().get_tensor_by_name(
    #     FLAGS.output_node)
    # y_max = tf.nn.softmax(y_output)
    # y_max = y_output
    # y_max = approximate_softmax(x=y_output, nr_terms=2)
    # y_max = max_pool(y_output=y_output, FLAGS=FLAGS)
    print('r_star: ', FLAGS.r_star)
    r_star = np.array(FLAGS.r_star)
    # r - r* (subtract the random vector r* from logits)
    y_max = tf.subtract(
        y_output,
        tf.convert_to_tensor(r_star, dtype=tf.float32))

    # Create configuration to encrypt input
    # config = server_config_from_flags(FLAGS, x_input.name)
    config = server_config_from_flags(FLAGS, x_input.name)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # model.initialize_weights(FLAGS.model_file)
        start_time = time.time()
        print(f"query shape before processing: {query.shape}")
        inference_start = time.time()
        y_hat = sess.run(y_max, feed_dict={x_input: query})
        inference_end = time.time()
        print(f"Inference time: {inference_end - inference_start}s")
        with open(inference_no_network_times_name, 'a') as outfile:
            outfile.write(str(inference_end - inference_start))
            outfile.write('\n')
        elasped_time = time.time() - start_time
        print("total time(s)", np.round(elasped_time, 3))
        party_id = int(FLAGS.port)

        msg = "doing 2pc"
        print(msg)
        log_timing(stage='server_client:' + msg,
                   log_file=FLAGS.log_timing_file)
        print('r_star (r*): ', array_str(r_star))
        r_star = round_array(x=r_star, exp=FLAGS.round_exp)
        print('rounded r_star (r*): ', array_str(r_star))
        if FLAGS.backend == 'HE_SEAL':
            argmax_time_start = time.time()
            with open(f'{out_server_name}{FLAGS.port}.txt',
                      'w') as outfile:  # party id
                # assume batch size of 1 for now TODO: make this work for > 1 batch size
                for val in r_star.flatten():
                    outfile.write(f"{int(val)}" + '\n')
            process = subprocess.Popen(
                ['./gc-emp-test/bin/argmax_1', '1', '12345',
                 f'{out_server_name}{FLAGS.port}.txt',
                 f'{out_final_name}{FLAGS.port}.txt'])
            # time.sleep(15)
            process.wait()
            argmax_time_end = time.time()
            with open(argmax_times_name, 'a') as outfile:
                outfile.write(str(argmax_time_end - argmax_time_start))
                outfile.write("\n")
        log_timing(stage='server_client:finished 2PC',
                   log_file=FLAGS.log_timing_file)


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
    if not FLAGS.from_pytorch:
        # (x_train, y_train), (x_test, y_test) = client_data.get_dataset(
        #     FLAGS.dataset)
        raise ValueError('must be from pytroch')
    else:
        queries, labels, noisies = client_data.load_data(FLAGS.dataset_path)
    query = queries[FLAGS.minibatch_id].transpose().reshape((-1, 32 * 32 * 3))
    label = labels[FLAGS.minibatch_id]
    noisy = noisies[FLAGS.minibatch_id]
    (x_train, y_train, x_test, y_test) = client_data.load_mnist_data(0, 1)
    # query = query.transpose(2, 1, 0).flatten("C")[None, ...]
    print(x_test.shape)
    query = x_test
    r_star = run_server(FLAGS, query)
    # TODO: add any checks here
