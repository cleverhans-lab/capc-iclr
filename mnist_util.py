#!/usr/bin/python3

# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import argparse
import os.path

# import ngraph_bridge
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

from utils.time_utils import get_timestamp

# The print below is to retain the ngraph_bridge import so that it is not
# discarded when we optimize the imports.
# print('tf version for ngraph_bridge: ', ngraph_bridge.TF_VERSION)

DEFAULT_PORT = 35000
import pprint


def print_nodes(graph_def=None):
    """Prints the node names of a graph_def.
        If graph_def is not provided, use default graph_def"""

    if graph_def is None:
        nodes = [n.name for n in
                 tf.compat.v1.get_default_graph().as_graph_def().node]
    else:
        nodes = [n.name for n in graph_def.node]
    print('nodes')
    pprint.pprint(nodes)


def load_mnist_data(start_batch=0, batch_size=10000):
    """Returns MNIST data in one-hot form"""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = tf.compat.v1.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.compat.v1.keras.utils.to_categorical(y_test, num_classes=10)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0

    x_test = x_test[start_batch:start_batch + batch_size]
    y_test = y_test[start_batch:start_batch + batch_size]

    return (x_train, y_train, x_test, y_test)


def load_pb_file(filename):
    """"Returns the graph_def from a saved protobuf file"""
    if not os.path.isfile(filename):
        raise Exception("File, " + filename + " does not exist")

    with tf.io.gfile.GFile(filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    print("Model restored")
    return graph_def


# https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/
def freeze_session(session,
                   keep_var_names=None,
                   output_names=None,
                   clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import (
        convert_variables_to_constants,
        remove_training_nodes,
    )

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(
                keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        print_nodes(input_graph_def)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        frozen_graph = remove_training_nodes(frozen_graph)
        return frozen_graph


def save_model(sess, output_names, directory, filename):
    frozen_graph = freeze_session(sess, output_names=output_names)
    # print_nodes(frozen_graph)
    path = tf.io.write_graph(frozen_graph, directory, filename + ".pb",
                             as_text=False)
    print("Model saved to: %s" % filename + ".pb")
    print(path)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("on", "yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("off", "no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch Size")

    return parser


def client_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument('--final_call', type=int, default=0,
                        help='always 0 unless final call to client.')
    parser.add_argument(
        "--hostname", type=str, default="localhost", help="Hostname of server")
    parser.add_argument(
        "--port", type=int,
        default=DEFAULT_PORT,
        help="Ports of server")
    parser.add_argument(
        "--encrypt_data_str",
        type=str,
        default="encrypt",
        help='"encrypt" to encrypt client data, "plain" to not encrypt',
    )
    parser.add_argument(
        "--tensor_name",
        type=str,
        default="import/Placeholder",
        help="Input tensor name")
    parser.add_argument(
        "--start_batch", type=int, default=0, help="Test data start index")
    parser.add_argument('--model_file', type=str, default='',
                        help="Filename of saved protobuf model")
    parser.add_argument('--n_parties', type=int, required=True)
    parser.add_argument(
        '--r_star',
        nargs='+',
        type=float,
        default=None,
        help="""For debug purposes: Each AP subtracts a vector of random numbers r* from the logits r (this is done via the homomorphic encryption). The encrypted result (r - r*) is sent back to the QP (client). When QP decrypts the received result, it obtains (r - r*) in plain text (note that this is not the plain result r). We can verify that this was done correctly by computing (r - r*) + r* = r."""
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="Enable the debug mode.")
    parser.add_argument(
        '--round_exp',
        type=int,
        default=None,
        help='Multiply r* and logits by 2^round_exp.'
    )
    parser.add_argument(
        "--predict_labels_file",
        type=str,
        default=None,
        help="The path to the file with numpy array with predicted labels on the clean model and input data for a given client.")
    parser.add_argument(
        '--dataset_path', type=str, default="", help='dataset to use.')
    parser.add_argument('--dataset_name', type=str, default='svhn',
                        help='name of dataset where queries came from.')
    parser.add_argument(
        '--from_pytorch', type=int, default=0,
        help='set to 1 to use pytorch bridge')
    parser.add_argument(
        '--minibatch_id', type=int, default=0,
        help='which index in the minibatch to work on.'
    )
    parser.add_argument('--log_timing_file', type=str,
                        help='name of the global log timing file',
                        default=f'log-timing-{get_timestamp()}.log')
    return parser


def server_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument('--dataset_name', type=str, default='svhn',
                        help='name of dataset where queries came from.')
    parser.add_argument(
        '--dataset_path', type=str, default="", help='dataset to use.')
    parser.add_argument(
        '--minibatch_id', type=int, default=0,
        help='which index in the minibatch to work on.'
    )
    parser.add_argument(
        '--from_pytorch', type=int, default=0,
        help='set to 1 to use pytorch bridge')
    parser.add_argument(
        "--enable_client",
        type=str2bool,
        default=False,
        help="Enable the client")
    parser.add_argument(
        "--enable_gc",
        type=str2bool,
        default=False,
        help="Enable garbled circuits")
    parser.add_argument(
        "--mask_gc_inputs",
        type=str2bool,
        default=False,
        help="Mask garbled circuits inputs",
    )
    parser.add_argument(
        "--mask_gc_outputs",
        type=str2bool,
        default=False,
        help="Mask garbled circuits outputs",
    )
    parser.add_argument(
        "--num_gc_threads",
        type=int,
        default=1,
        help="Number of threads to run garbled circuits with",
    )
    parser.add_argument(
        "--backend", type=str, default="HE_SEAL", help="Name of backend to use")
    parser.add_argument(
        "--encryption_parameters",
        type=str,
        default="",
        help=
        "Filename containing json description of encryption parameters, or json description itself",
    )
    parser.add_argument(
        "--encrypt_server_data",
        type=str2bool,
        default=False,
        help=
        "Encrypt server data (should not be used when enable_client is used)",
    )
    parser.add_argument(
        "--pack_data",
        type=str2bool,
        default=True,
        help="Use plaintext packing on data")
    parser.add_argument(
        "--start_batch", type=int, default=0, help="Test data start index")
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Filename of saved protobuf model")
    parser.add_argument(
        "--input_node",
        type=str,
        default="input:0",
        help="Tensor name of data input",
    )
    parser.add_argument(
        "--output_node",
        type=str,
        default="local__model/dense_1/BiasAdd:0",
        help="Tensor name of model output",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number for the server",
    )
    parser.add_argument(
        '--models_path', type=str, default='',
        help='dir containing all models')
    parser.add_argument(
        '--n_parties', type=int, default=1,
        help='n_parties in this. n_files = n_parties + 1')
    parser.add_argument(
        '--r_star',
        nargs='+',
        type=float,
        # default=np.zeros(10),
        default=None,
        # default=[-1.0],
        help="""Each AP subtracts a vector of random numbers r* from the logits r (this is done via the homomorphic encryption). The encrypted result (r - r*) is sent back to the QP (client). When QP decrypts the received result, it obtains (r - r*) in plain text (note that this is not the plain result r). We can verify that this was done correctly by computing (r - r*) + r* = r."""
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of possible classes in the classification task.",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="Enable the debug mode.")
    parser.add_argument(
        "--max_logit",
        type=float,
        default=100,
        help='The maximum value of a logit.',
    )
    parser.add_argument(
        "--rstar_exp",
        type=int,
        default=10,
        help='The exponent for 2 to generate the random r* from.',
    )
    parser.add_argument(
        '--round_exp',
        type=int,
        default=None,
        help='Multiply r* and logits by 2^round_exp.'
    )
    parser.add_argument(
        '--log_timing_file',
        type=str,
        help='name of the global log timing file',
        default=f'log-timing-{get_timestamp()}.log')

    return parser


def server_config_from_flags(FLAGS, tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    server_config.parameter_map["device_id"].s = b""
    server_config.parameter_map[
        "encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    server_config.parameter_map["enable_client"].s = str(
        FLAGS.enable_client).encode()
    server_config.parameter_map["enable_gc"].s = (str(FLAGS.enable_gc)).encode()
    server_config.parameter_map["mask_gc_inputs"].s = (str(
        FLAGS.mask_gc_inputs)).encode()
    server_config.parameter_map["mask_gc_outputs"].s = (str(
        FLAGS.mask_gc_outputs)).encode()
    server_config.parameter_map["num_gc_threads"].s = (str(
        FLAGS.num_gc_threads)).encode()
    server_config.parameter_map["port"].s = (str(FLAGS.port)).encode()

    if FLAGS.enable_client:
        server_config.parameter_map[tensor_param_name].s = b"client_input"
    elif FLAGS.encrypt_server_data:
        server_config.parameter_map[tensor_param_name].s = b"encrypt"

    if FLAGS.pack_data:
        server_config.parameter_map[tensor_param_name].s += b",packed"

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config


def from_labels_to_one_hot(a, num_classes):
    """
    Transform array of labels into 2-d one-hot encoding.

    >>> a = np.array([1, 0, 3])
    >>> c = from_labels_to_one_hot(a=a, num_classes=4)
    >>> # print("c: ", c)
    >>> desired = np.array([[ 0.,  1.,  0.,  0.], [ 1.,  0.,  0.,  0.], [ 0.,  0.,  0.,  1.]])
    >>> np.testing.assert_equal(actual=c, desired=desired)
    """
    # one-hot scaffolding of the result filled with zeros
    b = np.zeros((a.size, num_classes))
    # index which positions in b should be 1
    b_rows = np.arange(a.size)
    b[b_rows, a] = 1
    return b


def expx(x, nr_terms: int):
    """
    Compute approximated e^x.

    :param x: the input for e^x
    :param nr_terms: number of terms in the Taylor expansion
    :return: the Taylor series approximation of e^x

    >>> x = np.array([1.0, 2.0, 3.0])
    >>> desired = np.exp(x)
    >>> actual = expx(x, nr_terms=16)
    >>> # print(actual, desired)
    >>> np.testing.assert_allclose(actual=actual, desired=desired)
    """
    powx = 1.0  # power of x: x**i
    fact = 1  # factorial(i)
    result = 1.0
    for i in range(1, nr_terms):
        powx = tf.multiply(powx, x)
        fact *= i
        result = tf.add(result, tf.multiply(powx, 1 / fact))
    return result


def softmax(x):
    """
    Softmax function.

    :param x: the input array of real values
    :return: the softmax of x
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> maxx = softmax(x)
    >>> # print('maxx: ', maxx)
    >>> actual = maxx
    >>> desired = np.array([0.09003057, 0.24472847, 0.66524096])
    >>> np.testing.assert_almost_equal(actual=actual, desired=desired)
    """
    ex = np.exp(x)
    return ex / np.sum(ex)


def approximate_softmax(x, nr_terms: int = 16):
    """
    Approximate the softmax function with polynomial functions.
    We use the Taylor expansion.

    :param x: an array of real numbers (e.g., logits)
    :param nr_terms: number of terms in the Taylor expansion
    :return: the approximate softmax

    >>> x = np.array([1.0, 2.0, 3.0])
    >>> soft_max = approximate_softmax(x)
    >>> # print('soft_max: ', soft_max)
    >>> rsum = tf.reduce_sum(soft_max)
    >>> np.testing.assert_almost_equal(rsum, 1.0)
    >>> # print('sum: ', sum)
    >>> # desired = np.array([0.09003058, 0.24472849, 0.66524094])
    >>> desired = softmax(x)
    >>> np.testing.assert_almost_equal(actual=soft_max, desired=desired)
    """
    xexp = expx(x, nr_terms=nr_terms)
    sum = tf.reduce_sum(xexp)
    return tf.multiply(xexp, 1 / sum)


def scale_01(x):
    """
    Scale element of x to the 0 1 level.
    Preserve the order of the elments and they < order.

    :param x: an array of elements
    :return: elments scaled to 0 1 range

    >>> x = np.array([1.0, 2.0, 3.0])
    >>> scaled = scale_01(x)
    >>> # print('scaled: ', scaled)
    >>> desired = x / np.sum(x)
    >>> np.testing.assert_almost_equal(actual=scaled, desired=desired)
    """
    sum = tf.reduce_sum(x)
    return tf.divide(x, sum)


if __name__ == "__main__":
    import doctest

    tf.compat.v1.enable_eager_execution()

    # doctest.testmod()
    FLAGS, unparsed = server_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)
    # print('r_star: ', FLAGS.r_star)
    # print('is r_star [-1.0]', FLAGS.r_star == [-1.0])
