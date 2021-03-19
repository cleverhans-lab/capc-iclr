import tensorflow as tf
import numpy as np

# Add parent directory to path
from mnist_util import (
    load_pb_file,
    print_nodes,
)


def get_predict_labels(model_file, input_node, output_node, input_data):
    # Load saved model
    tf.import_graph_def(load_pb_file(model_file))

    print(f"predict labels - loaded model from file: {model_file}")
    print_nodes()

    # Get input / output tensors
    x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
        input_node)
    y_output = tf.compat.v1.get_default_graph().get_tensor_by_name(
        output_node)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        predicted_labels = sess.run(y_output,
                                    feed_dict={x_input: input_data})

    return np.argmax(predicted_labels)


if __name__ == "__main__":
    from mnist.example import x_test as mnist_x_test
    from mnist.example import y_test as mnist_y_test
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_node",
        type=str,
        default="import/input:0",
        help="Tensor name of data input",
    )
    parser.add_argument(
        "--output_node",
        type=str,
        default="import/output/BiasAdd:0",
        help="Tensor name of model output",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="./models/cryptonets-relu.pb",
        help="Filename of saved protobuf model")
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)

    # Get input / output tensors
    input_node = FLAGS.input_node
    output_node = FLAGS.output_node
    model_file = FLAGS.model_file
    input_data = mnist_x_test.reshape((1, 28, 28, 1))

    predic_labels = get_predict_labels(model_file=model_file,
                                       input_node=input_node,
                                       output_node=output_node,
                                       input_data=input_data)

    correct_labels = np.argmax(mnist_y_test)
    print('correct labels: ', correct_labels)
    print('predict labels: ', predic_labels)
    np.testing.assert_equal(actual=predic_labels,
                            desired=correct_labels)
