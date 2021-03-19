import numpy as np

from mnist_util import load_mnist_data


def round_array(x, exp):
    """
    Supports exp of every elem and rounds to arbitrary large int values.

    :param x: input array
    :param exp: the exponent used
    :return: the rounded values

    >>> a = round_array([0.28, 0.32], exp=10)
    >>> print(a)
    >>> b = [2, 3]
    >>> np.testing.assert_almost_equal(actual=a, desired=b)
    """
    # return np.array([int(elem * 2 ** exp) for elem in x])
    exp = 0
    shape = x.shape
    x = x.flatten()
    x = np.array([int(elem * 2 ** exp) for elem in x])
    return x.reshape(shape)


def print_array(x):
    """
    Print array x.
    :param x: input array
    :return: print array x
    """
    if len(x.shape) > 1:
        print('[')
        for xi in x:
            print_array(xi)
        print('],')
    else:
        print("[", ",".join([str(elem) for elem in x]), "],")


def array_str(x, out=""):
    """
    Print array x.
    :param x: input array
    :return: print array x
    """
    if len(x.shape) > 1:
        out += '['
        for xi in x:
            out = array_str(x=xi, out=out)
            out += ","
        out = out[:-1]  # remove last comma
        out += ']'
    else:
        out += "["
        out += ",".join([str(elem) for elem in x])
        out += "]"

    return out


def get_data(name, party_id, n_parties, FLAGS):
    if name == 'mnist':
        (x_train, y_train, x_test, y_test) = load_mnist_data(
            FLAGS.start_batch, FLAGS.batch_size)
        leftover = len(x_train) % (n_parties + 1)
        x_train, y_train = x_train[:-leftover], y_train[:-leftover]
        leftover = len(x_test) % (n_parties + 1)
        x_test, y_Test = x_test[:-leftover], y_test[:-leftover]
        train_indices = np.arange(len(x_train))
        # TODO: party_t* variables below are not used.
        party_train_indices = np.split(train_indices, n_parties)[party_id]
        test_indices = np.arange(len(x_test))
        party_test_indices = np.split(test_indices, n_parties)[party_id]
        return (x_train[train_indices], y_train[train_indices]), (
            x_test[test_indices], y_test[test_indices])
    else:
        raise ValueError(f"Invalid dataset name: {name}.")


if __name__ == "__main__":
    print('1 dim')
    x = np.random.uniform(low=0, high=10, size=(4,))
    print('x: ', x)
    print(array_str(x))

    print('2 dim')
    x = np.random.uniform(low=0, high=10, size=(2, 4))
    print('x: ', x)
    print(array_str(x))

    print('3 dim')
    x = np.random.uniform(low=0, high=10, size=(2, 3, 4))
    print('x: ', x)
    print(array_str(x))