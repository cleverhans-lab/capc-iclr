# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import numpy as np
import pyhe_client
import time
import consts
from consts import out_client_name, out_server_name
from utils.main_utils import array_str
from utils.main_utils import round_array
from utils.check_rstar import check_rstar_file_stage1

from mnist.example import x_test as mnist_x_test
from mnist.example import y_test as mnist_y_test

from mnist_util import load_mnist_data, client_argument_parser
import os
import subprocess

import csp


def run_client(FLAGS, data=None, labels=None):
    if data is None:
        data = np.load(consts.input_data)
    if labels is None:
        labels = np.load(consts.input_labels)

    r_rstars = {}
    for i, port in enumerate(FLAGS.ports):

        client = pyhe_client.HESealClient(
            FLAGS.hostname,
            port,
            FLAGS.batch_size,
            {FLAGS.tensor_name: (FLAGS.encrypt_data_str, data)},
        )

        raw_results = np.array(client.get_results())
        print('raw results: ', array_str(raw_results))

        rstar = None
        if FLAGS.debug is True:
            if FLAGS.rstar is not None:
                raise Exception(
                    "Either debug or r_star or both flags have to be None.")
            raw_shape_0 = raw_results.shape[0]
            expected_shape_0 = 2 * FLAGS.batch_size
            if raw_shape_0 != expected_shape_0:
                raise Exception(
                    f'Expected r_star for each example in the batch'
                    f'and dim 0 size of the result: {expected_shape_0}'
                    f', but received result with dim 0 size:'
                    f' {raw_shape_0}')
            r_rstar = raw_results[:FLAGS.batch_size]
            rstar = raw_results[FLAGS.batch_size:]

        else:
            if FLAGS.rstar is None:
                rstar = None
            elif FLAGS.rstar == [-1.0]:
                raise Exception('We do not generate r_star in the client.'
                                'r_star provided is [-1.0].')
            if FLAGS.rstar is not None:
                rstar = np.array(FLAGS.rstar)
            r_rstar = raw_results

        print('r_rstar (r-r*): ', array_str(r_rstar))

        if FLAGS.round_exp:
            # r_rstar = (r_rstar * 2 ** FLAGS.round_exp).astype(np.int64)
            r_rstar = round_array(x=r_rstar, exp=FLAGS.round_exp)
            print('rounded r_rstar (r-r*): ', array_str(r_rstar))

        r_rstars[port] = r_rstar
        y_pred_reshape = np.array(r_rstar).reshape(FLAGS.batch_size, 10)

        y_labels = labels.argmax(axis=1)
        print("y_test: ", y_labels)

        y_pred = y_pred_reshape.argmax(axis=1)
        print("y_pred: ", y_pred)

        correct = np.sum(np.equal(y_pred, y_labels))
        acc = correct / float(FLAGS.batch_size)
        print("correct from original result: ", correct)
        print(
            "Accuracy original result (batch size", FLAGS.batch_size, ") =",
            acc * 100.0, "%")
        with open(f'{out_client_name}{port}privacy.txt', 'w') as outfile:
            for val in y_pred_reshape.flatten():
                outfile.write(f"{int(val)}\n")

        if rstar is not None:
            results_r = y_pred_reshape + rstar
            y_pred_r = results_r.argmax(axis=1)
            print('y_pred_r: ', y_pred_r)
            correct = np.sum(np.equal(y_pred_r, y_labels))
            acc = correct / float(FLAGS.batch_size)
            print("correct after adding r*: ", correct)
            print(
                "Accuracy after adding r* (batch size", FLAGS.batch_size, ") =",
                acc * 100.0, "%")

        print(port, "DONE----------------------")
        time.sleep(5)

    # do 2 party computation with each Answering Party
    print('starting 2pc')
    completed = {port: False for port in FLAGS.ports}
    n_parties = len(completed.keys())
    max_t = time.time() + 100000
    processes = []
    while sum(completed.values()) < n_parties and time.time() < max_t:
        for port in completed.keys():
            if not completed[port]:
                if not os.path.exists(f"{out_client_name}{port}privacy.txt"):
                    raise ValueError('something broke')
                out_server_file = f"{out_server_name}{port}privacy.txt"
                if os.path.exists(out_server_file):
                    if FLAGS.predict_labels_file is not None:
                        predict_labels_file = FLAGS.predict_labels_file + str(
                            port) + '.npy'
                        predict_labels = np.load(predict_labels_file)
                        check_rstar_file_stage1(
                            rstar_file=out_server_file,
                            r_rstar=r_rstars[port],
                            labels=predict_labels,
                            port=port,
                        )
                    print(f'client starting 2pc with port: {port}')
                    completed[port] = True
                    process = subprocess.Popen(
                        ['./gc-emp-test/bin/argmax_1', '2', '12345',
                         f'{out_client_name}{port}privacy.txt'])
                    process.wait()
                    processes.append(process)
                else:
                    print(
                        f'Expected output file {out_server_file} from the party {port} does not exist yet!')
    max_t = time.time() + 10000
    while any([p.poll() for p in
               processes]) is None:  # wait on all argmaxs to finish first
        time.sleep(1)
        if time.time() > max_t:
            raise ValueError(
                f'something broke while waiting on processes for 2pc with servers: {[p.poll() for p in processes]}')
    print("Prepping for 2pc with CSP")
    if not sum(completed.values()) == n_parties:
        raise ValueError('a 2pc with a server failed')

    r_rstars = []
    for port in FLAGS.ports:
        with open(f'output{port}privacy.txt', 'r') as infile:
            r_rstar = []
            for line in infile:
                r_rstar.append(int(line))
            r_rstars.append(r_rstar)
    r_rstars = np.array(r_rstars, np.int64)
    print(r_rstars)
    print('done')

    if FLAGS.final_call:
        fs = [f"output{port}privacy.txt" for port in FLAGS.ports]
        array_sum = csp.sum_files(fs)
        print(array_sum)
        with open("output.txt", 'w') as outfile:
            for v in array_sum.flatten():
                outfile.write(f'{v}\n')
        csp_filenames = [f'noise{port}privacy.txt' for port in FLAGS.ports]
        label = csp.get_histogram(
            client_filename='output.txt',
            csp_filenames=csp_filenames,
            csp_sum_filename='final.txt')
        print(label)


if __name__ == "__main__":
    FLAGS, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)

    (x_train, y_train, x_test, y_test) = load_mnist_data(
        FLAGS.start_batch, FLAGS.batch_size)

    is_test = False
    if is_test:
        data = mnist_x_test
        y_test = [mnist_y_test]
    else:
        data = x_test.flatten("C")
        # print('data (x_test): ', data)
        # print('y_test: ', y_test)

    run_client(FLAGS=FLAGS, data=data, labels=y_test)
