import numpy as np
import os
import pyhe_client
import subprocess
import time

from consts import out_client_name, out_final_name, inference_times_name
from mnist_util import client_argument_parser
from utils import client_data
from utils.main_utils import array_str
from utils.time_utils import log_timing
from utils.main_utils import round_array


def run_client(FLAGS, data):
    port = FLAGS.port
    if isinstance(port, list) or isinstance(port, tuple):
        print("WARNING: list ports were passed. Only one should be passed.")
        port = port[0]  # only one port should be passed
    if FLAGS.batch_size > 1:
        raise ValueError('batch size > 1 not currently supported.')
    inference_start = time.time()
    client = pyhe_client.HESealClient(
        FLAGS.hostname,
        port,
        FLAGS.batch_size,
        {
            "import/input"
            # FLAGS.tensor_name
            : (FLAGS.encrypt_data_str, data)},
    )
    print(f"data shape: {data.shape}")
    r_rstar = np.array(client.get_results())
    inference_end = time.time()
    print(f"Inference time: {inference_end - inference_start}s")
    with open(inference_times_name, 'a') as outfile:
        outfile.write(str(inference_end - inference_start))
        outfile.write('\n')
    print('r_rstar (r-r*): ', array_str(r_rstar))

    rstar = FLAGS.r_star
    if rstar is None:
        raise ValueError('r_star should be provided but was None.')

    r_rstar = round_array(x=r_rstar, exp=FLAGS.round_exp)
    print('rounded r_rstar (r-r*): ', array_str(r_rstar))
    print("Writing out logits file to txt.")
    with open(f'{out_client_name}{port}privacy.txt', 'w') as outfile:
        for val in r_rstar.flatten():
            outfile.write(f"{int(val)}\n")

    # do 2 party computation with each Answering Party
    msg = 'starting 2pc with Answering Party'
    print(msg)
    log_timing(stage='client:' + msg,
               log_file=FLAGS.log_timing_file)
    # completed = {port: False for port in flags.ports}
    max_t = time.time() + 100000
    while not os.path.exists(f"{out_final_name}{port}privacy.txt"):
        print(f'client starting 2pc with port: {port}')
        process = subprocess.Popen(
            ['./gc-emp-test/bin/argmax_1', '2', '12345',
             f'{out_client_name}{port}privacy.txt'])
        time.sleep(1)
        if time.time() > max_t:
            raise ValueError("Step 1' of protocol never finished. Issue.")
    log_timing(stage='client:finished 2PC',
               log_file=FLAGS.log_timing_file)
    return r_rstar, rstar

    # print("Prepping for 2pc with CSP")
    #
    # r_rstars = []
    # for port in flags.ports:
    #     with open(f'output{port}privacy.txt', 'r') as infile:
    #         r_rstar = []
    #         for line in infile:
    #             r_rstar.append(int(line))
    #         r_rstars.append(r_rstar)
    # r_rstars = np.array(r_rstars, np.int64)
    # print(r_rstars)
    # print('done')
    #
    # if flags.final_call:
    #     fs = [f"output{port}privacy.txt" for port in flags.ports]
    #     array_sum = csp.sum_files(fs)
    #     print(array_sum)
    #     with open("output.txt", 'w') as outfile:
    #         for v in array_sum.flatten():
    #             outfile.write(f'{v}\n')
    #     csp_filenames = [f'noise{port}privacy.txt' for port in flags.ports]
    #     label = csp.get_histogram(
    #         client_filename='output.txt',
    #         csp_filenames=csp_filenames,
    #         csp_sum_filename='final.txt')
    #     print(label)


if __name__ == "__main__":
    FLAGS, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)

    if FLAGS.from_pytorch:
        queries, labels, noisies = client_data.load_data(FLAGS.dataset_path)
        query = queries[FLAGS.minibatch_id].transpose()
        label = labels[FLAGS.minibatch_id]
        noisy = noisies[FLAGS.minibatch_id]
        (x_train, y_train, x_test, y_test) = client_data.load_mnist_data(0, 1)
        query = x_test
    else:
        # (x_train, y_train), (x_test, y_test) = client_data.get_dataset(
        #     FLAGS.dataset)
        # query = x_test
        raise ValueError('must be from pytorch')

    start_time = time.time()
    print(query.shape)
    r_rstar, rstar = run_client(FLAGS=FLAGS, data=query[None, ...].flatten("C"))
    end_time = time.time()
    print(f'step 1 runtime: {end_time - start_time}s')
    log_timing('client_server:finish', log_file=FLAGS.log_timing_file)
    # Check if stage 1 was executed correctly.
    # if FLAGS.predict_labels_file is not None:
    #     port = FLAGS.ports[0]
    #     predict_labels_file = FLAGS.predict_labels_file + str(
    #         port) + '.npy'
    #     predict_labels = np.load(predict_labels_file)
    #     check_rstar_stage1(
    #         rstar=rstar,
    #         r_rstar=r_rstar,
    #         labels=predict_labels,
    #         port=port,
    #     )

    # y_labels = labels.argmax(axis=1)
    # print("y_test: ", y_labels)
    #
    # y_pred = y_pred_reshape.argmax(axis=1)
    # print("y_pred: ", y_pred)
    #
    # correct = np.sum(np.equal(y_pred, y_labels))
    # acc = correct / float(flags.batch_size)
    # print("correct from original result: ", correct)
    # print(
    #     "Accuracy original result (batch size", flags.batch_size, ") =",
    #     acc * 100.0, "%")
