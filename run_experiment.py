"""
This script assumes that a subdir with name {n_parties} exists in /models with the model files stored here.
The number of model files should equal the value of {n_parties} + 1.
It kicks off a server for each answering party and a single client who will be requesting queries.
client.py holds the clients training protocol, and server.py the response algorithms.
train_inits.py should be run first to train each model on a separate partition and save them as per the required scheme.
USAGE: call this file with: OMP_NUM_THREADS=24 NGRAPH_HE_VERBOSE_OPS=all NGRAPH_HE_LOG_LEVEL=3 python run_experiment.py
SETUP: create a tmux session with 3 panes, each in /home/dockuser/code/capc
"""

import warnings

from utils.client_data import get_data
from utils.time_utils import get_timestamp, log_timing

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import os
import time
import numpy as np
import atexit
import libtmux
from utils.remove_files import remove_files_by_name
import consts
from consts import out_client_name, out_server_name, out_final_name
import getpass
import get_r_star


def get_FLAGS():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--session', type=str, help='session name',
                        default='capc')
    parser.add_argument('--log_timing_file', type=str,
                        help='name of the global log timing file',
                        default=f'log-timing-{get_timestamp()}.log')
    parser.add_argument('--n_parties', type=int, default=1,
                        help='number of servers')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed for top level script')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset.')
    parser.add_argument(
        "--rstar_exp",
        type=int,
        default=10,
        help='The exponent for 2 to generate the random r* from.',
    )
    parser.add_argument(
        "--max_logit",
        type=float,
        default=36.0,
        help='The maximum value of a logit.',
    )
    parser.add_argument(
        "--user",
        type=str,
        default=getpass.getuser(),
        help="The name of the OS USER.",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=0,
        help='log level for he-transformer',
    )
    parser.add_argument(
        '--round_exp',
        type=int,
        default=3,
        help='Multiply r* and logits by 2^round_exp.'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=20,
        help='Number of threads.',
    )
    parser.add_argument(
        '--qp_id', type=int, default=0, help='which model is the QP?')
    parser.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="Test data start index")
    parser.add_argument(
        "--model_type",
        type=str,
        default='cryptonets-relu',
        help="The type of models used.",
    )
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
        '--dataset_path', type=str,
        default='/home/dockuser/queries',
        help='where the queries are.')
    parser.add_argument(
        '--dataset_name', type=str,
        default='svhn',
        help='name of dataset where queries came from')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--n_queries',
                        type=int,
                        default=1353,
                        help='total len(queries)')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/dockuser/checkpoints',
                        help='dir with all checkpoints')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='set to use cpu and no encryption.')
    parser.add_argument('--ignore_parties', default=False, action='store_true',
                        help='set when using crypto models.')
    parser.add_argument('--encryption_params',
                        default='$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L5_gc.json')
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)
    return FLAGS


def clean_old_files():
    for name in [out_client_name,
                 out_server_name,
                 out_final_name,
                 consts.input_data,
                 consts.input_labels,
                 consts.predict_labels]:
        remove_files_by_name(starts_with=name)


# Provide data.
def set_data_labels(FLAGS):
    data, labels = get_data(start_batch=FLAGS.start_batch,
                            batch_size=FLAGS.batch_size)
    np.save(consts.input_data, data)
    np.save(consts.input_labels, labels)


def get_models(model_dir, n_parties, ignore_parties):
    model_files = [f for f in os.listdir(model_dir) if
                   os.path.isfile(os.path.join(model_dir, f))]
    if len(model_files) != n_parties and not ignore_parties:
        raise ValueError(
            f'{len(model_files)} models found when {n_parties + 1} parties requested. Not equal.')
    return model_dir, model_files


def run(FLAGS):
    log_timing_file = FLAGS.log_timing_file
    log_timing('main: start capc', log_file=log_timing_file)

    server = libtmux.Server()
    session = server.find_where({"session_name": FLAGS.session})
    # session = server.list_sessions()[0]
    window = session.attached_window
    # window = session.new_window(attach=True, window_name="run_experiment")
    # window.split_window(attach=False)
    #
    processes = []

    def kill_processes():
        for p in processes:
            p.kill()

    # def kill_window():
    #   window.kill_window()
    #
    if not FLAGS.debug:
        # atexit.register(kill_window)
        atexit.register(kill_processes)

    n_parties = FLAGS.n_parties
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    rstar_exp = FLAGS.rstar_exp
    log_level = FLAGS.log_level
    round_exp = FLAGS.round_exp
    num_threads = FLAGS.num_threads
    input_node = FLAGS.input_node
    output_node = FLAGS.output_node
    backend = 'HE_SEAL' if not FLAGS.cpu else 'CPU'

    models_loc, model_files = get_models(
        FLAGS.checkpoint_dir, n_parties=n_parties,
        ignore_parties=FLAGS.ignore_parties)
    for port in range(37000, 37000 + n_parties):
        files_to_delete = [consts.out_client_name + str(port) + 'privacy.txt']
        files_to_delete += [consts.out_final_name + str(port) + 'privacy.txt']
        files_to_delete += [consts.out_server_name + str(port) + 'privacy.txt']
        files_to_delete += [f"{out_final_name}privacy.txt",
                            f"{out_server_name}privacy.txt"]  # aggregates across all parties
        files_to_delete += [consts.inference_times_name,
                            consts.argmax_times_name,
                            consts.client_csp_times_name,
                            consts.inference_no_network_times_name]
        for f in files_to_delete:
            if os.path.exists(f):
                print(f'delete file: {f}')
                os.remove(f)
    for query_num in range(FLAGS.n_queries):
        for port, model_file in zip(
                [37000 + int(i + query_num * n_parties) for i in
                 range(n_parties)],
                model_files):
            print(f"port: {port}")
            full_model_file = fr'{models_loc}/{model_file}'
            full_model_file_new = ""
            for s in full_model_file:
                if s == '(' or s == ')':
                    full_model_file_new += "\\"
                full_model_file_new += s
            full_model_file = full_model_file_new
            pane = window.select_pane('1')
            # cmd = " ".join(['python', 'configure_model_to_graph.py', full_model_file, str(port)])
            # pane.send_keys(cmd)
            # time.sleep(2)
            new_model_file = os.path.join("/home/dockuser/models",
                                          str(port) + ".pb")
            # Compute the predicted labels for tests.

            # predict_labels = get_predict_labels(
            #     model_file=full_model_file,
            #     input_node=input_node,
            #     output_node=output_node,
            #     input_data=np.load(consts.input_data))
            # np.save(file=consts.predict_labels + str(port) + '.npy',
            #         arr=predict_labels)
            r_star = get_r_star.get_rstar_server(
                max_logit=FLAGS.max_logit,
                batch_size=batch_size,
                num_classes=num_classes,
                exp=FLAGS.rstar_exp,
            ).flatten()
            print(f"run_exp rstar: {r_star}")
            # pane = window.select_pane('2')
            print(f"port: {port}")
            print('Start the servers (answering parties: APs)')
            log_timing('start server (AP)', log_file=log_timing_file)
            cmd = [
                      # f'OMP_NUM_THREADS={num_threads}',
                      # f'NGRAPH_HE_LOG_LEVEL={log_level}',
                      'python', 'server_client.py',
                      '--backend', backend,
                      '--model_file', new_model_file,
                      '--dataset_name', FLAGS.dataset_name,
                      '--encryption_parameters',
                      FLAGS.encryption_params,
                      '--enable_client', 'true',
                      '--enable_gc', 'true',
                      '--mask_gc_inputs', 'true',
                      '--mask_gc_outputs', 'true', '--from_pytorch', '1',
                      '--dataset_name', FLAGS.dataset_name,
                      '--dataset_path', FLAGS.dataset_path,
                      '--num_gc_threads', f'{num_threads}',
                      '--input_node', f'{input_node}',
                      '--output_node', f'{output_node}', '--minibatch_id',
                      f'{query_num}',
                      '--rstar_exp', f'{rstar_exp}',
                      '--num_classes', f'{num_classes}',
                      '--round_exp', f'{round_exp}',
                      '--log_timing_file', log_timing_file,
                      "--r_star"] + [str(x) for x in r_star] + [
                      '--port', f'{port}',
                  ]
            cmd_string = " ".join([
                                      f'OMP_NUM_THREADS={num_threads}',
                                      f'NGRAPH_HE_LOG_LEVEL={log_level}',
                                      'python', 'server_client.py',
                                      '--backend', backend,
                                      '--model_file', new_model_file,
                                      '--dataset_name', FLAGS.dataset_name,
                                      '--encryption_parameters',
                                      FLAGS.encryption_params,
                                      '--enable_client', 'true',
                                      '--enable_gc', 'true',
                                      '--mask_gc_inputs', 'true',
                                      '--mask_gc_outputs', 'true',
                                      '--from_pytorch', '1', '--dataset_name',
                                      FLAGS.dataset_name,
                                      '--dataset_path', FLAGS.dataset_path,
                                      '--num_gc_threads', f'{num_threads}',
                                      '--input_node', f'{input_node}',
                                      '--output_node', f'{output_node}',
                                      '--minibatch_id', f'{query_num}',
                                      '--rstar_exp', f'{rstar_exp}',
                                      '--num_classes', f'{num_classes}',
                                      '--round_exp', f'{round_exp}',
                                      '--log_timing_file', log_timing_file,
                                      "--r_star"] + [str(x) for x in r_star] + [
                                      '--port', f'{port}',
                                  ])
            # process1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            # processes.append(process1)
            # out1 = process1.communicate(timeout=30)
            pane.send_keys(cmd_string)
            if not FLAGS.cpu:
                time.sleep(1)
                print(f"port: {port}")
                # Start the client (the querying party: QP).
                log_timing('start QP', log_file=log_timing_file)
                pane = window.select_pane('2')
                cmd = [
                          # f'OMP_NUM_THREADS={num_threads}',
                          # f'NGRAPH_HE_LOG_LEVEL={log_level}',
                          'python', 'client_server.py',
                          '--batch_size', f'{batch_size}',
                          '--encrypt_data_str', 'encrypt',
                          '--n_parties', f'{n_parties}',
                          '--round_exp', f'{round_exp}',
                          '--from_pytorch', '1',
                          '--minibatch_id', f'{query_num}',
                          '--dataset_path', f'{FLAGS.dataset_path}',
                          '--port', f'{port}',
                          '--log_timing_file', log_timing_file,
                          '--dataset_name', FLAGS.dataset_name, '--r_star'
                      ] + [str(x) for x in r_star]
                cmd_string = " ".join([
                                          f'OMP_NUM_THREADS={num_threads}',
                                          f'NGRAPH_HE_LOG_LEVEL={log_level}',
                                          'python', 'client_server.py',
                                          '--batch_size', f'{batch_size}',
                                          '--encrypt_data_str', 'encrypt',
                                          '--n_parties', f'{n_parties}',
                                          '--round_exp', f'{round_exp}',
                                          '--from_pytorch', '1',
                                          '--minibatch_id', f'{query_num}',
                                          '--dataset_path',
                                          f'{FLAGS.dataset_path}',
                                          '--port', f'{port}',
                                          '--dataset_name', FLAGS.dataset_name,
                                          '--log_timing_file', log_timing_file,
                                          '--r_star'
                                      ] + [str(x) for x in r_star])
                # print(cmd)
                # time.sleep(2)
                # process2 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                # processes.append(process2)
                # out1 = process1.communicate(timeout=60)
                # out2 = process2.communicate(timeout=60)
                # print(out1)
                # print(out2)
                # process2.wait()
                pane.send_keys(cmd_string)
                time.sleep(70)
                # break
                # if port == 37002:
                #   break
            else:
                time.sleep(1)

        pane = window.select_pane('1')
        log_timing('start csp', log_file=log_timing_file)
        cmd = ['python', 'client_csp.py',
               f'{37000 + int(query_num * n_parties)}',
               # f'{37000 + 1}',
               f'{37000 + int(query_num * n_parties) + n_parties}'
               ]
        cmd_string = " ".join(['python', 'client_csp.py',
                               f'{37000 + int(query_num * n_parties)}',
                               # '37000',
                               # f'{37000 + 1}',
                               f'{37000 + int(query_num * n_parties) + n_parties}'
                               ])
        # process3 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        # out3 = process3.communicate()
        # processes.append(process3)
        # print(out3)

        pane.send_keys(cmd_string)
        # break
        time.sleep(10)
        if query_num >= 0:
            break
        # TODO: call final script here for CSP - client stuff which will get called once per query. (batch size == 1)

    log_timing('finish capc', log_file=log_timing_file)


if __name__ == "__main__":
    FLAGS = get_FLAGS()
    np.random.seed(FLAGS.seed)
    clean_old_files()
    set_data_labels(FLAGS=FLAGS)
    run(FLAGS=FLAGS)
