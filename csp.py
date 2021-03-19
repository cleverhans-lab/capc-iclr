import numpy as np
import threading
import os


class start_party(threading.Thread):
    def __init__(self, thread_name, cmd):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.cmd = cmd

    def run(self):
        print(self.thread_name)
        os.system(self.cmd)


def read_array(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(int(line.strip()))

    return np.array(data)


def write_array(filename, array):
    with open(filename, 'w') as writer:
        for element in array:
            writer.write(str(element) + '\n')


def sum_files(filenames):
    for i, filename in enumerate(filenames):
        if i == 0:
            sum = read_array(filename)
        else:
            sum = sum + read_array(filename)
    return sum


def call_parties(client_filename, csp_filename, output_filename):
    try:
        client_thread = start_party('Thread-1',
                                    './gc-emp-test/bin/sum_histogram 1 12345 {} {}'.format(
                                        client_filename, output_filename))
        client_thread.start()
        csp_thread = start_party('Thread-2',
                                 './gc-emp-test/bin/sum_histogram 2 12345 {} {}'.format(
                                     csp_filename, output_filename))
        csp_thread.start()

        client_thread.join()
        csp_thread.join()
    except:
        print('Thread Error: sum histogram failed')


def get_histogram(
        client_filename,
        csp_filenames,
        csp_sum_filename,
        output_filename):
    csp_sum = sum_files(csp_filenames)
    csp_sum = (csp_sum + np.random.laplace(size=len(csp_sum))).astype(int)
    write_array(csp_sum_filename, csp_sum)

    call_parties(client_filename, csp_sum_filename, output_filename)

    index = read_array(output_filename)
    return index
