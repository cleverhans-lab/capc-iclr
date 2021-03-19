import numpy as np
from utils.main_utils import array_str


def check_rstar_file_stage1(rstar_file, r_rstar, labels, port):
    rstar = np.genfromtxt(rstar_file)
    return check_rstar_stage1(rstar=rstar, r_rstar=r_rstar, labels=labels,
                              port=port)


def check_rstar_stage1(rstar, r_rstar, labels, port):
    if len(labels.shape) > 1:
        # change from one-hot encoding to labels
        labels = labels.argmax(axis=1)

    correct, pred_r = is_correct_pred(r_rstar=r_rstar, rstar=rstar,
                                      labels=labels)
    if correct:
        print(f'expected label(s) for port {port}: ', labels)
        print('r_rstar label(s): ', pred_r)
        print('stage 1 correct')
    else:
        print('rstar: ', array_str(rstar))
        print('r_rstar: ', array_str(r_rstar))
        print(f'expected label(s) for port {port}: ', labels, ' type: ',
              type(labels))
        print('r_rstar label(s): ', pred_r, ' type: ', type(pred_r))
        raise Exception(f'Failed stage 1 for port: {port}.')


def is_correct_pred(r_rstar, rstar, labels):
    pred_r = np.argmax(r_rstar + rstar)
    # All labels are predicted correctly.
    correct = np.all(np.equal(pred_r, labels))
    return correct, pred_r


def test_is_correct_labels():
    rstar = np.genfromtxt('log_files/noise0.txt')
    r_rstar = np.genfromtxt('log_files/logits0.txt')
    result = is_correct_pred(r_rstar=r_rstar, rstar=rstar, labels=[7])
    print('is correct label: ', result)


if __name__ == "__main__":
    test_is_correct_labels()
