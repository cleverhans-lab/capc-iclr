import numpy as np


def get_rstar_server(max_logit, batch_size, num_classes, exp):
    r_star = max_logit + np.random.uniform(
        low=-2 ** exp, high=2 ** exp, size=(batch_size, num_classes))
    return r_star
