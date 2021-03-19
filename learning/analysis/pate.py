from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.stats
import math


def _logsumexp(x):
    """
    Sum in the log space.

    Args:
        x: array-like.

    Returns:
        A scalar.
    """
    x = np.array(x)
    m = max(x)  # for numerical stability
    return m + math.log(sum(np.exp(x - m)))


def _log1mexp(x):
    """
    Numerically stable computation of log(1-exp(x)).

    Args:
        x: a scalar.

    Returns:
        A scalar.
    """
    assert x <= 0, "Argument must be non-positive!"
    if x < -1:
        return math.log1p(-math.exp(x))
    elif x < 0:
        return math.log(-math.expm1(x))
    else:
        return -np.inf


def rdp_to_dp(orders, rdp_eps, delta):
    """
    Conversion from (lambda, eps)-RDP to conventional (eps, delta)-DP.

    Args:
        orders: an array-like list of RDP orders.
        rdp_eps: an array-like list of RDP guarantees (of the same length as orders).
        delta: target delta (a scalar).

    Returns:
        A pair of (dp_eps, optimal_order).
    """
    assert not np.isscalar(orders) and not np.isscalar(rdp_eps) and len(orders) == len(
        rdp_eps), "'orders' and 'rdp_eps' must be array-like and of the same length!"
    dp_eps = np.array(rdp_eps) - math.log(delta) / (np.array(orders) - 1)
    idx_opt = np.argmin(dp_eps)
    return dp_eps[idx_opt], orders[idx_opt]


###############################
# RDP FOR THE GNMAX MECHANISM #
###############################


def compute_logq_gnmax(votes, sigma):
    """
    Computes an upper bound on log(Pr[outcome != argmax]) for the GNMax mechanism.

    Args:
        votes: a 1-D numpy array of raw ensemble votes for a given query.
        sigma: std of the Gaussian noise in the GNMax mechanism.

    Returns:
        A scalar upper bound on log(Pr[outcome != argmax]) where log denotes natural logarithm.
    """
    num_classes = len(votes)
    variance = sigma ** 2
    idx_max = np.argmax(votes)
    votes_gap = votes[idx_max] - votes
    votes_gap = votes_gap[np.arange(num_classes) != idx_max]  # exclude argmax
    # Upper bound log(q) via a union bound.
    logq = _logsumexp(scipy.stats.norm.logsf(votes_gap, scale=math.sqrt(2 * variance)))
    return min(logq, math.log(1 - (1 / num_classes)))  # another obvious upper bound


def compute_rdp_data_independent_gnmax(sigma, orders):
    """
    Computes data-independent RDP gurantees for the GNMax mechanism.

    Args:
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    if sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError("'sigma' must be non-negative, 'orders' must be array-like, and all elements in 'orders' must be greater than 1!")
    variance = sigma ** 2
    return np.array(orders) / variance


def compute_rdp_data_dependent_gnmax(logq, sigma, orders):
    """
    Computes data-dependent RDP gurantees for the GNMax mechanism.

    Args:
        logq: a union bound on log(Pr[outcome != argmax]) for the GNMax mechanism.
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    if logq > 0 or sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError(
            "'logq' must be non-positive, 'sigma' must be non-negative, 'orders' must be array-like, and all elements in 'orders' must be greater than 1!")

    if np.isneginf(logq):  # deterministic mechanism with sigma == 0
        return np.full_like(orders, 0., dtype=np.float)

    variance = sigma ** 2
    orders = np.array(orders)
    rdp_eps = orders / variance  # data-independent bound as baseline

    # Two different higher orders computed according to Proposition 10.
    rdp_order2 = math.sqrt(variance * -logq)
    rdp_order1 = rdp_order2 + 1

    # Filter out entries to which data-dependent bound does not apply.
    mask = np.logical_and(rdp_order1 > orders, rdp_order2 > 1)

    # Corresponding RDP guarantees for the two higher orders.
    rdp_eps1 = rdp_order1 / variance
    rdp_eps2 = rdp_order2 / variance

    log_a2 = (rdp_order2 - 1) * rdp_eps2

    # Make sure that logq lies in the increasing range and that A is positive.
    if (np.any(mask) and -logq > rdp_eps2 and logq <= log_a2 - rdp_order2 *
            (math.log(1 + 1 / (rdp_order1 - 1)) + math.log(1 + 1 / (rdp_order2 - 1)))):

        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1mq = _log1mexp(logq)  # log1mq = log(1-q)
        log_a = (orders - 1) * (
            log1mq - _log1mexp((logq + rdp_eps2) * (1 - 1 / rdp_order2)))
        log_b = (orders - 1) * (rdp_eps1 - logq / (rdp_order1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s = np.logaddexp(log1mq + log_a, logq + log_b)
        rdp_eps[mask] = np.minimum(rdp_eps, log_s / (orders - 1))[mask]

    assert np.all(rdp_eps >= 0)
    return rdp_eps


def is_data_independent_rdp_always_opt_gnmax(num_teachers, num_classes, sigma, orders):
    """
    Tests whether data-independent bound is always optimal for the GNMax mechanism.

    Args:
        num_teachers: number of teachers.
        num_classes: number of classes.
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A boolean array of the same length as orders.
    """
    unanimous_votes = np.array([num_teachers] + [0] * (num_classes - 1))
    logq = compute_logq_gnmax(unanimous_votes, sigma)

    rdp_eps_dep = compute_rdp_data_dependent_gnmax(logq, sigma, orders)
    rdp_eps_ind = compute_rdp_data_independent_gnmax(sigma, orders)
    return np.isclose(rdp_eps_dep, rdp_eps_ind)


###################################
# RDP FOR THE THRESHOLD MECHANISM #
###################################


def compute_logpr_answered(t, sigma, votes):
    """
    Computes log(Pr[answered]) for the threshold mechanism.

    Args:
        t: the threshold (a scalar).
        sigma: std of the Gaussian noise in the threshold mechanism.
        votes: a 1-D numpy array of raw ensemble votes for a given query.

    Returns:
        The value of log(Pr[answered]) where log denotes natural logarithm.
    """
    return scipy.stats.norm.logsf(t - round(max(votes)), scale=sigma)


def compute_rdp_data_independent_threshold(sigma, orders):
    """
    Computes data-independent RDP gurantees for the threshold mechanism.

    Args:
        sigma: std of the Gaussian noise in the threshold mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    # The input to the threshold mechanism has sensitivity 1 rather than 2 as
    # compared to the GNMax mechanism, hence the sqrt(2) factor below.
    return compute_rdp_data_independent_gnmax(2 ** .5 * sigma, orders)


def compute_rdp_data_dependent_threshold(logpr, sigma, orders):
    """
    Computes data-dependent RDP gurantees for the threshold mechanism.

    Args:
        logpr: the value of log(Pr[answered]) for the threshold mechanism.
        sigma: std of the Gaussian noise in the threshold mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    logq = min(logpr, _log1mexp(logpr))
    # The input to the threshold mechanism has sensitivity 1 rather than 2 as
    # compared to the GNMax mechanism, hence the sqrt(2) factor below.
    return compute_rdp_data_dependent_gnmax(logq, 2 ** .5 * sigma, orders)


def is_data_independent_rdp_always_opt_threshold(num_teachers, t, sigma, orders):
    """
    Tests whether data-independent bound is always optimal for the threshold mechanism.

    Args:
        num_teachers: number of teachers.
        t: the threshold (a scalar).
        sigma: std of the Gaussian noise in the threshold mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A boolean array of the same length as orders.
    """
    # Since the data-dependent bound depends only on max(votes), it suffices to
    # check whether the data-dependent bounds are better than the data-independent
    # bounds in the extreme cases where max(votes) is minimal or maximal.
    # For both Confident GNMax and Interactive GNMax it holds that
    #   0 <= max(votes) <= num_teachers.
    logpr1 = compute_logpr_answered(t, sigma, np.array([0]))
    logpr2 = compute_logpr_answered(t, sigma, np.array([num_teachers]))

    rdp_eps_dep1 = compute_rdp_data_dependent_threshold(logpr1, sigma, orders)
    rdp_eps_dep2 = compute_rdp_data_dependent_threshold(logpr2, sigma, orders)

    rdp_eps_ind = compute_rdp_data_independent_threshold(sigma, orders)
    return np.logical_and(np.isclose(rdp_eps_dep1, rdp_eps_ind), np.isclose(rdp_eps_dep2, rdp_eps_ind))


# if __name__ == "__main__":
#     num_teachers = 250
#     num_classes = 10
#     sigma_threshold = 200
#     sigma_gnmax = 40
#     t = 300
#     delta = 1e-6
#     orders = np.concatenate((np.arange(2, 100, .5), np.logspace(np.log10(100), np.log10(1000), num=200)))
#
#     unanimous_votes = np.array([num_teachers] + [0] * (num_classes - 1))
#     logq = compute_logq_gnmax(unanimous_votes, sigma_gnmax)
#     rdp_eps_dep = compute_rdp_data_dependent_gnmax(logq, sigma_gnmax, orders)
#     rdp_eps_ind = compute_rdp_data_independent_gnmax(sigma_gnmax, orders)
#     dp_eps_dep = rdp_to_dp(orders, rdp_eps_dep, delta)
#     dp_eps_ind = rdp_to_dp(orders, rdp_eps_ind, delta)
#     print("dp_eps_dep:", dp_eps_dep)
#     print("dp_eps_ind:", dp_eps_ind)
