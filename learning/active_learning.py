import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from functools import partial
from copy import deepcopy
# delta = partial(np.linalg.norm, )
from sklearn.metrics import pairwise_distances
delta = np.linalg.norm
import abc
import numpy
# from gurobipy import Model, LinExpr, UB, GRB
import pickle
import numpy.matlib
import time
import pickle
import bisect


class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self):
    pass

  def flatten_X(self, X):
    shape = X.shape
    flat_X = X
    if len(shape) > 2:
      flat_X = np.reshape(X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None


class kCenterGreedy(SamplingMethod):

  def __init__(self, metric='euclidean'):
    super().__init__()
    self.name = 'kcenter'
    self.metric = metric
    self.min_distances = None
    self.already_selected = []

  def update_distances(self, features, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.
    Args:
      features: features (projection) from model
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = features[cluster_centers]
      dist = pairwise_distances(features.detach().numpy(), x.detach().numpy(), metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, pool, model, already_selected, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.
    Args:
      pool: tuple of (X, Y)
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size
    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    try:
      # Assumes that the transform function takes in original data and not
      # flattened data.
      print('Getting transformed features...')
      features = model.forward(pool[0].float())
      print('Calculating distances...')
      self.update_distances(features, already_selected, only_new=False, reset_dist=True)
    except Exception as e:
      print(f"error: {e}")
      print('Using flat_X as features.')
      self.update_distances(features, already_selected, only_new=True, reset_dist=False)

    new_batch = []

    for _ in range(N):
      if self.already_selected is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(pool[0].shape[0]))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected

      self.update_distances(features, [ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))
    self.already_selected = already_selected

    return new_batch


# def solve_fac_loc(xx, yy, subset, n, budget):
#   model = Model("k-center")
#   x = {}
#   y = {}
#   z = {}
#   for i in range(n):
#     # z_i: is a loss
#     z[i] = model.addVar(obj=1, ub=0.0, vtype="B", name="z_{}".format(i))
#
#   m = len(xx)
#   for i in range(m):
#     _x = xx[i]
#     _y = yy[i]
#     # y_i = 1 means i is facility, 0 means it is not
#     if _y not in y:
#       if _y in subset:
#         y[_y] = model.addVar(obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
#       else:
#         y[_y] = model.addVar(obj=0, vtype="B", name="y_{}".format(_y))
#     # if not _x == _y:
#     x[_x, _y] = model.addVar(obj=0, vtype="B", name="x_{},{}".format(_x, _y))
#   model.update()
#
#   coef = [1 for j in range(n)]
#   var = [y[j] for j in range(n)]
#   model.addConstr(LinExpr(coef, var), "=", rhs=budget + len(subset), name="k_center")
#
#   for i in range(m):
#     _x = xx[i]
#     _y = yy[i]
#     # if not _x == _y:
#     model.addConstr(x[_x, _y], "<", y[_y], name="Strong_{},{}".format(_x, _y))
#
#   yyy = {}
#   for v in range(m):
#     _x = xx[v]
#     _y = yy[v]
#     if _x not in yyy:
#       yyy[_x] = []
#     if _y not in yyy[_x]:
#       yyy[_x].append(_y)
#
#   for _x in yyy:
#     coef = []
#     var = []
#     for _y in yyy[_x]:
#       # if not _x==_y:
#       coef.append(1)
#       var.append(x[_x, _y])
#     coef.append(1)
#     var.append(z[_x])
#     model.addConstr(LinExpr(coef, var), "=", 1, name="Assign{}".format(_x))
#   model.__data = x, y, z
#   return model


def greedy_k_center(model, pool, already_selected, batch_size):
  # note pool should have all points in a tuple of (X, Y)
  # already selected are the indices
  # this returns the indices o the selected samples
  selecter = kCenterGreedy()
  return selecter.select_batch_(pool, model, already_selected, batch_size)


def robust_k_center(x, y, z):
  budget = 10000

  start = time.clock()
  num_images = x.shape[0]
  dist_mat = numpy.matmul(x, x.transpose())

  sq = numpy.array(dist_mat.diagonal()).reshape(num_images, 1)
  dist_mat *= -2
  dist_mat += sq
  dist_mat += sq.transpose()

  elapsed = time.clock() - start
  print(f"Time spent in (distance computation) is: {elapsed}")

  num_images = 50000

  # We need to get k centers start with greedy solution
  budget = 10000
  subset = [i for i in range(1)]

  ub = UB
  lb = ub / 2.0
  max_dist = ub

  _x, _y = numpy.where(dist_mat <= max_dist)
  _d = dist_mat[_x, _y]
  subset = [i for i in range(1)]
  model = solve_fac_loc(_x, _y, subset, num_images, budget)
  # model.setParam( 'OutputFlag', False )
  x, y, z = model.__data
  delta = 1e-7
  while ub - lb > delta:
    print("State", ub, lb)
    cur_r = (ub + lb) / 2.0
    viol = numpy.where(_d > cur_r)
    new_max_d = numpy.min(_d[_d >= cur_r])
    new_min_d = numpy.max(_d[_d <= cur_r])
    print("If it succeeds, new max is:", new_max_d, new_min_d)
    for v in viol[0]:
      x[_x[v], _y[v]].UB = 0

    model.update()
    r = model.optimize()
    if model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
      failed = True
      print("Infeasible")
    elif sum([z[i].X for i in range(len(z))]) > 0:
      failed = True
      print("Failed")
    else:
      failed = False
    if failed:
      lb = max(cur_r, new_max_d)
      # failed so put edges back
      for v in viol[0]:
        x[_x[v], _y[v]].UB = 1
    else:
      print("sol found", cur_r, lb, ub)
      ub = min(cur_r, new_min_d)
      model.write("s_{}_solution_{}.sol".format(budget, cur_r))


# def k_center_greedy(pool, s0, budget):
#   s = s0
#   new_points = []
#   init_size = len(s0)
#   s0 = set(s0)
#   while len(s) < init_size + budget:
#     max_dist, index = 0, -1
#     for i in range(len(pool)):
#       if i in s:
#         continue
#       min_dist = 1000000000
#       for j in s:
#         min_dist = min(min_dist, delta(pool[i], pool[j]))
#       if min_dist > max_dist:
#         max_dist = min_dist
#         index = i
#     if index < 0:
#       raise ValueError('index should not have been -1, error ')
#     s.append(index)
#     new_points.append(index)
#   return s, new_points
#
# def feasible():
#   pass


# def robust_k_center(pool, s0, budget, bound):
#   greedy_init, greedy_points = k_center_greedy(pool, s0, budget)
#   d2_opt = 0
#   for i in range(len(pool)):
#     min_dist = 10000000
#     for j in greedy_init:
#       min_dist = min(min_dist, delta(pool[i], pool[j]))
#     d2_opt = max(d2_opt, min_dist)
#   lb = d2_opt / 2
#   ub = d2_opt
#   while lb < ub:
#     midpoint = (lb + ub) / 2
#     if feasible(budget, s0, midpoint, bound):
#       new_ub = 0
#       for i in range(len(pool)):
#         for j in greedy_init:
#           dist = delta(pool[i], pool[j])
#           if dist > new_ub and dist < midpoint:
#             new_ub = dist
#       ub = new_ub
#     else:
#       new_lb = 10000000000
#       for i in range(len(pool)):
#         for j in greedy_init:
#           dist = delta(pool[i], pool[j])
#           if dist < new_lb and dist > midpoint:
#             new_lb = dist
#       lb = new_lb
