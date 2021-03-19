import active_learning
import models
import numpy as np
import torch


class Args:
  pass

args = Args()
args.dataset = 'mnist'
args.num_classes = 10

m = models.Local_Model('test', args)
x = np.random.uniform(0, 1, (1000, 1, 28, 28)).astype(np.float64)
y = np.random.randint(0, 10, 1000).astype(np.float64)
y = y.reshape((-1, 1))
x = torch.from_numpy(x)
y = torch.from_numpy(y)
inds = active_learning.greedy_k_center(m, (x, y), [], 100)
print(inds)
