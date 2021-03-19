from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os


class QuerySet(Dataset):
    """Labeled dataset consisting of query-answer pairs."""

    def __init__(self, args, transform, id):
        super(QuerySet, self).__init__()
        # Queries
        filename = "model({:d})-raw-samples-(mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).npy".format(
            id + 1, args.mode, args.threshold, args.sigma_gnmax, args.sigma_threshold, args.budget)
        filepath = os.path.join(args.ensemble_model_path, filename)
        if os.path.isfile(filepath):
            self.samples = np.load(filepath)
        else:
            raise Exception("Queries '{}' do not exist, please generate them via 'query_ensemble_model(args)'!".format(filepath))
        # Answers
        filename = "ensemble({:d})-aggregated-labels-(mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).npy".format(
            id + 1, args.mode, args.threshold, args.sigma_gnmax, args.sigma_threshold, args.budget)
        filepath = os.path.join(args.ensemble_model_path, filename)
        if os.path.isfile(filepath):
            self.labels = np.load(filepath)
        else:
            raise Exception("Answers '{}' do not exist, please generate them via 'query_ensemble_model(args)'!".format(filepath))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.transform(Image.fromarray(self.samples[idx])), int(self.labels[idx])
