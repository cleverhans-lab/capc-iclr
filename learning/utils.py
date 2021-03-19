from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from queryset import QuerySet
import numpy as np
import torch
import sys
import os


def load_private_data_and_qap(args):
    """Load labeled private data and query-answer pairs for retraining private models."""
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        all_private_datasets = datasets.MNIST(root=args.dataset_path,
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      (0.13066048,),
                                                      (0.3081078,))]),
                                              download=True)
        private_dataset_size = len(all_private_datasets) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            begin = i * private_dataset_size
            end = len(all_private_datasets) if i == args.num_models - 1 else (
                                                                                         i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            query_dataset = QuerySet(args,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.13251461,),
                                                              (0.31048025,))]),
                                     id=i)
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset == 'svhn':
        trainset = datasets.SVHN(root=args.dataset_path,
                                 split='train',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.43768212, 0.44376972, 0.47280444),
                                         (
                                         0.19803013, 0.20101563, 0.19703615))]),
                                 download=True)
        extraset = datasets.SVHN(root=args.dataset_path,
                                 split='extra',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.42997558, 0.4283771, 0.44269393),
                                         (0.19630221, 0.1978732, 0.19947216))]),
                                 download=True)
        private_trainset_size = len(trainset) // args.num_models
        private_extraset_size = len(extraset) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            train_begin = i * private_trainset_size
            extra_begin = i * private_extraset_size
            train_end = len(trainset) if i == args.num_models - 1 else (
                                                                                   i + 1) * private_trainset_size
            extra_end = len(extraset) if i == args.num_models - 1 else (
                                                                                   i + 1) * private_extraset_size
            train_indices = list(range(train_begin, train_end))
            extra_indices = list(range(extra_begin, extra_end))
            private_trainset = Subset(trainset, train_indices)
            private_extraset = Subset(extraset, extra_indices)
            query_dataset = QuerySet(args,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.45242317,
                                                               0.45249586,
                                                               0.46897715),
                                                              (0.21943446,
                                                               0.22656967,
                                                               0.22850613))]),
                                     id=i)
            augmented_dataset = ConcatDataset(
                [private_trainset, private_extraset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        all_private_datasets = datasets.CIFAR10(args.dataset_path,
                                                train=True,
                                                transform=transforms.Compose([
                                                    transforms.Pad(4),
                                                    transforms.RandomCrop(32),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((
                                                                         0.49139969,
                                                                         0.48215842,
                                                                         0.44653093),
                                                                         (
                                                                         0.24703223,
                                                                         0.24348513,
                                                                         0.26158784))]),
                                                download=True)
        private_dataset_size = len(all_private_datasets) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            begin = i * private_dataset_size
            end = len(all_private_datasets) if i == args.num_models - 1 else (
                                                                                         i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            query_dataset = QuerySet(args,
                                     transform=transforms.Compose([
                                         transforms.Pad(4),
                                         transforms.RandomCrop(32),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.49421429,
                                                               0.4851314,
                                                               0.45040911),
                                                              (0.24665252,
                                                               0.24289226,
                                                               0.26159238))]),
                                     id=i)
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders


def save_raw_queries(args, indices, name):
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        dataset = datasets.MNIST(root=args.dataset_path,
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)
    elif args.dataset == 'svhn':
        dataset = datasets.SVHN(root=args.dataset_path,
                                split='test',
                                transform=transforms.ToTensor(),
                                download=True)
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        dataset = datasets.CIFAR10(args.dataset_path,
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=True)
    query_dataset = Subset(dataset, indices)
    queryloader = DataLoader(query_dataset, batch_size=args.batch_size,
                             shuffle=False, **kwargs)
    all_samples = []
    for data, _ in queryloader:
        all_samples.append(data.numpy())
    all_samples = np.concatenate(all_samples, axis=0).transpose(0, 2, 3, 1)
    assert len(all_samples.shape) == 4 and all_samples.shape[0] == len(indices)
    all_samples = np.squeeze((all_samples * 255).astype(np.uint8))
    assert len(all_samples.shape) == (3 if args.dataset == 'mnist' else 4)
    filename = "{}-raw-samples-(mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f})".format(
        name, args.mode, args.threshold, args.sigma_gnmax, args.sigma_threshold,
        args.budget)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, all_samples)


def save_queries(args, indices, name):
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        dataset = datasets.MNIST(root=args.dataset_path,
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.13251461,),
                                                          (0.31048025,))]),
                                 download=True)
    elif args.dataset == 'svhn':
        dataset = datasets.SVHN(root=args.dataset_path,
                                split='test',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.45242317, 0.45249586, 0.46897715),
                                        (0.21943446, 0.22656967, 0.22850613))]),
                                download=True)
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        dataset = datasets.CIFAR10(args.dataset_path,
                                   train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.49421429, 0.4851314, 0.45040911),
                                           (0.24665252, 0.24289226,
                                            0.26159238))]),
                                   download=True)
    query_dataset = Subset(dataset, indices)
    queryloader = DataLoader(query_dataset, batch_size=args.batch_size,
                             shuffle=False, **kwargs)
    all_samples = []
    for data, _ in queryloader:
        all_samples.append(data.numpy())
    all_samples = np.concatenate(all_samples, axis=0)
    assert len(all_samples.shape) == 4 and all_samples.shape[0] == len(indices)
    all_samples = np.squeeze(all_samples)
    assert len(all_samples.shape) == (3 if args.dataset == 'mnist' else 4)
    filename = "{}-samples-(mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f})".format(
        name, args.mode, args.threshold, args.sigma_gnmax, args.sigma_threshold,
        args.budget)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, all_samples)


def load_private_data_and_qap_imbalanced(args):
    """Load labeled private data (imbalanced) and query-answer pairs for retraining private models."""
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        all_private_datasets = datasets.MNIST(root=args.dataset_path,
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      (0.13066048,),
                                                      (0.3081078,))]),
                                              download=True)
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models))
                size_strong = (len(class_indices[
                                       c]) - size_weak * args.num_querying_parties) // (
                                          args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        data_indices[i].append(class_indices[c][begin:end])
                    else:
                        begin = size_weak * args.num_querying_parties + (
                                    i - args.num_querying_parties) * size_strong
                        end = len(class_indices[
                                      c]) if i == args.num_models - 1 else size_weak * \
                                                                           args.num_querying_parties + (
                                                                                       i + 1 - args.num_querying_parties) * size_strong
                        data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = len(
                        class_indices[c]) if i == args.num_models - 1 else (
                                                                                       i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [np.concatenate(data_indices[i], axis=0) for i in
                        range(args.num_models)]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets)
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets)
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            query_dataset = QuerySet(args,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.13251461,),
                                                              (0.31048025,))]),
                                     id=i)
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset == 'svhn':
        trainset = datasets.SVHN(root=args.dataset_path,
                                 split='train',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.43768212, 0.44376972, 0.47280444),
                                         (
                                         0.19803013, 0.20101563, 0.19703615))]),
                                 download=True)
        extraset = datasets.SVHN(root=args.dataset_path,
                                 split='extra',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.42997558, 0.4283771, 0.44269393),
                                         (0.19630221, 0.1978732, 0.19947216))]),
                                 download=True)
        trainset_class_indices = get_class_indices(trainset, args)
        trainset_data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(args.weak_class_ratio * (
                            len(trainset_class_indices[c]) / args.num_models))
                size_strong = (len(trainset_class_indices[c]) - size_weak *
                               args.num_querying_parties) // (
                                          args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        trainset_data_indices[i].append(
                            trainset_class_indices[c][begin:end])
                    else:
                        begin = size_weak * args.num_querying_parties + (
                                    i - args.num_querying_parties) * size_strong
                        end = len(trainset_class_indices[
                                      c]) if i == args.num_models - 1 else size_weak * \
                                                                           args.num_querying_parties + (
                                                                                       i + 1 - args.num_querying_parties) * size_strong
                        trainset_data_indices[i].append(
                            trainset_class_indices[c][begin:end])
            else:
                size = len(trainset_class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = len(trainset_class_indices[
                                  c]) if i == args.num_models - 1 else (
                                                                                   i + 1) * size
                    trainset_data_indices[i].append(
                        trainset_class_indices[c][begin:end])
        trainset_data_indices = [
            np.concatenate(trainset_data_indices[i], axis=0) for i in
            range(args.num_models)]
        assert sum([len(trainset_data_indices[i]) for i in
                    range(args.num_models)]) == len(trainset)
        assert len(set(np.concatenate(trainset_data_indices, axis=0))) == len(
            trainset)
        extraset_class_indices = get_class_indices(extraset, args)
        extraset_data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(args.weak_class_ratio * (
                            len(extraset_class_indices[c]) / args.num_models))
                size_strong = (len(extraset_class_indices[c]) - size_weak *
                               args.num_querying_parties) // (
                                          args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        extraset_data_indices[i].append(
                            extraset_class_indices[c][begin:end])
                    else:
                        begin = size_weak * args.num_querying_parties + (
                                    i - args.num_querying_parties) * size_strong
                        end = len(extraset_class_indices[
                                      c]) if i == args.num_models - 1 else size_weak * \
                                                                           args.num_querying_parties + (
                                                                                       i + 1 - args.num_querying_parties) * size_strong
                        extraset_data_indices[i].append(
                            extraset_class_indices[c][begin:end])
            else:
                size = len(extraset_class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = len(extraset_class_indices[
                                  c]) if i == args.num_models - 1 else (
                                                                                   i + 1) * size
                    extraset_data_indices[i].append(
                        extraset_class_indices[c][begin:end])
        extraset_data_indices = [
            np.concatenate(extraset_data_indices[i], axis=0) for i in
            range(args.num_models)]
        assert sum([len(extraset_data_indices[i]) for i in
                    range(args.num_models)]) == len(extraset)
        assert len(set(np.concatenate(extraset_data_indices, axis=0))) == len(
            extraset)
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_trainset = Subset(trainset, trainset_data_indices[i])
            private_extraset = Subset(extraset, extraset_data_indices[i])
            query_dataset = QuerySet(args,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.45242317,
                                                               0.45249586,
                                                               0.46897715),
                                                              (0.21943446,
                                                               0.22656967,
                                                               0.22850613))]),
                                     id=i)
            augmented_dataset = ConcatDataset(
                [private_trainset, private_extraset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        all_private_datasets = datasets.CIFAR10(args.dataset_path,
                                                train=True,
                                                transform=transforms.Compose([
                                                    transforms.Pad(4),
                                                    transforms.RandomCrop(32),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((
                                                                         0.49139969,
                                                                         0.48215842,
                                                                         0.44653093),
                                                                         (
                                                                         0.24703223,
                                                                         0.24348513,
                                                                         0.26158784))]),
                                                download=True)
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models))
                size_strong = (len(class_indices[
                                       c]) - size_weak * args.num_querying_parties) // (
                                          args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        data_indices[i].append(class_indices[c][begin:end])
                    else:
                        begin = size_weak * args.num_querying_parties + (
                                    i - args.num_querying_parties) * size_strong
                        end = len(class_indices[
                                      c]) if i == args.num_models - 1 else size_weak * \
                                                                           args.num_querying_parties + (
                                                                                       i + 1 - args.num_querying_parties) * size_strong
                        data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = len(
                        class_indices[c]) if i == args.num_models - 1 else (
                                                                                       i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [np.concatenate(data_indices[i], axis=0) for i in
                        range(args.num_models)]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets)
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets)
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            query_dataset = QuerySet(args,
                                     transform=transforms.Compose([
                                         transforms.Pad(4),
                                         transforms.RandomCrop(32),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.49421429,
                                                               0.4851314,
                                                               0.45040911),
                                                              (0.24665252,
                                                               0.24289226,
                                                               0.26159238))]),
                                     id=i)
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(augmented_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders


def load_private_data_imbalanced(args):
    """Load labeled private data for training private models in an imbalanced way."""
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        all_private_datasets = datasets.MNIST(root=args.dataset_path,
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      (0.13066048,),
                                                      (0.3081078,))]),
                                              download=True)
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models))
                size_strong = (len(class_indices[
                                       c]) - size_weak * args.num_querying_parties) // (
                                          args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        data_indices[i].append(class_indices[c][begin:end])
                    else:
                        begin = size_weak * args.num_querying_parties + (
                                    i - args.num_querying_parties) * size_strong
                        end = len(class_indices[
                                      c]) if i == args.num_models - 1 else size_weak * \
                                                                           args.num_querying_parties + (
                                                                                       i + 1 - args.num_querying_parties) * size_strong
                        data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = len(
                        class_indices[c]) if i == args.num_models - 1 else (
                                                                                       i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [np.concatenate(data_indices[i], axis=0) for i in
                        range(args.num_models)]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets)
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets)
        all_private_trainloaders = []
        for i in range(args.num_models):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            private_trainloader = DataLoader(private_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True, **kwargs)
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders
    elif args.dataset == 'svhn':
        trainset = datasets.SVHN(root=args.dataset_path,
                                 split='train',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.43768212, 0.44376972, 0.47280444),
                                         (
                                         0.19803013, 0.20101563, 0.19703615))]),
                                 download=True)
        extraset = datasets.SVHN(root=args.dataset_path,
                                 split='extra',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.42997558, 0.4283771, 0.44269393),
                                         (0.19630221, 0.1978732, 0.19947216))]),
                                 download=True)
        trainset_class_indices = get_class_indices(trainset, args)
        trainset_data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(args.weak_class_ratio * (
                            len(trainset_class_indices[c]) / args.num_models))
                size_strong = (len(trainset_class_indices[c]) - size_weak *
                               args.num_querying_parties) // (
                                          args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        trainset_data_indices[i].append(
                            trainset_class_indices[c][begin:end])
                    else:
                        begin = size_weak * args.num_querying_parties + (
                                    i - args.num_querying_parties) * size_strong
                        end = len(trainset_class_indices[
                                      c]) if i == args.num_models - 1 else size_weak * \
                                                                           args.num_querying_parties + (
                                                                                       i + 1 - args.num_querying_parties) * size_strong
                        trainset_data_indices[i].append(
                            trainset_class_indices[c][begin:end])
            else:
                size = len(trainset_class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = len(trainset_class_indices[
                                  c]) if i == args.num_models - 1 else (
                                                                                   i + 1) * size
                    trainset_data_indices[i].append(
                        trainset_class_indices[c][begin:end])
        trainset_data_indices = [
            np.concatenate(trainset_data_indices[i], axis=0) for i in
            range(args.num_models)]
        assert sum([len(trainset_data_indices[i]) for i in
                    range(args.num_models)]) == len(trainset)
        assert len(set(np.concatenate(trainset_data_indices, axis=0))) == len(
            trainset)
        extraset_class_indices = get_class_indices(extraset, args)
        extraset_data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(args.weak_class_ratio * (
                            len(extraset_class_indices[c]) / args.num_models))
                size_strong = (len(extraset_class_indices[c]) - size_weak *
                               args.num_querying_parties) // (
                                          args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        extraset_data_indices[i].append(
                            extraset_class_indices[c][begin:end])
                    else:
                        begin = size_weak * args.num_querying_parties + (
                                    i - args.num_querying_parties) * size_strong
                        end = len(extraset_class_indices[
                                      c]) if i == args.num_models - 1 else size_weak * \
                                                                           args.num_querying_parties + (
                                                                                       i + 1 - args.num_querying_parties) * size_strong
                        extraset_data_indices[i].append(
                            extraset_class_indices[c][begin:end])
            else:
                size = len(extraset_class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = len(extraset_class_indices[
                                  c]) if i == args.num_models - 1 else (
                                                                                   i + 1) * size
                    extraset_data_indices[i].append(
                        extraset_class_indices[c][begin:end])
        extraset_data_indices = [
            np.concatenate(extraset_data_indices[i], axis=0) for i in
            range(args.num_models)]
        assert sum([len(extraset_data_indices[i]) for i in
                    range(args.num_models)]) == len(extraset)
        assert len(set(np.concatenate(extraset_data_indices, axis=0))) == len(
            extraset)
        all_private_trainloaders = []
        for i in range(args.num_models):
            private_dataset = ConcatDataset(
                [Subset(trainset, trainset_data_indices[i]),
                 Subset(extraset, extraset_data_indices[i])])
            private_trainloader = DataLoader(private_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True, **kwargs)
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        all_private_datasets = datasets.CIFAR10(args.dataset_path,
                                                train=True,
                                                transform=transforms.Compose([
                                                    transforms.Pad(4),
                                                    transforms.RandomCrop(32),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((
                                                                         0.49139969,
                                                                         0.48215842,
                                                                         0.44653093),
                                                                         (
                                                                         0.24703223,
                                                                         0.24348513,
                                                                         0.26158784))]),
                                                download=True)
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models))
                size_strong = (len(class_indices[
                                       c]) - size_weak * args.num_querying_parties) // (
                                          args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        data_indices[i].append(class_indices[c][begin:end])
                    else:
                        begin = size_weak * args.num_querying_parties + (
                                    i - args.num_querying_parties) * size_strong
                        end = len(class_indices[
                                      c]) if i == args.num_models - 1 else size_weak * \
                                                                           args.num_querying_parties + (
                                                                                       i + 1 - args.num_querying_parties) * size_strong
                        data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = len(
                        class_indices[c]) if i == args.num_models - 1 else (
                                                                                       i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [np.concatenate(data_indices[i], axis=0) for i in
                        range(args.num_models)]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets)
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets)
        all_private_trainloaders = []
        for i in range(args.num_models):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            private_trainloader = DataLoader(private_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True, **kwargs)
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders


def load_private_data(args):
    """Load labeled private data for training private models."""
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        all_private_datasets = datasets.MNIST(root=args.dataset_path,
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      (0.13066048,),
                                                      (0.3081078,))]),
                                              download=True)
        private_dataset_size = len(all_private_datasets) // args.num_models
        all_private_trainloaders = []
        for i in range(args.num_models):
            begin = i * private_dataset_size
            end = len(all_private_datasets) if i == args.num_models - 1 else (
                                                                                         i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            private_trainloader = DataLoader(private_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True, **kwargs)
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders
    elif args.dataset == 'svhn':
        trainset = datasets.SVHN(root=args.dataset_path,
                                 split='train',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.43768212, 0.44376972, 0.47280444),
                                         (
                                         0.19803013, 0.20101563, 0.19703615))]),
                                 download=True)
        extraset = datasets.SVHN(root=args.dataset_path,
                                 split='extra',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.42997558, 0.4283771, 0.44269393),
                                         (0.19630221, 0.1978732, 0.19947216))]),
                                 download=True)
        private_trainset_size = len(trainset) // args.num_models
        private_extraset_size = len(extraset) // args.num_models
        all_private_trainloaders = []
        for i in range(args.num_models):
            train_begin = i * private_trainset_size
            extra_begin = i * private_extraset_size
            train_end = len(trainset) if i == args.num_models - 1 else (
                                                                                   i + 1) * private_trainset_size
            extra_end = len(extraset) if i == args.num_models - 1 else (
                                                                                   i + 1) * private_extraset_size
            train_indices = list(range(train_begin, train_end))
            extra_indices = list(range(extra_begin, extra_end))
            private_dataset = ConcatDataset([Subset(trainset, train_indices),
                                             Subset(extraset, extra_indices)])
            private_trainloader = DataLoader(private_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True, **kwargs)
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        all_private_datasets = datasets.CIFAR10(args.dataset_path,
                                                train=True,
                                                transform=transforms.Compose([
                                                    transforms.Pad(4),
                                                    transforms.RandomCrop(32),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((
                                                                         0.49139969,
                                                                         0.48215842,
                                                                         0.44653093),
                                                                         (
                                                                         0.24703223,
                                                                         0.24348513,
                                                                         0.26158784))]),
                                                download=True)
        private_dataset_size = len(all_private_datasets) // args.num_models
        all_private_trainloaders = []
        for i in range(args.num_models):
            begin = i * private_dataset_size
            end = len(all_private_datasets) if i == args.num_models - 1 else (
                                                                                         i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            private_trainloader = DataLoader(private_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True, **kwargs)
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders


def load_ordered_unlabeled_data(args, indices):
    """Load unlabeled private data according to a specific order."""
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        dataset = datasets.MNIST(root=args.dataset_path,
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.13251461,),
                                                          (0.31048025,))]),
                                 download=True)
    elif args.dataset == 'svhn':
        dataset = datasets.SVHN(root=args.dataset_path,
                                split='test',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.45242317, 0.45249586, 0.46897715),
                                        (0.21943446, 0.22656967, 0.22850613))]),
                                download=True)
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        dataset = datasets.CIFAR10(args.dataset_path,
                                   train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.49421429, 0.4851314, 0.45040911),
                                           (0.24665252, 0.24289226,
                                            0.26159238))]),
                                   download=True)
    # A part of the original testset is loaded according to a specific order.
    unlabeled_dataset = Subset(dataset, indices)
    unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                      batch_size=args.batch_size, shuffle=False,
                                      **kwargs)
    return unlabeled_dataloader


def load_unlabeled_data(args):
    """Load unlabeled private data for query selection."""
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        dataset = datasets.MNIST(root=args.dataset_path,
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.13251461,),
                                                          (0.31048025,))]),
                                 download=True)
    elif args.dataset == 'svhn':
        dataset = datasets.SVHN(root=args.dataset_path,
                                split='test',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.45242317, 0.45249586, 0.46897715),
                                        (0.21943446, 0.22656967, 0.22850613))]),
                                download=True)
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        dataset = datasets.CIFAR10(args.dataset_path,
                                   train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.49421429, 0.4851314, 0.45040911),
                                           (0.24665252, 0.24289226,
                                            0.26159238))]),
                                   download=True)
    # Only a part of the original testset is used for query selection.
    size = args.num_unlabeled_samples // args.num_querying_parties
    all_unlabeled_dataloaders = []
    for i in range(args.num_querying_parties):
        begin = i * size
        end = args.num_unlabeled_samples if i == args.num_querying_parties - 1 else (
                                                                                                i + 1) * size
        indices = list(range(begin, end))
        unlabeled_dataset = Subset(dataset, indices)
        unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False, **kwargs)
        all_unlabeled_dataloaders.append(unlabeled_dataloader)
    return all_unlabeled_dataloaders


def load_training_data(args):
    """Load labeled data for training non-private baseline models."""
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        trainset = datasets.MNIST(root=args.dataset_path,
                                  train=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.13066048,),
                                                           (0.3081078,))]),
                                  download=True)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 shuffle=True, **kwargs)
    elif args.dataset == 'svhn':
        trainset = datasets.SVHN(root=args.dataset_path,
                                 split='train',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.43768212, 0.44376972, 0.47280444),
                                         (
                                         0.19803013, 0.20101563, 0.19703615))]),
                                 download=True)
        extraset = datasets.SVHN(root=args.dataset_path,
                                 split='extra',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         (0.42997558, 0.4283771, 0.44269393),
                                         (0.19630221, 0.1978732, 0.19947216))]),
                                 download=True)
        trainloader = DataLoader(ConcatDataset([trainset, extraset]),
                                 batch_size=args.batch_size, shuffle=True,
                                 **kwargs)
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        trainset = datasets.CIFAR10(args.dataset_path,
                                    train=True,
                                    transform=transforms.Compose([
                                        transforms.Pad(4),
                                        transforms.RandomCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.49139969,
                                                              0.48215842,
                                                              0.44653093),
                                                             (0.24703223,
                                                              0.24348513,
                                                              0.26158784))]),
                                    download=True)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 shuffle=True, **kwargs)
    return trainloader


def load_evaluation_data(args):
    """Load labeled data for evaluation."""
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        dataset = datasets.MNIST(root=args.dataset_path,
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.13251461,),
                                                          (0.31048025,))]),
                                 download=True)
    elif args.dataset == 'svhn':
        dataset = datasets.SVHN(root=args.dataset_path,
                                split='test',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.45242317, 0.45249586, 0.46897715),
                                        (0.21943446, 0.22656967, 0.22850613))]),
                                download=True)
    else:
        assert args.dataset == 'cifar10', "Dataset name must be 'mnist', 'svhn' or 'cifar10'!"
        dataset = datasets.CIFAR10(args.dataset_path,
                                   train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.49421429, 0.4851314, 0.45040911),
                                           (0.24665252, 0.24289226,
                                            0.26159238))]),
                                   download=True)
    # Only a part of the original testset is used for evaluation.
    evalset = Subset(dataset,
                     list(range(args.num_unlabeled_samples, len(dataset))))
    evalloader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False,
                            **kwargs)
    return evalloader


def train(model, trainloader, optimizer, args):
    """Train a given model on a given dataset using a given optimizer for one epoch."""
    model.train()
    losses = []
    for batch_id, (data, target) in enumerate(trainloader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = F.cross_entropy(output, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(losses)
    return train_loss


def evaluate(model, evalloader, args):
    """Evaluate the accuracy of a given model on a given dataset."""
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for data, target in evalloader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            losses.append(F.cross_entropy(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    eval_loss = np.mean(losses)
    return eval_loss, 100. * correct / len(evalloader.dataset)


def evaluate_detailed(model, evalloader, args):
    """
    Evaluate the class-specific accuracy of a given model on a given dataset.

    Returns:
        A 1-D numpy array of length L = num-classes, containg the accuracy for each class.
    """
    model.eval()
    correct = np.zeros(args.num_classes, dtype=np.int64)
    wrong = np.zeros(args.num_classes, dtype=np.int64)
    with torch.no_grad():
        for data, target in evalloader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            preds = output.data.max(dim=1)[1].cpu().numpy().astype(np.int64)
            target = target.data.cpu().numpy().astype(np.int64)
            for label, pred in zip(target, preds):
                if label == pred:
                    correct[label] += 1
                else:
                    wrong[label] += 1
    assert correct.sum() + wrong.sum() == len(evalloader.dataset)
    return 100. * correct / (correct + wrong)


def one_hot(indices, num_classes):
    """
    Convert labels into one-hot vectors.

    Args:
        indices: a 1-D vector containing labels.
        num_classes: number of classes.

    Returns:
        A 2-D matrix contaning one-hot vectors, with one vector per row.
    """
    onehot = torch.zeros((len(indices), num_classes))
    for i in range(len(indices)):
        onehot[i][indices[i]] = 1
    return onehot


def augmented_print(text, file, flush=False):
    """Print to both the standard output and the given file."""
    assert isinstance(text, str)
    print(text)
    file.write(text + "\n")
    if flush:
        sys.stdout.flush()
        file.flush()


def class_ratio(dataset, args):
    """The ratio of each class in the given dataset."""
    counts = np.zeros(args.num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        counts[dataset[i][1]] += 1
    assert counts.sum() == len(dataset)
    return counts, 100. * counts / len(dataset)


def get_class_indices(dataset, args):
    """The indices of samples belonging to each class."""
    indices = [[] for i in range(args.num_classes)]
    for i in range(len(dataset)):
        indices[dataset[i][1]].append(i)
    indices = [np.asarray(indices[i]) for i in range(args.num_classes)]
    assert sum([len(indices[i]) for i in range(args.num_classes)]) == len(
        dataset)
    assert len(set(np.concatenate(indices, axis=0))) == len(dataset)
    return indices


def print_args(args, get_str=False):
    if 'delimiter' in args:
        delimiter = args.delimiter
    elif 'sep' in args:
        delimiter = args.sep
    else:
        delimiter = ';'
    print('args: ')
    keys = sorted(
        [a for a in dir(args) if not (
                a.startswith('__') or a.startswith(
            '_') or a == 'sep' or a == 'delimiter')])
    values = [getattr(args, key) for key in keys]
    if get_str:
        keys_str = delimiter.join([str(a) for a in keys])
        values_str = delimiter.join([str(a) for a in values])
        print(keys_str)
        print(values_str)
        return keys_str, values_str
    else:
        for key, value in zip(keys, values):
            print(key, ': ', value)
