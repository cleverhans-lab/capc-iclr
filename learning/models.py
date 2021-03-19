from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import utils
import torch
import math
import os
from architectures.tiny_resnet import ResNet6
from architectures.small_resnet import ResNet8
from architectures.resnet import ResNet10
from architectures.resnet import ResNet12
from architectures.resnet import ResNet14
from architectures.resnet import ResNet16
from architectures.resnet import ResNet18


def Private_Model(name, args):
    """Private model held by each party."""
    if args.architecture.startswith('VGG'):
        return VGG(name=name, args=args)
    elif args.architecture == 'ResNet6':
        return ResNet6(name=name, args=args)
    elif args.architecture == 'ResNet8':
        return ResNet8(name=name, args=args)
    elif args.architecture == 'ResNet10':
        return ResNet10(name=name, args=args)
    elif args.architecture == 'ResNet12':
        return ResNet12(name=name, args=args)
    elif args.architecture == 'ResNet14':
        return ResNet14(name=name, args=args)
    elif args.architecture == 'ResNet16':
        return ResNet16(name=name, args=args)
    elif args.architecture == 'ResNet18':
        return ResNet18(name=name, args=args)
    else:
        raise Exception(f'Unknown architecture: {args.architecture}')


class VGG(nn.Module):

    def __init__(self, name, args):
        super(VGG, self).__init__()
        self.name = name
        self.num_classes = args.num_classes
        self.in_channels = 1 if args.dataset == 'mnist' else 3
        if args.architecture == 'VGG9':
            self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 'M',
                        512, 'M']
        elif args.architecture == 'VGG11':
            self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
                        512, 'M']
        elif args.architecture == 'VGG13':
            self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512,
                        'M', 512, 512, 'M']
        elif args.architecture == 'VGG16':
            self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512,
                        512, 512, 'M', 512, 512, 512, 'M']
        elif args.architecture == 'VGG19':
            self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                        512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        elif args.architecture == 'VGG3':
            self.cfg = [64, 'M', 128, 128, 'M']
        elif args.architecture == 'VGG5':
            self.cfg = [64, 'M', 128, 'M', 256, 'M', 512]
        elif args.architecture == 'VGG7':
            self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512]
        elif args.dataset == 'mnist':
            self.cfg = [64, 'M', 128, 128, 'M']
        elif args.dataset == 'cifar10':
            self.cfg = [64, 'M', 128, 'M', 256, 'M', 512]
        elif args.dataset == 'svhn':
            self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512]
        else:
            raise Exception(
                "Dataset name must be 'mnist', 'svhn' or 'cifar10'!")
        self.features = self.make_layers()
        self.classifier = nn.Linear(128 if args.dataset == 'mnist' else 512,
                                    self.num_classes)
        self._initialize_weights()
        print("Building private model '{}'!".format(self.name))

    def forward(self, x):
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def make_layers(self):
        layers = []
        in_channels = self.in_channels
        for c in self.cfg:
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1,
                                 bias=False)
                layers += [conv, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                in_channels = c
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Ensemble_Model(nn.Module):
    """Noisy ensemble of private models."""

    def __init__(self, name, models, args):
        super(Ensemble_Model, self).__init__()
        self.name = name
        self.models = models
        self.num_classes = args.num_classes
        print("Building ensemble model '{}'!".format(self.name))

    def evaluate(self, evalloader, args):
        """Evaluate the accuracy of noisy ensemble model."""
        gap_list = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        with torch.no_grad():
            for data, target in evalloader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                # Generate raw ensemble votes
                votes = torch.zeros((data.shape[0], self.num_classes))
                for model in self.models:
                    output = model(data)
                    onehot = utils.one_hot(output.data.max(dim=1)[1].cpu(),
                                           self.num_classes)
                    votes += onehot
                # Add Gaussian noise
                assert args.sigma_gnmax >= 0
                if args.sigma_gnmax > 0:
                    noise = torch.from_numpy(
                        np.random.normal(0., args.sigma_gnmax, (
                            data.shape[0], self.num_classes))).float()
                    votes += noise
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()
                preds = votes.max(dim=1)[1].numpy().astype(np.int64)
                target = target.data.cpu().numpy().astype(np.int64)
                for label, pred, gap in zip(target, preds, gaps):
                    gap_list[label] += gap
                    if label == pred:
                        correct[label] += 1
                    else:
                        wrong[label] += 1
        total = correct.sum() + wrong.sum()
        assert total == len(evalloader.dataset)
        return 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gap_list.sum() / total, gap_list / (
                       correct + wrong)

    def inference(self, unlabeled_dataloader, args):
        """Generate raw ensemble votes for RDP analysis."""
        all_votes = []
        with torch.no_grad():
            for data, _ in unlabeled_dataloader:
                if args.cuda:
                    data = data.cuda()
                data = Variable(data)
                # Generate raw ensemble votes
                votes = torch.zeros((data.shape[0], self.num_classes))
                for model in self.models:
                    output = model(data)
                    onehot = utils.one_hot(output.data.max(dim=1)[1].cpu(),
                                           self.num_classes)
                    votes += onehot
                all_votes.append(votes.numpy())
        all_votes = np.concatenate(all_votes, axis=0)
        assert all_votes.shape == (
            len(unlabeled_dataloader.dataset), self.num_classes)
        assert np.all(all_votes.sum(axis=-1) == len(self.models))
        filename = '{}-raw-votes-(mode:{})'.format(self.name, args.mode)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, all_votes)
        return all_votes

    def query(self, queryloader, args, indices_queried):
        """Query a noisy ensemble model."""
        indices_answered = []
        all_preds = []
        all_labels = []
        gaps_detailed = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        with torch.no_grad():
            begin = 0
            end = 0
            for data, target in queryloader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                end += data.shape[0]
                # Generate raw ensemble votes
                votes = torch.zeros((data.shape[0], self.num_classes))
                for model in self.models:
                    output = model(data)
                    onehot = utils.one_hot(output.data.max(dim=1)[1].cpu(),
                                           self.num_classes)
                    votes += onehot
                # Threshold mechanism
                assert args.sigma_threshold > 0
                noise_threshold = np.random.normal(0., args.sigma_threshold,
                                                   data.shape[0])
                answered = (votes.data.max(dim=1)[
                                0].numpy() + noise_threshold) > args.threshold
                indices_answered.append(indices_queried[begin:end][answered])
                # GNMax mechanism
                assert args.sigma_gnmax > 0
                noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
                    data.shape[0], self.num_classes))
                preds = \
                    (votes + torch.from_numpy(noise_gnmax).float()).max(dim=1)[
                        1].numpy().astype(np.int64)[answered]
                all_preds.append(preds)
                # Gap between the ensemble votes of the two most probable classes.
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()[
                    answered]
                # Target labels
                target = target.data.cpu().numpy().astype(np.int64)[answered]
                all_labels.append(target)
                assert len(target) == len(preds) == len(gaps)
                for label, pred, gap in zip(target, preds, gaps):
                    gaps_detailed[label] += gap
                    if label == pred:
                        correct[label] += 1
                    else:
                        wrong[label] += 1
                begin += data.shape[0]
        indices_answered = np.concatenate(indices_answered, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        total = correct.sum() + wrong.sum()
        assert len(indices_answered) == len(all_preds) == len(
            all_labels) == total
        filename = "{}-aggregated-labels-(mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f})".format(
            self.name, args.mode, args.threshold, args.sigma_gnmax,
            args.sigma_threshold, args.budget)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, all_preds)
        filename = "{}-labels-(mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f})".format(
            self.name, args.mode, args.threshold, args.sigma_gnmax,
            args.sigma_threshold, args.budget)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, all_labels)
        return indices_answered, 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gaps_detailed.sum() / total, gaps_detailed / (
                       correct + wrong)
