from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass

import argparse
import numpy as np
import os
import pickle
import random
import scipy.stats
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import analysis
import models
import utils

user = getpass.getuser()

# commands = ['train_private_models']
commands = ['query_ensemble_model', 'retrain_private_models']

# dataset = 'cifar10'
dataset = 'svhn'
if dataset == 'cifar10':
    lr = 0.01
    weight_decay = 1e-5
    batch_size = 128
    end_id = 50
    num_epochs = 500
    num_models = 50
    threshold = 50.
    sigma_gnmax = 7.0
    sigma_threshold = 30.0
    budget = 20.0
    budgets = [budget]
    architecture = 'ResNet12'
elif dataset == 'svhn':
    lr = 0.1
    weight_decay = 1e-4
    batch_size = 128
    end_id = 1
    num_epochs = 200
    num_models = 250
    threshold = 300.
    sigma_gnmax = 40.
    sigma_threshold = 200.0
    budget = 2.0
    # budget = 6.0
    # budget = float('inf')
    # budgets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5]
    budgets = [budget]
    architecture = 'VGG'
    # architecture = 'ResNet6'
    if architecture.startswith('ResNet'):
        lr = 0.01
        weight_decay = 1e-5
        num_epochs = 300

else:
    raise Exception('Unknown dataset: {}'.format(dataset))

parser = argparse.ArgumentParser(
    description='Confidential And Private Collaborative Learning')
parser.add_argument('--path', type=str,
                    # default=f'/home/{user}/capc-learning-imba',
                    default=f'/home/{user}/code/capc-learning',
                    help='path to the project')
# General parameters
parser.add_argument('--dataset', type=str,
                    default=dataset,
                    help='name of the dataset')
parser.add_argument(
    '--dataset_type',
    type=str,
    default='balanced',
    # default='imbalanced',
    help='Type of the dataset.')
parser.add_argument('--begin-id', type=int, default=0,
                    help='train private models with id number in [begin_id, end_id)')
parser.add_argument('--end-id', type=int, default=end_id,
                    help='train private models with id number in [begin_id, end_id)')
parser.add_argument('--num-querying-parties', type=int, default=3,
                    help='number of parties that pose queries')
parser.add_argument('--mode', type=str, default='random',
                    help='method for generating utility scores')
parser.add_argument('--weak-classes', type=str, default='1,2',
                    help='indices of weak classes')
parser.add_argument('--weak-class-ratio', type=float, default=0.1,
                    help='ratio of samples belonging to weak classes')
# Training parameters
parser.add_argument('--batch-size', type=int, default=batch_size,
                    help='batch size for training and evaluation')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=weight_decay,
                    help='L2 weight decay factor')
parser.add_argument('--seed', type=int, default=111, help='random seed')
parser.add_argument('--lr', type=float, default=lr,
                    help='initial learning rate')
parser.add_argument('--num-epochs', type=int, default=num_epochs,
                    help='number of epochs for training')
parser.add_argument(
    '--architecture',
    type=str,
    default=architecture,
    # default='ResNet10',
    help='Architecture of the model.')
parser.add_argument(
    '--cuda_visible_devices',
    type=int,
    default=0,
    # default=5,
    help='Cuda visible devices.')
parser.add_argument(
    '--scheduler_type',
    type=str,
    default='ReduceLROnPlateau',
    # default='MultiStepLR',
    help='Type of the scheduler.')
parser.add_argument(
    '--scheduler_milestones',
    nargs='+',
    type=int,
    default=None,
    help='The milestones for the multi-step scheduler.'
)
parser.add_argument(
    '--schedule_factor',
    type=float,
    default=0.1,
    help='The factor for scheduler.'
)
parser.add_argument(
    '--schedule_patience',
    type=int,
    default=10,
    help='The patience for scheduler.'
)
parser.add_argument(
    '--heterogenous_models',
    nargs='+',
    type=str,
    # default=['VGG16', 'VGG19', 'VGG5', 'VGG13', 'VGG11'],
    # default=['ResNet8', 'ResNet10'],
    # default=['VGG'],
    default=[architecture],
    help='The architectures of heterogenous models.',
)

# Privacy parameters
parser.add_argument('--num-models', type=int, default=num_models,
                    help='number of private models')
parser.add_argument('--threshold', type=float, default=threshold,
                    help='threshold value (a scalar) in the threshold mechanism')
parser.add_argument('--sigma-gnmax', type=float, default=sigma_gnmax,
                    help='std of the Gaussian noise in the GNMax mechanism')
parser.add_argument('--sigma-threshold', type=float, default=sigma_threshold,
                    help='std of the Gaussian noise in the threshold mechanism')
parser.add_argument('--budget', type=float, default=budget,
                    help='pre-defined epsilon value for (eps, delta)-DP')
parser.add_argument('--budgets', nargs="+",
                    type=float, default=budgets,
                    help='pre-defined epsilon value for (eps, delta)-DP')

# Command parameters (what to run).
parser.add_argument(
    '--commands',
    nargs='+',
    type=str,
    # default=['train_private_models'],
    # default=['query_ensemble_model', 'retrain_private_models'],
    default=commands,
    help='which command to run')

args = parser.parse_args()
utils.print_args(args=args)


# os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda_visible_devices}'

######################
# NON-PRIVATE MODELS #
######################

def train_non_private_model(args):
    """Train a non-private baseline model on the given dataset for comparison."""
    # Dataloaders
    trainloader = utils.load_training_data(args)
    evalloader = utils.load_evaluation_data(args)
    # Logs
    file = open(os.path.join(args.non_private_model_path,
                             'logs-(initial-lr:{:.2f})-(num-epochs:{:d}).txt'.format(
                                 args.lr, args.num_epochs)), 'w')
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Training a non-private model on '{}' dataset!".format(args.dataset),
        file)
    utils.augmented_print("Initial learning rate: {:.2f}".format(args.lr), file)
    utils.augmented_print(
        "Number of training epochs: {:d}".format(args.num_epochs), file)
    # Non-private model
    model = models.Private_Model('model(non-private)', args)
    if args.cuda:
        model.cuda()
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    decay_steps = [int(args.num_epochs * 0.5), int(args.num_epochs * 0.75),
                   int(args.num_epochs * 0.9)]
    # Training steps
    for epoch in range(args.num_epochs):
        if epoch in decay_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        utils.train(model, trainloader, optimizer, args)
        eval_loss, acc = utils.evaluate(model, evalloader, args)
        acc_detailed = utils.evaluate_detailed(model, evalloader, args)
        utils.augmented_print(
            "Epoch {:d} | Evaluation Loss: {:.4f} | Accuracy: {:.2f}% | Detailed Accuracy(%): {}".format(
                epoch + 1, eval_loss, acc,
                np.array2string(acc_detailed, precision=2, separator=', ')),
            file, flush=True)
    # Checkpoints
    state = {'epoch': args.num_epochs, 'accuracy': acc, 'eval_loss': eval_loss,
             'state_dict': model.state_dict()}
    filename = "checkpoint-{}.pth.tar".format(model.name)
    filepath = os.path.join(args.non_private_model_path, filename)
    torch.save(state, filepath)
    utils.augmented_print("##########################################", file)
    file.close()


###########################
# ORIGINAL PRIVATE MODELS #
###########################

def get_scheduler(args, optimizer, gamma=0.1):
    scheduler_type = args.scheduler_type
    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode='min',
            factor=args.schedule_factor,
            patience=args.schedule_patience)
    elif scheduler_type == 'MultiStepLR':
        milestones = args.scheduler_milestones
        if milestones is None:
            milestones = [int(args.num_epochs * 0.5),
                          int(args.num_epochs * 0.75),
                          int(args.num_epochs * 0.9)]
        scheduler = MultiStepLR(
            optimizer=optimizer, milestones=milestones,
            gamma=args.schedule_factor)
    else:
        raise Exception("Unknown scheduler type: {}".format(scheduler_type))
    return scheduler


def train_private_models(args):
    """Train N = num-models private models."""
    assert 0 <= args.begin_id and args.begin_id < args.end_id and args.end_id <= args.num_models
    # Logs
    filename = 'logs-(id:{:d}-{:d})-(num-epochs:{:d}).txt'.format(
        args.begin_id + 1, args.end_id, args.num_epochs)
    file = open(os.path.join(args.private_model_path, filename), 'w')
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Training private models on '{}' dataset!".format(args.dataset), file)
    utils.augmented_print(
        "Training private models on '{}' architecture!".format(
            args.architecture), file)
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file)
    utils.augmented_print("Initial learning rate: {:.2f}".format(args.lr), file)
    utils.augmented_print(
        "Number of epochs for training each model: {:d}".format(
            args.num_epochs), file)
    # Dataloaders
    if args.dataset_type == 'imbalanced':
        all_private_trainloaders = utils.load_private_data_imbalanced(args)
    elif args.dataset_type == 'balanced':
        all_private_trainloaders = utils.load_private_data(args)
    else:
        raise Exception('Unknown dataset type: {}'.format(args.dataset_type))
    evalloader = utils.load_evaluation_data(args)
    # Training
    all_acc = []
    for id in range(args.begin_id, args.end_id):
        utils.augmented_print("##########################################",
                              file)
        # Private model
        model = models.Private_Model('model({:d})'.format(id + 1), args)
        if args.cuda:
            model.cuda()
        trainloader = all_private_trainloaders[id]
        counts, ratios = utils.class_ratio(trainloader.dataset, args)
        utils.augmented_print(
            "Label counts: {}".format(np.array2string(counts, separator=', ')),
            file)
        utils.augmented_print("Class ratios: {}".format(
            np.array2string(ratios, precision=2, separator=', ')), file)
        utils.augmented_print(
            "Number of samples: {:d}".format(len(trainloader.dataset)), file)
        utils.augmented_print("Steps per epoch: {:d}".format(len(trainloader)),
                              file)
        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

        scheduler = get_scheduler(args=args, optimizer=optimizer)

        # Training steps
        for epoch in range(args.num_epochs):
            train_loss = utils.train(model, trainloader, optimizer, args)
            # Scheduler step is based only on the train data, we do not use the
            # test data to schedule the decrease in the learning rate.
            scheduler.step(train_loss)
        eval_loss, acc = utils.evaluate(model, evalloader, args)
        acc_detailed = utils.evaluate_detailed(model, evalloader, args)
        all_acc.append(acc)
        utils.augmented_print(
            "Model '{}' | Evaluation Loss: {:.4f} | Accuracy: {:.2f}% | Detailed Accuracy(%): {}".format(
                model.name, eval_loss, acc,
                np.array2string(acc_detailed, precision=2, separator=', ')),
            file, flush=True)
        # Checkpoints
        state = {'epoch': args.num_epochs, 'accuracy': acc,
                 'eval_loss': eval_loss, 'state_dict': model.state_dict()}
        filename = "checkpoint-{}.pth.tar".format(model.name)
        filepath = os.path.join(args.private_model_path, filename)
        torch.save(state, filepath)
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Average accuracy of private models: {:.2f}%".format(np.mean(all_acc)),
        file)
    utils.augmented_print("##########################################", file)
    file.close()


##################
# NOISY ENSEMBLE #
##################

def evaluate_ensemble_model(args):
    """Evaluate the accuracy of noisy ensemble model under varying noise scales."""
    # Logs
    file = open(
        os.path.join(args.ensemble_model_path, 'logs-ensemble(all).txt'), 'w')
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Evaluating ensemble model 'ensemble(all)' on '{}' dataset!".format(
            args.dataset), file)
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file)
    # Load private models
    ensemble = []
    for id in range(args.num_models):
        filename = "checkpoint-{}.pth.tar".format('model({:d})'.format(id + 1))
        filepath = os.path.join(args.private_model_path, filename)
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            model = models.Private_Model('model({:d})'.format(id + 1), args)
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model.cuda()
            model.eval()
            ensemble.append(model)
        else:
            raise Exception(
                "Checkpoint file '{}' does not exist, please generate it via 'train_private_models(args)'!".format(
                    filepath))
    # Create an ensemble model
    ensemble_model = models.Ensemble_Model('ensemble(all)', ensemble, args)
    # Evalloader
    evalloader = utils.load_evaluation_data(args)
    # Different sigma values
    sigma_list = [200, 150, 100, 50, 45, 40, 35, 30, 25, 20, 10, 5,
                  0]  # svhn 250 models
    # sigma_list = [40, 35, 30, 25, 20, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # cifar10 50 models
    for sigma in sigma_list:
        args.sigma_gnmax = sigma
        acc, acc_detailed, gap, gap_detailed = ensemble_model.evaluate(
            evalloader, args)
        utils.augmented_print("Sigma: {:.4f}".format(args.sigma_gnmax), file)
        utils.augmented_print("Accuracy on evalset: {:.2f}%".format(acc), file)
        utils.augmented_print("Detailed accuracy on evalset: {}".format(
            np.array2string(acc_detailed, precision=2, separator=', ')), file)
        utils.augmented_print("Gap on evalset: {:.2f}% ({:.2f}|{:d})".format(
            100. * gap / args.num_models, gap, args.num_models), file)
        utils.augmented_print("Detailed gap on evalset: {}".format(
            np.array2string(gap_detailed, precision=2, separator=', ')), file,
            flush=True)
    utils.augmented_print("##########################################", file)
    file.close()


################
# QUERY-ANSWER #
################

def compute_utility_scores_entropy(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset."""
    with torch.no_grad():
        # Entropy value as a proxy for utility.
        entropy = []
        for data, _ in dataloader:
            if args.cuda:
                data = data.cuda()
            data = Variable(data)
            output = model(data)
            prob = F.softmax(output, dim=1).cpu().numpy()
            entropy.append(scipy.stats.entropy(prob, axis=1))
        entropy = np.concatenate(entropy, axis=0)
        # Maximum entropy is achieved when the distribution is uniform.
        entropy_max = np.log(args.num_classes)
        # Sanity checks
        assert len(entropy.shape) == 1 and entropy.shape[0] == len(
            dataloader.dataset)
        assert np.all(entropy <= entropy_max) and np.all(0 <= entropy)
        # Normalize utility scores to [0, 1]
        utility = entropy / entropy_max
        # Save utility scores
        filename = "{}-utility-scores-(mode:entropy)".format(model.name)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, utility)
        return utility


def compute_utility_scores_gap(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset."""
    with torch.no_grad():
        # Gap between the probabilities of the two most probable classes as a proxy for utility.
        gap = []
        for data, _ in dataloader:
            if args.cuda:
                data = data.cuda()
            data = Variable(data)
            output = model(data)
            sorted_output = output.sort(dim=-1, descending=True)[0]
            prob = F.softmax(sorted_output[:, :2], dim=1).cpu().numpy()
            gap.append(prob[:, 0] - prob[:, 1])
        gap = np.concatenate(gap, axis=0)
        # Sanity checks
        assert len(gap.shape) == 1 and gap.shape[0] == len(dataloader.dataset)
        assert np.all(gap <= 1) and np.all(0 <= gap)
        # Convert gap values into utility scores
        utility = 1 - gap
        # Save utility scores
        filename = "{}-utility-scores-(mode:gap)".format(model.name)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, utility)
        return utility


def query_ensemble_model(args):
    """Query-answer process"""
    # Logs
    file_name = 'logs-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt'.format(
        args.num_models,
        args.num_querying_parties, args.mode,
        args.threshold, args.sigma_gnmax,
        args.sigma_threshold, args.budget)
    print('ensemble_model_path: ', args.ensemble_model_path)
    print('file_name: ', file_name)
    file = open(os.path.join(args.ensemble_model_path,
                             file_name), 'w')
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Query-answer process on '{}' dataset!".format(args.dataset), file)
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file)
    utils.augmented_print(
        "Number of querying parties: {:d}".format(args.num_querying_parties),
        file)
    utils.augmented_print("Querying mode: {}".format(args.mode), file)
    utils.augmented_print("Confidence threshold: {:.1f}".format(args.threshold),
                          file)
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the GNMax mechanism: {:.1f}".format(
            args.sigma_gnmax), file)
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the threshold mechanism: {:.1f}".format(
            args.sigma_threshold), file)
    utils.augmented_print(
        "Pre-defined privacy budget: ({:.2f}, {:.0e})-DP".format(args.budget,
                                                                 args.delta),
        file)
    utils.augmented_print("##########################################", file)
    # Load private models
    private_models = []
    if args.heterogenous_models is None:
        model_types = [args.architecture]
    else:
        model_types = args.heterogenous_models
    nr_model_types = len(model_types)
    for id in range(args.num_models):
        model_type = model_types[id % nr_model_types]
        print('model type: ', model_type, ' id: ', id)
        args.private_model_path = os.path.join(
            args.path, 'private-models',
            args.dataset, model_type, '{:d}-models'.format(
                args.num_models))
        filename = f"checkpoint-model({id + 1}).pth.tar"
        filepath = os.path.join(args.private_model_path, filename)
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            # print('file path: ', filepath)
            # print('checkpoint keys: ', checkpoint.keys())
            # print('state dict keys: ', checkpoint['state_dict'].keys())
            # sys.stdout.flush()
            args.architecture = model_type
            model = models.Private_Model('model({:d})'.format(id + 1), args)
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model.cuda()
            model.eval()
            private_models.append(model)
        else:
            raise Exception(
                "Checkpoint file '{}' does not exist, please generate it via 'train_private_models(args)'!".format(
                    filepath))
    # Querying parties
    parties_q = private_models[:args.num_querying_parties]
    # Answering parties
    parties_a = []
    for i in range(args.num_querying_parties):
        # For a given querying party, skip this very querying party as its
        # own answering party.
        parties_a.append(models.Ensemble_Model("ensemble({:d})".format(i + 1),
                                               private_models[
                                               :i] + private_models[i + 1:],
                                               args))
    # Compute utility scores and sort available queries
    utils.augmented_print("##########################################", file,
                          flush=True)
    # Utility function
    if args.mode == 'entropy':
        utility_function = compute_utility_scores_entropy
    elif args.mode == 'gap':
        utility_function = compute_utility_scores_gap
    else:
        assert args.mode == 'random'
    if args.mode != 'random':
        # Dataloaders
        unlabeled_dataloaders = utils.load_unlabeled_data(args)
        # Utility scores
        utility_scores = []
        for i in range(args.num_querying_parties):
            filename = "{}-utility-scores-(mode:{}).npy".format(
                parties_q[i].name, args.mode)
            filepath = os.path.join(args.ensemble_model_path, filename)
            if os.path.isfile(filepath):
                utils.augmented_print(
                    "Loading utility scores for '{}' in '{}' mode!".format(
                        parties_q[i].name, args.mode), file)
                utility = np.load(filepath)
            else:
                utils.augmented_print(
                    "Computing utility scores for '{}' in '{}' mode!".format(
                        parties_q[i].name, args.mode), file)
                utility = utility_function(parties_q[i],
                                           unlabeled_dataloaders[i], args)
            utility_scores.append(utility)
        # Sort unlabeled data according to their utility scores.
        all_indices = []
        for i in range(args.num_querying_parties):
            offset = i * (
                    args.num_unlabeled_samples // args.num_querying_parties)
            indices = utility_scores[i].argsort()[::-1] + offset
            all_indices.append(indices)
            assert len(set(indices)) == len(indices)
        assert len(set(
            np.concatenate(all_indices, axis=0))) == args.num_unlabeled_samples
    else:
        all_indices = []
        size = args.num_unlabeled_samples // args.num_querying_parties
        for i in range(args.num_querying_parties):
            begin = i * size
            end = args.num_unlabeled_samples if i == args.num_querying_parties - 1 else (
                                                                                                i + 1) * size
            all_indices.append(np.array(list(range(begin, end))))
        assert len(set(
            np.concatenate(all_indices, axis=0))) == args.num_unlabeled_samples
    utils.augmented_print("##########################################", file,
                          flush=True)
    utils.augmented_print(
        "Select queries according to their utility scores subject to the pre-defined privacy budget",
        file, flush=True)
    for i in range(args.num_querying_parties):
        # Raw ensemble votes
        filename = '{}-raw-votes-(mode:{}).npy'.format(parties_a[i].name,
                                                       args.mode)
        filepath = os.path.join(args.ensemble_model_path, filename)
        if os.path.isfile(filepath):
            utils.augmented_print(
                "Loading raw ensemble votes for '{}' in '{}' mode!".format(
                    parties_a[i].name, args.mode), file)
            votes = np.load(filepath)
        else:
            utils.augmented_print(
                "Generating raw ensemble votes for '{}' in '{}' mode!".format(
                    parties_a[i].name, args.mode), file)
            # Load unlabeled data according to a specific order
            unlabeled_dataloader_ordered = utils.load_ordered_unlabeled_data(
                args, all_indices[i])
            votes = parties_a[i].inference(unlabeled_dataloader_ordered, args)
        # Analyze how the pre-defined privacy budget will be exhausted when answering queries
        max_num_query, dp_eps, partition, answered, order_opt = analysis.analyze(
            votes, args.threshold, args.sigma_threshold, args.sigma_gnmax,
            args.budget, args.delta, file)
        utils.augmented_print("Querying party: {}".format(parties_q[i].name),
                              file)
        utils.augmented_print(
            "Maximum number of queries: {}".format(max_num_query), file)
        utils.augmented_print(
            "Privacy guarantee achieved: ({:.4f}, {:.0e})-DP".format(
                dp_eps[max_num_query - 1], args.delta), file)
        utils.augmented_print(
            "Expected number of queries answered: {:.3f}".format(
                answered[max_num_query - 1]), file)
        utils.augmented_print("Partition of privacy cost: {}".format(
            np.array2string(partition[max_num_query - 1], precision=3,
                            separator=', ')), file)

        utils.augmented_print("##########################################",
                              file,
                              flush=True)
        utils.augmented_print("Generate query-answer pairs.", file)
        indices_queried = all_indices[i][:max_num_query]
        queryloader = utils.load_ordered_unlabeled_data(args, indices_queried)
        indices_answered, acc, acc_detailed, gap, gap_detailed = parties_a[
            i].query(queryloader, args, indices_queried)
        utils.save_queries(args, indices_answered, parties_q[i].name)
        utils.save_raw_queries(args, indices_answered, parties_q[i].name)
        utils.augmented_print("Accuracy on queries: {:.2f}%".format(acc), file)
        utils.augmented_print("Detailed accuracy on queries: {}".format(
            np.array2string(acc_detailed, precision=2, separator=', ')), file)
        utils.augmented_print("Gap on queries: {:.2f}% ({:.2f}|{:d})".format(
            100. * gap / len(parties_a[i].models), gap,
            len(parties_a[i].models)), file)
        utils.augmented_print("Detailed gap on queries: {}".format(
            np.array2string(gap_detailed, precision=2, separator=', ')), file)

        utils.augmented_print("##########################################",
                              file,
                              flush=True)
        utils.augmented_print("Check query-answer pairs.", file)
        queryloader = utils.load_ordered_unlabeled_data(args, indices_answered)
        counts, ratios = utils.class_ratio(queryloader.dataset, args)
        utils.augmented_print(
            "Label counts: {}".format(np.array2string(counts, separator=', ')),
            file)
        utils.augmented_print("Class ratios: {}".format(
            np.array2string(ratios, precision=2, separator=', ')), file)
        utils.augmented_print(
            "Number of samples: {:d}".format(len(queryloader.dataset)), file)
        utils.augmented_print("##########################################",
                              file, flush=True)
    file.close()


############################
# RETRAINED PRIVATE MODELS #
############################

def retrain_private_models(args):
    """Retrain N = num-querying-parties private models."""
    assert 0 <= args.begin_id and args.begin_id < args.end_id and args.end_id <= args.num_querying_parties
    # Logs
    filename = 'logs-(num_models:{:d})-(id:{:d}-{:d})-(num-epochs:{:d})-(budget:{:f})-(dataset:{})-(architecture:{}).txt'.format(
        args.num_models,
        args.begin_id + 1, args.end_id,
        args.num_epochs,
        args.budget,
        args.dataset,
        args.architecture,
    )
    print('filename: ', filename)
    file = open(os.path.join(args.retrained_private_model_path, filename), 'w')
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Retraining the private models of all querying parties on '{}' dataset!".format(
            args.dataset), file)
    utils.augmented_print(
        "Number of querying parties: {:d}".format(args.num_querying_parties),
        file)
    utils.augmented_print("Initial learning rate: {:.2f}".format(args.lr), file)
    utils.augmented_print(
        "Number of epochs for retraining each model: {:d}".format(
            args.num_epochs), file)
    # Dataloaders
    # all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(args)
    all_augmented_dataloaders = utils.load_private_data_and_qap(args)
    evalloader = utils.load_evaluation_data(args)
    # Training
    for id in range(args.begin_id, args.end_id):
        utils.augmented_print("##########################################",
                              file)
        # Different random seed
        seed_list = [11, 13, 17, 113, 117]
        eval_loss_list = []
        acc_list = []
        acc_detailed_list = []
        for seed in seed_list:
            args.seed = seed
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # Private model
            model = models.Private_Model('model({:d})'.format(id + 1), args)
            if args.cuda:
                model.cuda()
            trainloader = all_augmented_dataloaders[id]
            # Optimizer
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            decay_steps = [int(args.num_epochs * 0.5),
                           int(args.num_epochs * 0.75),
                           int(args.num_epochs * 0.9)]
            # Training steps
            for epoch in range(args.num_epochs):
                if epoch in decay_steps:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1
                utils.train(model, trainloader, optimizer, args)
            eval_loss, acc = utils.evaluate(model, evalloader, args)
            acc_detailed = utils.evaluate_detailed(model, evalloader, args)
            eval_loss_list.append(eval_loss)
            acc_list.append(acc)
            acc_detailed_list.append(acc_detailed)
            # # Checkpoints
            # state = {'epoch': args.num_epochs, 'accuracy': acc, 'eval_loss': eval_loss, 'state_dict': model.state_dict()}
            # filename = "checkpoint-{}.pth.tar".format(model.name)
            # filepath = os.path.join(args.retrained_private_model_path, filename)
            # torch.save(state, filepath)
        eval_loss_list = np.array(eval_loss_list)
        acc_list = np.array(acc_list)
        acc_detailed_list = np.array(acc_detailed_list)
        counts, ratios = utils.class_ratio(trainloader.dataset, args)
        utils.augmented_print(
            "Label counts: {}".format(np.array2string(counts, separator=', ')),
            file)
        utils.augmented_print("Class ratios: {}".format(
            np.array2string(ratios, precision=2, separator=', ')), file)
        utils.augmented_print(
            "Number of samples: {:d}".format(len(trainloader.dataset)), file)
        utils.augmented_print("Steps per epoch: {:d}".format(len(trainloader)),
                              file)
        utils.augmented_print(
            "Model: '{}' | Architecture: '{}' | Dataset: '{}' | Number of models: {:d} | Budget: {:.4f} | Evaluation Loss: {:.4f} | Accuracy: {:.2f}% | Detailed Accuracy(%): {} | Standard Deviation(%): {}".format(
                model.name,
                args.architecture,
                args.dataset,
                args.num_models,
                args.budget,
                eval_loss_list.mean(axis=0), acc_list.mean(axis=0),
                np.array2string(acc_detailed_list.mean(axis=0), precision=2,
                                separator=', '),
                np.array2string(acc_detailed_list.std(axis=0), precision=2,
                                separator=', ')), file, flush=True)
    utils.augmented_print("##########################################", file)
    file.close()


def pytorch2pickle(args):
    for id in range(args.num_models):
        filename = "checkpoint-{}.pth.tar".format('model({:d})'.format(id + 1))
        filepath = os.path.join(args.private_model_path, filename)
        if os.path.isfile(filepath):
            # Load private model
            checkpoint = torch.load(filepath)
            model = models.Private_Model('model({:d})'.format(id + 1), args)
            model.load_state_dict(checkpoint['state_dict'])
            # Copy weights
            conv_count = 0
            bn_count = 0
            linear_count = 0
            weights = {}
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    key_weight = 'conv_{}_weight'.format(conv_count)
                    weights[key_weight] = m.weight.data.clone().cpu().numpy()
                    conv_count += 1
                elif isinstance(m, nn.BatchNorm2d):
                    key_gamma = 'bn_{}_gamma'.format(bn_count)
                    key_beta = 'bn_{}_beta'.format(bn_count)
                    key_rm = 'bn_{}_rm'.format(bn_count)
                    key_rv = 'bn_{}_rv'.format(bn_count)
                    weights[key_gamma] = m.weight.data.clone().cpu().numpy()
                    weights[key_beta] = m.bias.data.clone().cpu().numpy()
                    weights[key_rm] = m.running_mean.data.clone().cpu().numpy()
                    weights[key_rv] = m.running_var.data.clone().cpu().numpy()
                    bn_count += 1
                elif isinstance(m, nn.Linear):
                    key_weight = 'linear_{}_weight'.format(linear_count)
                    key_bias = 'linear_{}_bias'.format(linear_count)
                    weights[key_weight] = m.weight.data.clone().cpu().numpy()
                    weights[key_bias] = m.bias.data.clone().cpu().numpy()
                    linear_count += 1
            # Serialize weight dictionary
            filename = "checkpoint-{}.pickle".format(model.name)
            filepath = os.path.join(args.private_model_path, filename)
            with open(filepath, 'wb') as handle:
                pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Model {}'s weights serialized with success!".format(
                model.name))
        else:
            raise Exception(
                "Checkpoint file '{}' does not exist, please generate it via 'train_private_models(args)'!".format(
                    filepath))


def main(args):
    # CUDA support
    args.cuda = torch.cuda.is_available()
    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Dataset
    args.weak_classes = [int(c) for c in args.weak_classes.split(',')]
    args.dataset = args.dataset.lower()
    if args.dataset == 'mnist':
        args.dataset_path = os.path.join(args.path, 'MNIST')
        args.num_unlabeled_samples = 9000
        args.num_classes = 10
        # Hyper-parameter delta in (eps, delta)-DP.
        args.delta = 1e-5
    elif args.dataset == 'svhn':
        args.dataset_path = os.path.join(args.path, 'SVHN')
        args.num_unlabeled_samples = 10000
        args.num_classes = 10
        args.delta = 1e-6
    elif args.dataset == 'cifar10':
        args.dataset_path = os.path.join(args.path, 'CIFAR10')
        args.num_unlabeled_samples = 9000
        args.num_classes = 10
        args.delta = 1e-5
    else:
        raise Exception("Dataset name must be 'mnist', 'svhn' or 'cifar10'!")

    for model in args.heterogenous_models:
        args.architecture = model
        print('architecture: ', architecture)
        for num_models in [args.num_models]:
            # for num_models in [100, 300]:
            print('num_models: ', num_models)
            args.num_models = num_models

            # Folders
            args.private_model_path = os.path.join(
                args.path, 'private-models',
                args.dataset, args.architecture, '{:d}-models'.format(
                    args.num_models))
            print('args.private_model_path: ', args.private_model_path)

            args.ensemble_model_path = os.path.join(
                args.path, 'ensemble-models',
                args.dataset, args.architecture, '{:d}-models'.format(
                    args.num_models))
            args.non_private_model_path = os.path.join(
                args.path, 'non-private-models',
                args.dataset, args.architecture)
            # dir = [args.mode, 'threshold:{:.1f}'.format(args.threshold), 'sigma-gnmax:{:.1f}'.format(args.sigma_gnmax),
            #        'sigma-threshold:{:.1f}'.format(args.sigma_threshold), 'budget:{:.2f}'.format(args.budget)]
            args.retrained_private_model_path = os.path.join(
                args.path,
                'retrained-private-models',
                args.dataset,
                args.architecture,
                '{:d}-models'.format(
                    args.num_models),
                args.mode)

            print('args.retrained_private_models_path: ',
                  args.retrained_private_model_path)

            if not os.path.exists(args.private_model_path):
                os.makedirs(args.private_model_path)
            if not os.path.exists(args.ensemble_model_path):
                os.makedirs(args.ensemble_model_path)
            if not os.path.exists(args.non_private_model_path):
                os.makedirs(args.non_private_model_path)
            if not os.path.exists(args.retrained_private_model_path):
                os.makedirs(args.retrained_private_model_path)


            # for budget in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
            # for budget in [float('inf')]:
            for budget in args.budgets:
                # for budget in [args.budget]:
                # for budget in [2.8]:
                args.budget = budget
                print('main budget: ', budget)
                for command in args.commands:
                    if command == 'train_non_private_model':
                        train_non_private_model(args)
                    elif command == 'train_private_models':
                        train_private_models(args)
                    elif command == 'evaluate_ensemble_model':
                        evaluate_ensemble_model(args)
                    elif command == 'query_ensemble_model':
                        query_ensemble_model(args)
                    elif command == 'retrain_private_models':
                        retrain_private_models(args)
                    elif command == 'pytorch2pickle':
                        pytorch2pickle(args)
                    else:
                        raise Exception('Unknown command: {}'.format(args.command))


if __name__ == '__main__':
    main(args)
