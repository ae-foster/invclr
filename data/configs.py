import json

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from collections import defaultdict

from data.augmentation import ColourDistortion, TensorNormalise, ModuleCompose
from data.spirograph import DrawSpirograph
from data.dataset import *
from models import *


def get_mean_std(dataset):
    CACHED_MEAN_STD = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
    }
    return CACHED_MEAN_STD[dataset]


def get_datasets(dataset, augment_clf_train=False, add_indices_to_data=False, augment_test=False,
                 train_proportion=1.,s=0.5):
    if dataset == 'spirograph':
        return get_spirograph_dataset(train_proportion=train_proportion)
    else:
        return get_img_datasets(dataset=dataset, augment_clf_train=augment_clf_train,
                                add_indices_to_data=add_indices_to_data, augment_test=augment_test,
                                train_proportion=train_proportion, s=s)


def get_spirograph_dataset(train_proportion=1., rgb_fore_bounds=(.4, 1), rgb_back_bounds=(0, .6), h_bounds=(.5, 2.5)):

    spirograph = DrawSpirograph(['m', 'b', 'sigma', 'rfore'], ['h', 'rback', 'gfore', 'gback', 'bfore', 'bback'],
                                rgb_fore_bounds= rgb_fore_bounds, rgb_back_bounds=rgb_back_bounds, h_bounds=h_bounds,
                                train_proportion=train_proportion)
    stem = StemCIFAR
    trainset, clftrainset, testset = spirograph.dataset()
    return trainset, testset, clftrainset, 3, stem, spirograph, spirograph


def get_root(dataset):
    PATHS = {
        'cifar10': '/data/cifar10/',
        'cifar100': '/data/cifar100/',
    }
    try:
        with open('dataset-paths.json', 'r') as f:
            local_paths = json.load(f)
            PATHS.update(local_paths)
    except FileNotFoundError:
        pass
    root = PATHS[dataset]
    return root


def get_img_datasets(dataset, augment_clf_train=False, add_indices_to_data=False, augment_test=False,
                     train_proportion=1., s=0.5):

    root = get_root(dataset)

    # Data
    img_size = 32

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    if augment_test:
        transform_test = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_mean_std(dataset)),
        ])

    if augment_clf_train:
        transform_clftrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_clftrain = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*get_mean_std(dataset)),
        ])

    return get_datasets_from_transform(dataset, root, transform_train, transform_test, transform_clftrain,
                                       add_indices_to_data=add_indices_to_data, train_proportion=train_proportion, s=s)


def get_datasets_from_transform(dataset, root, transform_train, transform_test, transform_clftrain,
                                add_indices_to_data=False, train_proportion=1., s=0.5):

    if dataset == 'cifar100':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR100)
        else:
            dset = torchvision.datasets.CIFAR100
        trainset = CIFAR100Biaugment(root=root, train=True, download=True, transform=transform_train)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        num_classes = 100
        stem = StemCIFAR
    elif dataset == 'cifar10':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR10)
        else:
            dset = torchvision.datasets.CIFAR10
        trainset = CIFAR10Biaugment(root=root, train=True, download=True, transform=transform_train)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        num_classes = 10
        stem = StemCIFAR
    else:
        raise ValueError("Bad dataset value: {}".format(dataset))

    if train_proportion < 1.:
        trainset = make_stratified_subset(trainset, train_proportion)
        clftrainset = make_stratified_subset(clftrainset, train_proportion)

    col_distort = ColourDistortion(s=s)
    batch_transform = ModuleCompose([
        col_distort,
        TensorNormalise(*get_mean_std(dataset))
    ])

    return trainset, testset, clftrainset, num_classes, stem, col_distort, batch_transform


def make_stratified_subset(trainset, train_proportion):

    target_n_per_task = int(len(trainset) * train_proportion / len(trainset.classes))
    target_length = target_n_per_task * len(trainset.classes)
    indices = []
    counts = defaultdict(lambda: 0)
    for i in torch.randperm(len(trainset)):
        y = trainset.targets[i]
        if counts[y] < target_n_per_task:
            indices.append(i)
            counts[y] += 1
        if len(indices) >= target_length:
            break
    return Subset(trainset, indices)
