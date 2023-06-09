#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   prepare_data.py
@Time    :   2022/09/12 09:19:20
@Author  :   Bo 
'''
import os
import numpy as np 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch 
import utils.utils as utils 


def _get_cifar(conf, name, root, split, transform, target_transform, download):
    """Args:
    conf: the configuration class 
    name: str, cifar10/cifar100 
    root: the location to save/load the dataset 
    split: "train" / "test" 
    transform: the data augmentation for training  
    target_transform: the data augmentation for testing 
    download: bool variable
    """
    is_train = True if "train" in split else False

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = (
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = (
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        )
        
    normalize = normalize if conf.pn_normalize else None

    # decide data type.
    if is_train:
        if conf.apply_transform:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    transforms.ToTensor(),
                ]
                + ([normalize] if normalize is not None else [])
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
        
    
def get_dataset(conf, name, datasets_path, split="train", transform=None, target_transform=None,
                download=True):
    """Args:
    conf: the configuration class 
    name: str, cifar10/cifar100 
    datasets_path: the location to save/load the dataset 
    split: "train" / "test" 
    transform: the data augmentation for training  
    target_transform: the data augmentation for testing 
    download: bool variable
    """
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)
    if name == "cifar10" or name == "cifar100":
        return _get_cifar(
            conf, name, root, split, transform, target_transform, download
            )
