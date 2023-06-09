#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   create_dataset.py
@Time    :   2022/09/12 09:18:15
@Author  :   Bo 
'''
import numpy as np 
import torch
import os
import data.prepare_partition as pp 
import data.prepare_cifar as prepare_cifar
import utils.utils as utils 


def load_data_batch(_input, _target, is_on_cuda=True):
    """Args:
    conf: argparse
    _input: [batch_size, channel, imh, imw]
    _target: [batch_size]
    """
    if is_on_cuda == True:
        _input, _target = _input.cuda(), _target.cuda()
    _data_batch = {"input": _input, "target": _target}
    return _data_batch


def define_val_dataset(conf, train_dataset, test_dataset):
    """Args:
    train_dataset: dataset class 
    test_dataset: dataset class 
    """
    assert conf.val_data_ratio >= 0

    partition_sizes = [
        (1 - conf.val_data_ratio) * conf.train_data_ratio,
        (1 - conf.val_data_ratio) * (1 - conf.train_data_ratio),
        conf.val_data_ratio,
    ]
    data_partitioner = pp.DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="original",
        consistent_indices=False,
        partition_obj=False
    )
    
    tr_index = data_partitioner.partitions[0]
    val_index = data_partitioner.partitions[2]
    print("The number of overlapped index between training and validating", np.sum([v for v in val_index if v in tr_index]))
    
    train_dataset = data_partitioner.use(0)
    # split for val data.
    if conf.val_data_ratio > 0:
        assert conf.partitioned_by_user is False
        val_dataset = data_partitioner.use(2)
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, None, test_dataset


def define_data_loader(conf, dataset, localdata_id=None, is_train=True, shuffle=True, 
                       data_partitioner=None, drop_last=True):
    """Args: 
    conf: the argparse 
    dataset: a dictionary or i.e., train_dataset, val_dataset from the define_dataset function
    localdata_id: client id
    is_train: bool variable 
    shuffle: bool variable 
    data_partitioner: a class: pp.DataPartitioner
    Ops:
    during training, conf.user_original_client_data is always "combine". However, during testing, it
    can be set as "only_real" or "only_fake" for evaluating the gradients
    """
    if is_train:
        world_size = conf.n_clients 
        partition_size = [1.0 / world_size for _ in range(conf.n_clients)]
        assert localdata_id is not None 
        if data_partitioner is None:
            data_partitioner = pp.DataPartitioner(conf, dataset, partition_sizes=partition_size,
                                                  partition_type=conf.partition_type)
        data_to_load = data_partitioner.use(localdata_id)
    else:
        data_to_load = dataset 
        
    batch_size = conf.batch_size
    data_loader = torch.utils.data.DataLoader(data_to_load, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=conf.num_workers, 
                                            pin_memory=True,
                                            drop_last=drop_last)
    conf.num_batches_per_device_per_epoch = len(data_loader)
    conf.num_whole_batches_per_worker = (
        conf.num_batches_per_device_per_epoch * conf.local_n_epochs
    )    
    return data_loader, data_partitioner


def get_test_dataset(conf, shuffle=False):
    utils.seed_everything(conf.seed_use)
    test_dataset = prepare_cifar.get_dataset(conf, conf.dataset,  conf.image_path, split="test")
    tt_data_loader, _ = define_data_loader(conf, dataset=test_dataset, 
                                        localdata_id=0, 
                                        is_train=False,
                                        shuffle=shuffle,
                                        drop_last=False)
    return test_dataset, tt_data_loader 

