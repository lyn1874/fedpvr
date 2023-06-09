#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   prepare_partition.py
@Time    :   2022/09/12 09:19:29
@Author  :   Bo 
'''
import functools
import torch
import numpy as np
import math 
import os
import torch.distributed as dist 
import pickle 
import utils.utils as utils 


class Partition(object):
    def __init__(self, data, index):
        """Get a subset of data based on the index
        Args:
            data: object, full datset 
            index: index, full dataset
        """
        self.data = data 
        self.index = index 
        print(len(self.data), len(self.index))
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, sub_index):
        """Args:
        sub_index: the sub index for a particular partition of the dataset
        """
        data_idx = self.index[sub_index]
        return self.data[data_idx]      
    
    
class DataSampler(object):
    def __init__(self, conf, data, data_scheme, data_percentage=None, selected_classes=None):
        """
        conf: configuration files, argparse
        data: dataset
        data_scheme: str, which scheme to use 
        data_percentage: float, the percentage of the data that we will use
        selected_classes: a list of integers, which classes to select       
        """  
        self.conf = conf 
        self.data = data 
        self.data_size = len(self.data)
        self.data_scheme = data_scheme
        self.data_percentage = data_percentage
        self.selected_classes = self.selected_classes
        
        self.indices = np.array([x for x in range(self.data_size)])
        self.sampled_indices = None 
        
    def sample_indices(self):
        if self.data_scheme == "random_sampling":
            self.sampled_indices = self.conf.random_state.choice(self.indices, size=int(self.data_size * self.data_percentage), replace=False) 
            # returning a subset of shuffled indices
    
    def use_indices(self, sampled_indices=None):
        assert sampled_indices is not None or self.sampled_indices is not None 
        indices_to_use = sampled_indices if sampled_indices is not None else self.sampled_indices
        return Partition(self.data, indices=indices_to_use)  # data is the entire dataset, but this indices to use is a subset
    
    
def build_non_iid_by_dirichlet(random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    """Args:
    random_state: numpy random state for sampling from dirichlet distribution
    indices2targets: [N, 2] [indices, class_labels]
    non_iid_alpha: a float value, the smaller the value is, the more non-iid the data is distributed 
    num_classes: the number of unique class labels, i.e., 10 for CIFAR10 
    num_indices: the length of the indices2targets 
    n_workers: number of clients
    """
    if n_workers == 40 and non_iid_alpha == 0.1:
        n_auxi_worker = 20 
    elif n_workers == 40 and non_iid_alpha == 0.01:
        n_auxi_worker = 10 
    elif n_workers == 100 and num_classes == 10:
        n_auxi_worker = 10 
    elif n_workers == 100 and num_classes == 100 and non_iid_alpha == 0.01:
        n_auxi_worker = 20
    else:
        n_auxi_worker = n_workers 
    random_state.shuffle(indices2targets)  # shuffle along the row-dimension
    
    from_index = 0 
    splitted_targets = []
    num_split = math.ceil(n_workers / n_auxi_worker)
    
    split_n_workers = [n_auxi_worker if idx < num_split - 1 else n_workers - n_auxi_worker * (num_split - 1) for idx in range(num_split)]
    split_ratios = [v / n_workers for v in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(ratio * num_indices)
        splitted_targets.append(np.array(indices2targets[from_index:to_index]))
        from_index = to_index
    
    idx_batch = []
    
    for idx, s_target in enumerate(splitted_targets):
        s_target_size = len(s_target)
        s_n_worker = split_n_workers[idx]
        min_size = 0 
        while min_size < int(0.5 * s_target_size / s_n_worker):
            _idx_batch = [[] for _ in range(s_n_worker)]
            for s_class in range(num_classes):
                map_orig_class_index = np.where(s_target[:, 1] == s_class)[0]
                idx_class = s_target[map_orig_class_index, 0]
                try:
                    proportions = random_state.dirichlet(np.repeat(non_iid_alpha, s_n_worker))
                    # proportions = [1 / s_n_worker for _ in range(s_n_worker)]  # perfect class imbalance situation
                    q = 0
                    for p, idx_j in zip(proportions, _idx_batch):
                        if len(idx_j) >= s_target_size / s_n_worker:
                            proportions[q] = 0 
                        q+=1                            
                    proportions = proportions / np.sum(proportions)
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1] 
                    split_baseon_proportion = np.split(idx_class, proportions)
                    for batch_idx, s_batch in enumerate(_idx_batch):
                        s_batch += split_baseon_proportion[batch_idx].tolist()
                        _idx_batch[batch_idx] = s_batch 
                    sizes = [len(s_batch) for s_batch in _idx_batch]
                    min_size = np.min(sizes)
                    # print("class label ", s_class, " minimum size", min_size)
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch

    
class DataPartitioner(object):
    def __init__(self, conf, data, partition_sizes, partition_type, 
                 consistent_indices=True, partition_obj=True,
                 ):
        """Args:
        conf: the configuration arguments 
        data: Partition object or data array
        partition_sizes: number of data per device, [Number clients]
        partition_type: str
        consistent_indices: bool. If True, the indices are broadcast to all the devices        
        Ops:
        Majority of this function is from https://github.com/epfml/federated-learning-public-code/tree/7e002ef5ff0d683dba3db48e2d088165499eb0b9/codes/FedDF-code
        """
        self.conf = conf 
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices
        
        path_init = utils.get_path_init("home")
        self.folder_name = conf.folder_name 
        self.path2save_index = "%s/%s/%s/sampling_index.obj" % (path_init, self.folder_name, self.conf.dir_name)
    
        self.partitions = []
        
        if partition_obj == False: 
            self.data_size = len(data.targets)
            self.data = data 
            indices = np.array([x for x in range(self.data_size)])
        else:        
            self.data_size = len(data.index)
            self.data = data.data 
            indices = data.index 
        self.partition_indices(indices)
            
    def partition_indices(self, indices):
        # if self.conf.graph.rank == 0:
        indices = self._create_indices(indices)  # server create indices (I am not sure that I understand this part)
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)
        from_index = 0 
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index         
            
    def _create_indices(self, indices):
        if self.partition_type == "original":
            pass 
        elif self.partition_type == "random":
            self.conf.random_state.shuffle(indices)
        elif self.partition_type == "sort":
            indices2targets = np.array([(idx, s_target) for idx, s_target in enumerate(self.data.targets) if idx in indices])
            indices = indices2targets[np.argsort(indices2targets[:, 1]), 0]
            # indices = np.argsort(self.data.targets[indices])
            
        elif self.partition_type == "non_iid":
            if not os.path.isfile(self.path2save_index):
                num_class = len(np.unique(self.data.targets))
                num_indices = len(indices)
                num_workers = len(self.partition_sizes)
                indices2targets = np.array([(idx, s_target) for idx, s_target in enumerate(self.data.targets) if idx in indices])
                # okay, so this step deals with the actual training data from define_val_data 
                list_of_indices = build_non_iid_by_dirichlet(random_state=self.conf.random_state, 
                                                            indices2targets=indices2targets, 
                                                            non_iid_alpha=self.conf.non_iid_alpha, 
                                                            num_classes=num_class, num_indices=num_indices,
                                                            n_workers=num_workers)
                print("save sampling index with seed", self.conf.seed_use)
                with open("../exp_data/%s/%s/" % (self.folder_name, self.conf.dir_name) + "sampling_index.obj", "wb") as f:
                    pickle.dump(list_of_indices, f)
            else:
                print("Load the saved index", self.path2save_index, " from non_iid")
                list_of_indices = pickle.load(open(self.path2save_index, "rb"))
                
            indices = functools.reduce(lambda a, b: a+b, list_of_indices)  # concatenate over the list of indices 
        else:
            raise NotImplementedError("The partition type %s is not implemented yet" % self.partition_type)
        return indices 
    
    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices
            
    def use(self, partition_id):
        return Partition(self.data, self.partitions[partition_id])
            
        
        
        
        
        
