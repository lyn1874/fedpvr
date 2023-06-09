#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   sample.py
@Time    :   2022/09/20 09:53:57
@Author  :   Bo 
'''
import configs.conf as const 
import os
import utils.utils as utils 
import numpy as np 
import data.create_dataset as create_dataset 
import data.prepare_cifar as prepare_cifar
import shutil 


def subsample(conf):
    seed_use = np.random.randint(0, 20000, 1)[0]
    conf.seed_use = seed_use
    np.random.seed(conf.seed_use)
    path_init = utils.get_path_init(conf.loc)
    conf = utils.get_dir_name(conf)
        
    model_dir = path_init + "/%s/%s/communication_round_%03d/" % (conf.folder_name, 
                                                                  conf.dir_name,
                                                                  conf.communication_round)
    print("Creating model dir")
    utils.create_dir(model_dir)
    utils.create_dir(model_dir.split("/communication_round")[0].replace(path_init, "../exp_data/"))
    
    selected_clients = utils.select_random_clients(conf.n_clients, conf.sample_ratio, model_dir)
    print("The selected clients", selected_clients)

    if conf.communication_round == 0:       
        conf.random_state = np.random.RandomState(seed_use)
        if "dsprite" not in conf.dataset:
            if conf.dataset == "cifar10" or conf.dataset == "cifar100":
                train_dataset = prepare_cifar.get_dataset(conf, conf.dataset, conf.image_path, split="train")
                test_dataset = prepare_cifar.get_dataset(conf, conf.dataset, conf.image_path, split="test")
            train_dataset_update, \
                val_dataset_update, \
                    test_dataset = create_dataset.define_val_dataset(conf, train_dataset, test_dataset)
            data_group = {"train": train_dataset_update, "val": val_dataset_update, "test": test_dataset}
            _, data_partitioner_use = create_dataset.define_data_loader(conf, dataset=data_group["train"], 
                                                                        localdata_id=0, 
                                                                        is_train=True, data_partitioner=None)  # this is only for getting a data_partitioner 

    elif conf.communication_round >= 2 and conf.sample_ratio == 1.0 and conf.communication_round - 2 <= 100: # remove the client checkpoints to free up some space
        path2remove = path_init + "/%s/%s/communication_round_%03d/" % (conf.folder_name, 
                                                                        conf.dir_name,
                                                                        conf.communication_round - 2)
        sub_files = [v for v in os.listdir(path2remove) if "client_id" in v]
        if len(sub_files) > 0:
            for v in sub_files:
                shutil.rmtree(path2remove + v)
                
    if conf.communication_round >= 2 and conf.sample_ratio != 1.0: # remove some of client checkpoint to free up some space
        if "fed_avg" in conf.folder_name or "fed_prox" in conf.folder_name:
            path2remove = path_init + "/%s/%s/communication_round_%03d/" % (conf.folder_name, 
                                                                            conf.dir_name,
                                                                            conf.communication_round - 2)
            sub_files = [v for v in os.listdir(path2remove) if "client_id" in v]
            if len(sub_files) > 0:
                for v in sub_files:
                    shutil.rmtree(path2remove + v)

    
if __name__ == "__main__":
    conf = const.give_fed_args() 
    subsample(conf)

    
    

