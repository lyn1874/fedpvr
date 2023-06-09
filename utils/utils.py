#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/12/08 13:25:56
@Author  :   Bo 
'''
import torch 
import random 
import numpy as np 
import os 


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def create_single_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        

def get_dir_name(conf):
    dir_name = "non_iid_alpha_%.3f_lr_%.3f_version_%d_%s_n_clients_%d_arch_%s" % (
        conf.non_iid_alpha, conf.lr, conf.version, conf.dataset, conf.n_clients, 
        conf.arch)
    if conf.sample_ratio != 1.0:
        dir_name += "_sample_%.2f" % conf.sample_ratio
    folder_name = "data_align_%s" %  conf.aggregation
    conf.folder_name = folder_name 
    conf.dir_name = dir_name
    return conf    


def get_replace_for_init_path(loc):
    if loc == "nobackup":
        rep = "/nobackup/blia/"
    elif loc == "scratch":
        rep = "/scratch/blia/"
    else:
        rep = "../"
    return rep 


def get_path_init(loc):
    if loc == "nobackup":
        return "/nobackup/blia/exp_data/"
    elif loc == "home":
        return "../exp_data/"
    elif loc == "scratch":
        return "/scratch/blia/exp_data/"


def select_random_clients(num_clients, ratio, model_dir):
    if len([v for v in os.listdir(model_dir) if "client_id" in v]) == 0:
        num_select = int(num_clients * ratio)
        select_clients = sorted(np.random.choice(np.arange(num_clients), num_select, replace=False))
        for v in select_clients:
            create_single_dir(model_dir + "/client_id_%02d/" % v)
    else:
        select_clients = sorted([int(v.split("client_id_")[1]) for v in os.listdir(model_dir) if "client_id" in v])
    return select_clients    






    
