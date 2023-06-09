#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   communicate.py
@Time    :   2022/09/14 15:46:31
@Author  :   Bo 
'''
import numpy as np 
import os 
import aggregate.fed_avg as fed_avg 
import aggregate.scaffold as scaffold 
import configs.conf as const 
import torch 

import create_train as create_train
import utils.utils as utils 



def find_fail_experiment(conf, model_dir):
    path_use = model_dir + "/communication_round_%03d/" % conf.communication_round 
    client_subset = sorted([v for v in os.listdir(path_use) if "client_id_" in v])
    for i, s_client in enumerate(client_subset):
        num_ckpt = [v for v in os.listdir(path_use + "/" + s_client) if ".pt" in v]
        if len(num_ckpt) >= 1:
            print("Client %s at communication round %d is succeed" % (s_client, conf.communication_round))
        else:
            print("Somehow, client id %s at communication round %d failed" % (s_client, conf.communication_round))
            print("===========================Retrain=========================")
            conf.use_local_id = i
            create_train.run(conf)
            
    return np.array(sorted([int(v.split("_")[-1]) for v in client_subset]))


if __name__ == "__main__":
    conf = const.give_fed_args()
    path_init = utils.get_path_init(conf.loc)
    conf = utils.get_dir_name(conf)
    
    model_dir = path_init + "/%s/%s/" % (conf.folder_name, conf.dir_name) 
    print("---------------start to communicate %d---------------" % conf.communication_round)
    a = torch.zeros([2]).to(torch.device("cuda"))
    
    if conf.worker_for_occupy_gpu == False:
        select_client = find_fail_experiment(conf, model_dir)
        if conf.aggregation == "fed_avg" or conf.aggregation == "fed_prox":
            agg_obj = fed_avg.Aggregator(conf, model_dir, select_client=select_client, communication_round=conf.communication_round, 
                                         evaluate_on_tt=True, device=torch.device("cuda"))
        elif conf.aggregation == "scaffold" or conf.aggregation == "fed_pvr":
            agg_obj = scaffold.Aggregator(conf, model_dir, select_client=select_client, communication_round=conf.communication_round, 
                                          evaluate_on_tt=True, device=torch.device("cuda"))
    else:
        for i in range(100000):
            a = torch.zeros([1]).to(torch.device("cuda"))
    
    