#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   create_train.py
@Time    :   2022/10/10 14:11:23
@Author  :   Bo 
'''
import train_fed_avg as train_fed_avg 
import train_scaffold as train_scaffold 
import configs.conf as const 
import torch 


device = torch.device("cuda")


def run(conf):
    if conf.aggregation == "fed_avg":
        train_fed_avg.train_with_conf(conf)
    elif conf.aggregation == "scaffold":
        train_scaffold.train_with_conf(conf)
    elif conf.aggregation == "fed_pvr":
        train_scaffold.train_with_conf(conf)
        
if __name__ == "__main__":
    a = torch.zeros([1]).to(device)
    conf = const.give_fed_args()
    run(conf)