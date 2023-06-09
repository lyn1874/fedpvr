#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   opt.py
@Time    :   2022/09/12 12:42:06
@Author  :   Bo 
'''
import torch 


def define_optimizer(conf, model, optimizer_name, lr=None):
    # define the param to optimize.
    weight_decay_group = {}
    iterr_index = 0 
    lr_group = {}
    momentum_group = {}
    if conf.aggregation == "fed_avg" or conf.aggregation == "fed_dyn" or conf.aggregation == "centralised" or conf.aggregation == "fed_prox":
        for key, value in model.named_parameters():
            lr_use = conf.lr 
            lr_group[key] = lr_use 
            momentum_group[key] = conf.momentum_factor 
            weight_decay_group[key] = conf.weight_decay
    elif conf.aggregation == "scaffold" or conf.aggregation == "fed_pvr":
        for key, value in model.named_parameters():
            lr_group[key] = conf.lr
            if iterr_index < conf.start_layer:
                _momentum = conf.momentum_factor 
            else:
                _momentum = 0.0 
            momentum_group[key] = _momentum
            weight_decay_group[key] = 0.0 
            iterr_index += 1 
                
    print("The weight decay situation")
    print(weight_decay_group)
    print("The learning rate situation")
    print(lr_group)
    print("The momentum factor situation ")
    print(momentum_group)
    
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": weight_decay_group[key],
            "param_size": value.size(),
            "nelement": value.nelement(),
            "lr": lr_group[key],
            "momentum": momentum_group[key],
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            nesterov=conf.use_nesterov,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=conf.lr if lr is None else lr,
            betas=(conf.adam_beta_1, conf.adam_beta_2),
            eps=conf.adam_eps,
        )
    else:
        raise NotImplementedError
    return optimizer


