#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   cls_model.py
@Time    :   2022/12/08 13:14:53
@Author  :   Bo 
'''
import torch 
import fed_model.vgg as vgg 
import fed_model.resnet as resnet 
    
    
def get_model(conf, device):
    if "VGG" in conf.arch:
        model_obj = vgg.vgg(conf)
        model_obj.to(device)
    elif "resnet" in conf.arch:
        model_obj = resnet.resnet(conf)
        model_obj.to(device)
    return model_obj 


def get_model_params(dataset, arch):
    class PARAM:
        dataset = "cifar10"
    conf = PARAM 
    conf.dataset = dataset
    conf.arch = arch 
    conf.vgg_scaling = None     
    model_use = get_model(conf, torch.device("cpu"))
    param_size, title_group = [], []
    for name, p in model_use.named_parameters():
        if p.requires_grad and "bias" not in name:
            param_size.append(p.shape)
            title_group.append(name)
    return param_size, title_group 