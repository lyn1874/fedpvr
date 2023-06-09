#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   get_correction.py
@Time    :   2022/09/22 10:18:26
@Author  :   Bo 
'''
import torch 
import numpy as np 
import os 
import pickle 
import copy 


def get_fake_ci_and_c(model_use):
    server_correction = {}
    client_correction = {}
    for name, p in model_use.named_parameters():
        if p.requires_grad:
            server_correction[name] = []
            client_correction[name] = []
    return server_correction, client_correction


def set_correction_to_zeros(model_use, device):
    correction = {}
    for name, p in model_use.named_parameters():
        if p.requires_grad:
            correction[name] = torch.zeros_like(p.data).to(p.data.dtype).to(device)
    for name in correction.keys():
        correction[name].requires_grad = False 
    return correction 


def look_for_most_recent_client_correction(path, client_id, model_use, device):
    """Args:
    path: i.e., /nobackup/blia/exp_data/client_drift_scaffold/communication_round_../
    client_id: the client id that is being looked    
    model_use: if the client id doesn't exist, we need to initialize the client_correction to zeros
    """
    path_mom = path.split("communication_round")[0]
    communication_rounds = sorted([v for v in os.listdir(path_mom) if "communication_round" in v])[:-1]
    sub_folder = [v for v in communication_rounds if "client_id_%02d" % client_id in os.listdir(path_mom + v)]
    if len(sub_folder) == 0:
        print("initialise client correction on client %d with zeros" % client_id)
        client_correction = set_correction_to_zeros(model_use, device)
    else:
        print("initialise client %d with the saved client correction" % client_id, sub_folder[-1])
        client_path = path_mom + "/" + sub_folder[-1] + "/client_id_%02d/client_correction_%02d.obj" % (client_id, client_id)
        client_correction = pickle.load(open(client_path, "rb"))
    return client_correction


def get_client_correction_and_server_correction(model_use, p_path, conf, device):
    print("The path to load the server and client corrections\n", 
          p_path)
    if conf.communication_round == 0:
        client_correction = set_correction_to_zeros(model_use, device)
        server_correction = set_correction_to_zeros(model_use, device)
    else:
        if "aggregate" in p_path and ".pt" in p_path:
            p_path = p_path.split("aggregate")[0]
        server_correction = pickle.load(open(p_path + "/server_correction.obj", "rb"))  # previous communication rounds 
        if conf.sample_ratio == 1:
            client_correction = pickle.load(open(p_path + "/client_id_%02d/client_correction_%02d.obj" % (conf.use_local_id, 
                                                                                                          conf.use_local_id), 
                                                 "rb"))
        else:
            client_correction = look_for_most_recent_client_correction(p_path, 
                                                                       conf.use_local_id, 
                                                                       model_use, device)
        for k in server_correction.keys():
            server_correction[k] = server_correction[k].to(device)
            client_correction[k] = client_correction[k].to(device)

    for k in client_correction.keys():
        assert client_correction[k].requires_grad == False 
        assert server_correction[k].requires_grad == False 
    return client_correction, server_correction

    
def layer_wise_scaffold(correction_term, conf, device):
    """This function set the client correction and server correction before the conf.start_layer to 0
    """
    correction_term_update = copy.deepcopy(correction_term)
    if conf.layer_wise_correction == True:
        for i, key in enumerate(correction_term_update.keys()):
            if i < conf.start_layer:
                s_value = correction_term_update[key]
                correction_term_update[key] = torch.zeros_like(s_value).to(s_value.dtype).to(device)
    return correction_term_update


def verify_layerwise_activation(correction_term, conf):
    if conf.layer_wise_correction == True:
        for i, k in enumerate(correction_term.keys()):
            if i < conf.start_layer:
                assert correction_term[k].abs().sum() == 0
                print("correctiion succed", k)
            else:
                print("I am not correcting this param", k)
            
            
def load_gradients_for_feddyn(p_path, conf, device, model_use):
    """This function loads the gradients from the previous communication round
    for a specific client. p_path should be the previous round. 
    """        
    if conf.communication_round > 0:
        if conf.sample_ratio == 1.0:
            assert int(p_path.split("communication_round_")[1].replace("/", "")) == conf.communication_round - 1
            client_grad = pickle.load(open(p_path + "/client_id_%02d/client_gradients_%02d.obj" % (conf.use_local_id,
                                                                                                conf.use_local_id), "rb"))
        else:
            client_grad = look_for_most_recent_gradient(p_path, conf.use_local_id, model_use, device)
    else:
        client_grad = set_grad_to_zeros(model_use)
    client_grad = client_grad.to(device)
    assert client_grad.requires_grad == False
    return client_grad 

def set_grad_to_zeros(model_use):
    client_grad = []
    for name, p in model_use.named_parameters():
        if p.requires_grad:
            client_grad.append(torch.zeros_like(p).view(-1))
    client_grad = torch.cat(client_grad, dim=0)        
    return client_grad 

def look_for_most_recent_gradient(path, client_id, model_use, device):
    """Args:
    path: i.e., /nobackup/blia/exp_data/client_drift_scaffold/communication_round_../
    client_id: the client id that is being looked    
    model_use: if the client id doesn't exist, we need to initialize the client_correction to zeros
    """
    path_mom = path.split("communication_round")[0]
    communication_rounds = sorted([v for v in os.listdir(path_mom) if "communication_round" in v])[:-1]
    sub_folder = [v for v in communication_rounds if "client_id_%02d" % client_id in os.listdir(path_mom + v)]
    if len(sub_folder) == 0:
        print("initialise client gradients on client %d with zeros" % client_id)
        client_grad = set_grad_to_zeros(model_use)
    else:
        print("initialise client %d with the saved client correction" % client_id, sub_folder[-1])
        grad_path = path_mom + "/" + sub_folder[-1] + "/client_id_%02d/client_gradients_%02d.obj" % (client_id,
                                                                                                     client_id)
        client_grad = pickle.load(open(grad_path, "rb"))
    
    return client_grad