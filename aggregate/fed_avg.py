#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   fed_avg.py
@Time    :   2022/09/14 14:36:03
@Author  :   Bo 
'''
import torch 
import torch.nn as nn 
import os 
import numpy as np
import pickle 
import fed_model.cls_model as cls_model 
import utils.utils as utils 
import data.create_dataset as create_dataset


device=torch.device("cuda") 


class Aggregator(object):
    def __init__(self, conf, path, select_client=[], communication_round=0, 
                 evaluate_on_tt=False, device=torch.device("cuda")):
        self.conf = conf 
        self.path = path 
        self.n_clients = conf.n_clients 
        self.communication_round = communication_round
        self.path_use = self.path + "/communication_round_%03d/" % self.communication_round
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
        rep = utils.get_replace_for_init_path(conf.loc)
        
        path2save = self.path.replace(rep, "../") + "statistics.obj"
        
        if not os.path.exists(path2save.replace("/statistics.obj", "")):
            os.makedirs(path2save.replace("statistics.obj", ""))
        if communication_round == 0:
            self.stat = {}
            for k in range(self.n_clients):
                self.stat["client_%02d_accuracy" % (k)] = []
                self.stat["client_%02d_loss" % k] = []
            self.stat["aggregate_accuracy"] = []
            self.stat["aggregate_loss"] = []  
        else:          
            self.stat = pickle.load(open(path2save, "rb"))
        
        if len(select_client) == 0:
            select_client = np.arange(self.n_clients)
        self.select_client = select_client
        self.ratio = torch.FloatTensor([1.0 / len(self.select_client) for _ in self.select_client]).to(device)
        self.tt_data_loader = self._get_data()
        self.ckpt_dir = self._get_ckpt_dir()
        self.weights_group = self._get_model(evaluate_on_tt)
        print("-------------------save model----------------------")
        torch.save(self.weights_group, self.path_use + "/aggregated_model.pt")
        with open(path2save, "wb") as f:
            pickle.dump(self.stat, f)
        

    def _get_ckpt_dir(self):
        sub_path = sorted([self.path_use + v for v in os.listdir(self.path_use) if "client_id" in v and int(v.split("_")[-1]) in self.select_client])
        assert len(sub_path) == len(self.select_client)
        ckpt_dir = []
        for i, s_path in enumerate(sub_path):
            all_ckpt = [s_path + "/" + v for v in os.listdir(s_path) if ".pt" in v]
            accu = [int(v.split("-")[1]) for v in all_ckpt]
            ckpt_dir.append(all_ckpt[np.argmax(accu)])
            print(all_ckpt[np.argmax(accu)])
        return ckpt_dir 
    
    def _get_model(self, evaluate=False):
        weights_group = {}
        for i, s_ckpt in enumerate(self.ckpt_dir):
            model_value = torch.load(s_ckpt, map_location=device)
            if evaluate:
                _, _, accu, loss = self._evaluate_model(model_value)
                self.stat["client_%02d_accuracy" % self.select_client[i]].append(accu.detach().cpu().numpy())
                self.stat["client_%02d_loss" % self.select_client[i]].append(loss.detach().cpu().numpy())
            if i == 0:
                for k in model_value.keys():
                    weights_group[k] = model_value[k] * self.ratio[i]
            else:
                for k in model_value.keys():
                    weights_group[k] += model_value[k] * self.ratio[i]
            del model_value 
        pred, tt_label, accu, loss = self._evaluate_model(weights_group)        
        self.stat["aggregate_accuracy"].append(accu.detach().cpu().numpy())
        self.stat["aggregate_loss"].append(loss.detach().cpu().numpy())
        return weights_group
    
    def _get_data(self):
        _, tt_data_loader = create_dataset.get_test_dataset(self.conf, shuffle=False)
        return tt_data_loader 
    
    def _evaluate_model(self, model_param):
        model_use = cls_model.get_model(self.conf, device)
        model_use.eval()
        model_use.load_state_dict(model_param)
        model_use.requires_grad_(False)
        model_use.to(device)
        accu = 0.0 
        num_data = 0.0
        tt_loss = 0.0
        pred, tt_label = [], []
        for i, s_data in enumerate(self.tt_data_loader):
            _x, _y = s_data[0].to(device), s_data[1].to(device)
            _pred = model_use(_x)
            tt_loss += self.loss_func(_pred, _y)
            accu += (_pred.argmax(axis=-1) == _y).sum()
            num_data += len(_x)
            pred.append(_pred.argmax(axis=-1).detach().cpu().numpy())
            tt_label.append(_y.detach().cpu().numpy())
        return np.concatenate(pred), np.concatenate(tt_label), accu / num_data, tt_loss / num_data
        
        
