#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/09/12 13:54:02
@Author  :   Bo 
'''
import torch 
import numpy as np 
import os
import torch.nn as nn 
from tqdm import tqdm 

import fed_model.cls_model as cls_model 
import opt.opt as opt 

import data.create_dataset as create_dataset 
import data.prepare_cifar as prepare_cifar

import configs.conf as const 

import utils.utils as utils 
import sys


device=torch.device("cuda")


class Train(object):
    def __init__(self, conf, model_dir, data_group, ckpt_dir, num_local_epochs):
        
        self.conf = conf 
        self.model_dir = model_dir 
        self.data_group = data_group 
        self.num_local_epochs = num_local_epochs
        a = torch.randn([1]).to(device)
        
        self.model_use = cls_model.get_model(conf, device=torch.device("cuda"))
        
        if ckpt_dir:
            ckpt_use = torch.load(ckpt_dir, map_location=device)
            self.model_use.load_state_dict(ckpt_use)
        
        self.optimizer = opt.define_optimizer(conf, self.model_use, conf.optimizer_name, 
                                              lr=conf.lr)
                
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.tr_data_loader, self.val_data_loader, self.tt_data_loader = data_group
        
        print("Training data size", len(self.tr_data_loader))
        print("Validating data size", len(self.val_data_loader))
        print("Testing data size", len(self.tt_data_loader))
        
        parameter_list = [p for p in self.model_use.parameters() if p.requires_grad == False]
        assert len(parameter_list) == 0 
        self.saved_model_iters = {}

    def _update_lr(self, global_epoch):
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]["lr"] = self.conf.lr
        
    def _update_batch_tr(self, _image, _label, global_step):
        _data_batch = create_dataset.load_data_batch(_image, _label, is_on_cuda=True)
        self.optimizer.zero_grad()
        _pred = self.model_use(_data_batch["input"])
        _loss = self.loss_fn(_pred, _data_batch["target"]) / len(_image)
        _loss.backward()
        self.optimizer.step()
                
        accu = (_pred.argmax(axis=-1) == _data_batch["target"]).sum().div(len(_image))
        print("Training loss: {:.4f} and Training accuracy {:.2f}".format(_loss.item(), accu.item()))
        
    def _eval(self, global_step, data_use, str_use):
        self.model_use.eval()
        val_loss, val_accu = 0.0, 0.0
        for i, (_image, _label) in enumerate(data_use):
            _data = create_dataset.load_data_batch(_image, _label, is_on_cuda=True)
            _pred = self.model_use(_data["input"])
            _loss = self.loss_fn(_pred, _data["target"]) 
            _accu = (_pred.argmax(axis=-1) == _data["target"]).sum()
            val_loss += _loss.detach().cpu().numpy()
            val_accu += _accu.detach().cpu().numpy()
        print("{} loss: {:.4f} and {} accuracy {:.2f}".format(str_use, val_loss / len(data_use)/len(_image),
                                                                str_use, val_accu / len(data_use) / len(_image)))
        return val_loss, val_accu / len(data_use) / len(_image)
    
    def _select_top_model(self, global_step, val_loss):
        num_save=1
        if len(self.saved_model_iters.keys()) < num_save:
            self._save_model(global_step, val_loss)
            self.saved_model_iters["%d" % global_step] = val_loss 
        else:
            key_g = list(self.saved_model_iters.keys())
            val_accu_g = [self.saved_model_iters[k] for k in key_g]
            if val_loss <= np.max(val_accu_g):                
                _ind = np.argmax(val_accu_g)
                del self.saved_model_iters[key_g[_ind]]
                self.saved_model_iters["%d" % global_step] = val_loss  
                self._save_model(global_step, val_loss)
                ckpt_remove = [v for v in os.listdir(self.model_dir) if "model-{:05d}-{:.4f}.pt".format(int(key_g[_ind]), 
                                                                                                        np.min(val_accu_g)) in v][0]
                if os.path.isfile(self.model_dir + "/" + ckpt_remove):
                    os.remove(self.model_dir + "/" + ckpt_remove)                
    
    def _save_model(self, epoch, val_accu):
        torch.save(self.model_use.state_dict(), self.model_dir + "/model-{:05d}-{:.4f}.pt".format(epoch, val_accu))
                            
    def run(self):
        global_step = 0 
        for _, epoch in enumerate(range(self.num_local_epochs+1)[1:]):
            self.model_use.train()
            self._update_lr(epoch)
            for i, (_image, _label) in enumerate(self.tr_data_loader):
                if i == 0 and epoch == 1:
                    print(_image.shape, _label.shape)
                self._update_batch_tr(_image, _label, global_step)
                global_step += 1 
            if epoch % 2 == 0:                
                _val_loss, _val_accu = self._eval(global_step, self.val_data_loader, "validation")
                self._select_top_model(global_step, _val_loss)
                
        self._eval(global_step, self.tt_data_loader, "testing")
                        
        if np.isnan(_val_loss) == False:
            self._save_model(global_step, _val_loss)            

            
            
def run_train(conf):       
    if conf.dataset == "cifar10" or conf.dataset == "cifar100":
        train_dataset = prepare_cifar.get_dataset(conf, conf.dataset, conf.image_path, split="train")
        test_dataset = prepare_cifar.get_dataset(conf, conf.dataset, conf.image_path, split="test")
    
    print("==========================================")
    print("The shape of the training data", len(train_dataset), np.unique(train_dataset.targets, return_counts=True))
    print("The shape of the testing data", len(test_dataset), np.unique(test_dataset.targets, return_counts=True))

    train_dataset_update, \
        val_dataset_update, \
            test_dataset = create_dataset.define_val_dataset(conf, train_dataset, test_dataset)
    data_group = {"train": train_dataset_update, "val": val_dataset_update, "test": test_dataset}
    
    _, data_partitioner_use = create_dataset.define_data_loader(conf, dataset=data_group["train"], 
                                                                localdata_id=0, 
                                                                is_train=True, data_partitioner=None)  # this is only for getting a data_partitioner 
    tr_data_loader, _ = create_dataset.define_data_loader(conf, dataset=data_group["train"],
                                                        localdata_id=conf.use_local_id, 
                                                        is_train=True, data_partitioner=data_partitioner_use)
    
    val_data_loader, _ = create_dataset.define_data_loader(conf, dataset=data_group["val"],
                                                        localdata_id=0, 
                                                        is_train=False)
    tt_data_loader, _ = create_dataset.define_data_loader(conf, dataset=data_group["test"], 
                                                        localdata_id=0, 
                                                        is_train=False)
    data_loader_group = [tr_data_loader, val_data_loader, tt_data_loader]
    
    for i, (s_im, s_la) in enumerate(tr_data_loader):
        print(s_im.shape, s_la.shape)
        if i >= 1:
            break
            
    if conf.communication_round > 0:
        ckpt_dir = conf.model_dir.split("client_id")[0].replace("communication_round_%03d" % conf.communication_round, 
                                                                "communication_round_%03d" % (conf.communication_round - 1)) + "aggregated_model.pt"
    else:
        ckpt_dir = None    
        
    Train(conf, conf.model_dir, data_loader_group, ckpt_dir, conf.local_n_epochs).run()

   
def train_with_conf(conf):    
    path_init = utils.get_path_init(conf.loc)
        
    conf = utils.get_dir_name(conf)
    model_dir = path_init + "/%s/%s/communication_round_%03d/" % (conf.folder_name,
                                                                          conf.dir_name,
                                                                          conf.communication_round)
    
    
    selected_clients = utils.select_random_clients(conf.n_clients, conf.sample_ratio, model_dir)
    
    conf.selected_clients = selected_clients

    print("The selected clients", selected_clients)
    print("Local id", conf.use_local_id)

    model_dir += "client_id_%02d/" % selected_clients[conf.use_local_id] 
    
    
    conf.use_local_id = selected_clients[conf.use_local_id]
    
    conf.model_dir = model_dir 
    
    stdoutOrigin = sys.stdout
    
    sys.stdout = open(conf.model_dir + "training_statistics.txt", 'w')

    if conf.communication_round == 0:
        conf.random_state = np.random.RandomState(conf.seed_use)
        utils.seed_everything(conf.seed_use)
    else:
        conf.seed_use = np.random.randint(1, 20000, [1])[0]
        conf.random_state = np.random.RandomState(conf.seed_use)
        utils.seed_everything(conf.seed_use)
                        
    for arg in vars(conf):
        print(arg, getattr(conf, arg))
    run_train(conf)
    a = torch.zeros([1]).to(device)
    
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    
    
if __name__ == "__main__":
    conf = const.give_fed_args()
    train_with_conf(conf)

    

    
    
