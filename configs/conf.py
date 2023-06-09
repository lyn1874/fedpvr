"""
Created on 14:58 at 24/11/2021
@author: bo
"""
import argparse
import numpy as np 


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_fed_args():
    parser = argparse.ArgumentParser(description='VAE-Reconstruction')
    parser.add_argument('--dataset', type=str, default="Shape")
    parser.add_argument("--num_class", default=18, type=int)

    parser.add_argument("--arch", default="fc", type=str)
    parser.add_argument("--sample_ratio", default=1.0, type=float)
    
    parser.add_argument("--val_data_ratio", default=0.1, type=float)
    parser.add_argument("--train_data_ratio", default=1.0, type=float)
    
    # parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    
    parser.add_argument("--num_workers", default=4, type=int)    

    parser.add_argument("--optimizer_name", default="sgd", type=str)

    parser.add_argument("--momentum_factor", default=0, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
        
    parser.add_argument("--local_n_epochs", default=10, type=int)
    parser.add_argument("--lr_schedule", default="constant", type=str)
    parser.add_argument("--use_local_id", default=0, type=int, help="which client am I using")
    
    parser.add_argument("--n_clients", default=10, type=int)
    parser.add_argument("--selected_clients")
    parser.add_argument("--partition_type", default="non_iid", type=str)
    parser.add_argument("--non_iid_alpha", default=100, type=float)
    parser.add_argument("--partitioned_by_user", default=False, type=str2bool)
    parser.add_argument("--seed_use", type=int, default=7)
    parser.add_argument("--communication_round", type=int, default=0)
    
    parser.add_argument("--dir_name", type=str)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--loc", default="nobackup", type=str)
    parser.add_argument("--random_shuffle", default=True, type=str2bool)
    
    parser.add_argument("--aggregation", default="fed_avg", type=str)
    parser.add_argument("--start_layer", default=16, type=int)
    
    parser.add_argument("--worker_for_occupy_gpu", default=False, type=str2bool)
    
    parser.add_argument("--vgg_scaling", default=None, type=int)
        
    parser.add_argument("--freeze_bn", default=False, type=str2bool)
    parser.add_argument("--freeze_bn_affine", default=False, type=str2bool)
    parser.add_argument("--group_norm_num_groups", default=2, type=int)
        
    parser.add_argument("--pn_normalize", default=True, type=str2bool)
    parser.add_argument("--apply_transform", default=True, type=str2bool)
    
    parser.add_argument("--save_gradients", default=False, type=str2bool)
    parser.add_argument("--ciplus_option", default="option_two", type=str)
    parser.add_argument("--layer_wise_correction", default=True, type=str2bool)
    parser.add_argument("--free_up_space", default=True, type=str2bool)
    
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument("--centralised", default=False, type=str2bool)
    parser.add_argument("--mu_prox", default=0.01, type=float)
    
    parser.add_argument("--image_path", type=str, default="../image_dataset/")

    return parser.parse_args()


