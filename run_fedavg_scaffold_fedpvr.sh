#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT
g=${1?:-8}
data=${2?:-cifar100}
v_g=${3:-30}
n_devices=${4:-10}
loc=${5:-nobackup}
agg=${6:-fed_avg}


if [ "$agg" == fed_avg ]; then 
    if [ "$data" == cifar10 ]; then 
        lr_alpha_h=0.05
    elif [ "$data" == cifar100 ]; then 
        lr_alpha_h=0.1
    fi 
elif [ "$agg" == scaffold ]; then 
    if [ "$data" == cifar10 ]; then 
        lr_alpha_h=0.05
    elif [ "$data" == cifar100 ]; then 
        lr_alpha_h=0.1
    fi 
elif [ "$agg" == fed_pvr ]; then 
    if [ "$data" == cifar10 ]; then 
        lr_alpha_h=0.05
    elif [ "$data" == cifar100 ]; then 
        lr_alpha_h=0.1
    fi 
fi 


for s_v in $v_g 
do
    ./brun.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 10 VGG_11 0 10 0 "$g" "$data" 0 "$loc" 
done 
