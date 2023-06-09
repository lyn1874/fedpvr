#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT

non_iid_alpha=${1?:Error: what is the non_iid_ratio}
v=${2?:Error: the version }
lr_init=${3?:Error: default lr is 0.05}
aggregation=${4?:Error: use scaffold or not}
weight_decay=${5?:Error: the weight decay for the bottleneck}
loc_n_epoch=${6?:Error: the number epoch per round}
model_arch=${7:-fc}
momentum_factor=${8:-0}
n_clients=${9:-18}
start=${10:-0}
repeat_gpu=${11:-8}
dataset=${12:-dsprint}
start_layer=${13:-16}
loc=${14:-scratch}
batch_size=${15:-128}

num2=8
num3=16
num4=24
num5=32
lr_schedule=constant
use_wandb=false
 

end_commu=101


if [ "$non_iid_alpha" == 0 ]; then
    partition_type=sort 
else
    partition_type=non_iid 
fi


if [ "$aggregation" == fed_pvr ]; then 
    start_layer=16
    momentum_factor=0.9
else    
    start_layer=0
    momentum_factor=0.0
fi 



for j in $(seq "$start" 1 "$end_commu")
do
    python3 sample.py --local_n_epochs "$loc_n_epoch" \
        --lr "$lr_init" --non_iid_alpha "$non_iid_alpha" --weight_decay "$weight_decay" \
        --communication_round "$j" \
        --lr_schedule "$lr_schedule" --version "$v" --loc "$loc" --arch "$model_arch" \
        --n_clients "$n_clients" --aggregation "$aggregation" \
        --partition_type "$partition_type" --data "$dataset" \
        --start_layer "$start_layer" \

    for i in $(seq 0 1 "$((n_clients-1))")
    do
        if [ "$i" -lt "$num2" ]; then
            gpu_index="$i"
        elif [ "$i" -ge "$num2" ] && [ "$i" -lt "$num3" ]; then 
            gpu_index="$((i-repeat_gpu))"
        elif [ "$i" -ge "$num3" ] && [ "$i" -lt "$num4" ]; then 
            gpu_index="$((i-num3))"
        elif [ "$i" -ge "$num4" ] && [ "$i" -lt "$num5" ]; then 
            gpu_index="$((i-num4))"
        else
            gpu_index="$((i-num5))"
        fi
        echo "$gpu_index"
        export CUDA_VISIBLE_DEVICES="$gpu_index"
        
        python3 create_train.py --use_local_id "$i" --local_n_epochs "$loc_n_epoch" \
            --lr "$lr_init" --non_iid_alpha "$non_iid_alpha" --weight_decay "$weight_decay" \
            --communication_round "$j" --loc "$loc" \
            --lr_schedule "$lr_schedule" --version "$v" --arch "$model_arch" \
            --partition_type "$partition_type" \
            --data "$dataset" \
            --start_layer "$start_layer" --batch_size "$batch_size" \
            --n_clients "$n_clients" --aggregation "$aggregation" --momentum_factor "$momentum_factor" &
    done
    wait 

    echo "Done training all the clients"
    for i in $(seq 0 1 "$((num2-1))")
    do
        export CUDA_VISIBLE_DEVICES="$i"
        if [ "$i" == 0 ]; then 
            worker_for_occupy_gpu=false
        else
            worker_for_occupy_gpu=true 
        fi 
        python3 communicate.py --use_local_id "$i" --local_n_epochs "$loc_n_epoch" \
            --lr "$lr_init" --non_iid_alpha "$non_iid_alpha" --weight_decay "$weight_decay" \
            --communication_round "$j" --loc "$loc" \
            --lr_schedule "$lr_schedule" --version "$v" --arch "$model_arch" \
            --partition_type "$partition_type" \
            --data "$dataset" --worker_for_occupy_gpu "$worker_for_occupy_gpu" \
            --start_layer "$start_layer" \
            --n_clients "$n_clients" --aggregation "$aggregation" --momentum_factor "$momentum_factor" &
    done
    wait 
    echo "Done communicating"
done