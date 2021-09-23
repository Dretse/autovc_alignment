#!/bin/bash


echo starting $1 experiment
nohup env CUDA_VISIBLE_DEVICES=0 python main.py --config configs/$1.yaml --logging logs/$1.log &
echo starting $2 experiment
nohup env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/$2.yaml --logging logs/$2.log &
wait
python plotting_graphs.py --logfile $1
python plotting_graphs.py --logfile $2

#rm logs/$expe_name.log
#touch logs/$expe_name.log
#echo "Start of logging :" > logs/$expe_name.log
#env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/$expe_name.yaml
#nohup env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/$expe_name.yaml --logging logs/$expe_name.log
#sleep 1 
#tail -f logs/$expe_name.log
#nohup env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/first_exp.yaml  &> logs/first_exp.log &
