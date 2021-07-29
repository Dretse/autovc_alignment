#!/bin/bash

for wait_time in "$@"
do
    echo starting $wait_time
    { sleep $wait_time; echo waking up after $wait_time seconds; } &
    wait
    echo end of it
done

#rm logs/$expe_name.log
#touch logs/$expe_name.log
#echo "Start of logging :" > logs/$expe_name.log
#env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/$expe_name.yaml
#nohup env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/$expe_name.yaml --logging logs/$expe_name.log
#sleep 1 
#tail -f logs/$expe_name.log
#nohup env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/first_exp.yaml  &> logs/first_exp.log &