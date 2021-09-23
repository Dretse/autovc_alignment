#!/bin/bash
declare x
if [ $# -eq 0 ]; then
    x=0
else
    x=$1
fi

echo starting resnet training $x
nohup env CUDA_VISIBLE_DEVICES=0 python training_fastresnet_half.py --num_expe $x &

#echo starting resnet training 2
#nohup env CUDA_VISIBLE_DEVICES=1 python training_halfresnet.py --num_expe 1 &

#rm logs/$expe_name.log
#touch logs/$expe_name.log
#echo "Start of logging :" > logs/$expe_name.log
#env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/$expe_name.yaml
#nohup env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/$expe_name.yaml --logging logs/$expe_name.log
#sleep 1 
#tail -f logs/$expe_name.log
#nohup env CUDA_VISIBLE_DEVICES=1 python main.py --config configs/first_exp.yaml  &> logs/first_exp.log &
