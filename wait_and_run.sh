#!/bin/bash
echo sleeping $1 seconds
sleep $1
echo end of sleep launching experiment
nohup bash launch.sh 300ep_neck8_emb1 300ep_neck8_emb01 &
