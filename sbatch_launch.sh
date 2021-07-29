#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres gpu:gtx1080:1
#SBATCH --job-name testing_slurm
#SBATCH --time 03-00
#SBATCH --mem 20G
#SBATCH -o logs/$1.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.thebaud@orange.com

bash launch.sh $1
