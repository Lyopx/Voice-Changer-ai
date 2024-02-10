#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=p-V100

# ===================== Training for M4Singer (target: ProSinger) =====================
python -u main.py --debug False \
--dataset 'ProSinger' --model 'Transformer' \
--lr 1e-4 --batch_size 32 --epochs 400

# ===================== Training for M4Singer (target: Opencpop) =====================
# python -u main.py --debug False \
# --dataset 'Opencpop' --model 'Transformer' \
# --lr 1e-4 --batch_size 32 --epochs 400