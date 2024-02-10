#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=p-RTX2080
#SBATCH --output=run_inference_prosinger.out

# ===================== Inference for M4Singer (target: ProSinger) =====================
python -u main.py --debug False --evaluate True \
--dataset 'M4Singer' --converse True --model 'Transformer' \
--resume '/mntnfs/lee_data1/haoyan/SLP/Singing-Voice-Conversion/model/ckpts/ProSinger/Transformer_lr_0.0001/132.pt'

# ===================== Inference for M4Singer (target: Opencpop) =====================
# python -u main.py --debug False --evaluate True \
# --dataset 'M4Singer' --converse True --model 'Transformer' \
# --resume '/mntnfs/lee_data1/haoyan/SLP/Singing-Voice-Conversion/model/ckpts/Opencpop/Transformer_lr_0.0001/104.pt'