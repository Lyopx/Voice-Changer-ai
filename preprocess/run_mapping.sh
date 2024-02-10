#!/bin/bash
#SBATCH --job-name=mapping
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=p-V100

# python extract_mcep.py
python extract_whisper.py