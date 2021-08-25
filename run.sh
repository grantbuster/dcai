#!/bin/bash
#SBATCH --account=seasiasolar
#SBATCH --output=./logs/dcai_gcb_08_%A.log
#SBATCH --error=./logs/dcai_gcb_08_%A.log
#SBATCH --time=120
#SBATCH --qos=high

python train.py $1 label_book
