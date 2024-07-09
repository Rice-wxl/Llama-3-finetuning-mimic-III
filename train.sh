#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=48G
#SBATCH -J finetune_llama3
#SBATCH -o finetune_llama3_%j.out
#SBATCH -e finetune_llama3_%j.err

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate llm_finetune

python run.py 