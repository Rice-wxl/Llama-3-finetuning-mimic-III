#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=48G
#SBATCH -J finetune_llama3
#SBATCH -o icd-20k-cyclic2-1e-4-batch16-run2.out
#SBATCH -e icd-20k-cyclic2-1e-4-batch16-run2.out

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate llm_finetune

python notes_icd.py 