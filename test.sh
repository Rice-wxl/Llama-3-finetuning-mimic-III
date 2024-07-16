#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=96G
#SBATCH -J evaluation
#SBATCH -o evaluation_icd_20k_cyclic2_1e-4_16_2nd.out
#SBATCH -e evaluation_icd_20k_cyclic2_1e-4_16_2nd.err

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate llm_finetune

python evaluation.py 