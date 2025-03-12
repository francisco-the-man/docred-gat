#!/bin/bash
#SBATCH --job-name=GAT_inference
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=averylou@stanford.edu

# Load required modules
module load python/3.9
module load cuda/11.7
module load cudnn/8.6.0.163

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# Create logs directory if it doesn't exist
mkdir -p logs

# Run inference with best configuration
# Replace BEST_LAYERS, BEST_HEADS with your best values from tuning
python run.py \
    --model_name_or_path roberta-large \
    --transformer_type roberta \
    --data_dir DocRED \
    --test_file test.json \
    --test_batch_size 8 \
    --gnn_layers BEST_LAYERS \
    --gat_heads BEST_HEADS \
    --load_path checkpoints/gat_lBEST_LAYERS_hBEST_HEADS_lrBEST_LR/best.ckpt \
    --do_test \
    --save_path results/best_model_predictions

# Deactivate virtual environment
deactivate