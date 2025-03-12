#!/bin/bash
#SBATCH --job-name=tune_GAT
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/tune_GAT_%j.out
#SBATCH --error=logs/tune_GAT_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=averylou@stanford.edu

# Load required modules
module purge
module load python/3.9
module load cuda/11.7
module load cudnn/8.6.0.163

# Set Python path to include user site-packages
export PYTHONPATH=/home/users/averylou/.local/lib/python3.9/site-packages:$PYTHONPATH

# Remove existing installations
pip uninstall -y numpy scikit-learn matplotlib contourpy


# Install numpy first and verify
pip install --user numpy==1.22.4
echo "Checking numpy installation..."
pip list | grep numpy
python -c "import numpy; print(numpy.__version__)"

# Install scikit-learn with compatible version
pip install --user scikit-learn

# Install other packages
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install --user networkx<3.0
pip install --user torch-geometric
pip install --user transformers
pip install --user wandb
pip install --user tqdm
pip install --user opt_einsum
pip install --user ujson
pip install --user pandas
pip install --user datetime
pip install --user pickle-mixin

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Create directories
mkdir -p logs
mkdir -p checkpoints

# Set WANDB API key and mode
export WANDB_API_KEY="dadf02e826360d567b6a67f767d340555535d378"
export WANDB_MODE="offline"

# Verify Python can import numpy before running main script
python -c "import numpy; print('Numpy import successful')"

# Run hyperparameter tuning for GAT
for gat_layers in 2 3 4; do
    for gat_heads in 4 8 12; do
        for lr in 1e-5 2e-5 5e-5; do
            echo "Training with GAT layers=${gat_layers}, heads=${gat_heads}, lr=${lr}"
            
            python run.py \
                --model_name_or_path roberta-large \
                --transformer_type roberta \
                --data_dir DocRED \
                --train_file train_annotated.json \
                --dev_file dev.json \
                --test_file test.json \
                --train_batch_size 4 \
                --test_batch_size 8 \
                --gradient_accumulation_steps 1 \
                --num_train_epochs 20 \
                --learning_rate ${lr} \
                --max_grad_norm 1.0 \
                --warmup_ratio 0.06 \
                --num_labels 4 \
                --gnn_layers ${gat_layers} \
                --gat_heads ${gat_heads} \
                --do_train \
                --evaluation_steps 2500 \
                --save_path "checkpoints/gat_l${gat_layers}_h${gat_heads}_lr${lr}" \
                --display_name "GAT_l${gat_layers}_h${gat_heads}_lr${lr}"
        done
    done
done
