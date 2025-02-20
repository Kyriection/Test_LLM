#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=spam_1b_fp4_ab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00
#SBATCH --output=./logs_1b/spam_350m_fp4_ab.out

# module purge
# module load 2023
# source activate spam
# Your job starts in the directory where you call sbatch
cd ../

source activate spam
torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_350m.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 64 \
    --total_batch_size 512 \
    --lr 0.002 \
    --warmup_steps 1000 \
    --num_training_steps 20000 \
    --optimizer spam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-ablations \
    --name 350m-spam-fp4 \
    --save_dir /scratch-shared/HTJ/1b_model-spam-fp4 \
    --restore_optimizer \
    --fp4 