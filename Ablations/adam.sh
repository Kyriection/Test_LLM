#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=adam_ab_fp4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=./logs_1b/adam_ab_fp4.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
cd ../

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_60m.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr 0.001 \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --optimizer adamw \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-ablations \
    --name 60m_adam_fp4 \
    --save_dir /scratch-shared/HTJ/1b-model-stablespam-fp4 \
    --restore_optimizer \
    --fp4 


