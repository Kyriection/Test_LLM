#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=adam_fp4_1bm_5e-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=./logs_1b/adam_fp4_1b_5e-4.out

# module purge
# module load 2023
# source activate spam
# Your job starts in the directory where you call sbatch
cd ../
source activate spam
wandb login --relogin 0831d62c2353a87b23a963dcb28ecc86cef378ee
torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_1b.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 64 \
    --total_batch_size 512 \
    --lr 0.0005 \
    --warmup_steps 2000 \
    --num_training_steps 20000 \
    --optimizer adamw \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-c4 \
    --name adam_fp4-1b_5e-4_noclip \
    --save_dir /scratch-shared/HTJ/1b-model-baseline-fp4 \
    --restore_optimizer \
    --fp4

