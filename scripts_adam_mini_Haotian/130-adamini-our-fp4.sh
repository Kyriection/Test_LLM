#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=adam_mini_130_fp4_our
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=./logs_1b/adam_mini_130_fp4_our.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
cd ../

wandb login --relogin 652c840880355982b883464461fb41868153b9a6

torchrun --standalone --nproc_per_node 2 main_pretrain.py \
    --model_config configs/llama_130m.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 256 \
    --total_batch_size 512 \
    --lr 0.001 \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --optimizer adam_mini_our \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-c4 \
    --name 130_adam_mini_our_fp4 \
    --save_dir /scratch-shared/HTJ/1b-model-baseline-fp4 \
    --restore_optimizer \
    --gamma1 0.7 \
    --gamma2 0.9 \
    --theta 0.999 \
    --fp4

