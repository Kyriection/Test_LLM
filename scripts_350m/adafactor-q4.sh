#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=adafactor_1b_q4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=./logs_1b/adafactor_1b_q4.out

# module purge
# module load 2023
# source activate spam
# Your job starts in the directory where you call sbatch
cd ../

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_350m.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr 0.001 \
    --warmup_steps 2000 \
    --num_training_steps 20000 \
    --optimizer adafactor \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-c4 \
    --name 1b-baseline-fp4 \
    --save_dir /scratch-shared/HTJ/1b-model-adafactor-q4 \
    --restore_optimizer \
    --act_quant \
    --act_group_size 64 \
    --act_topk 2 \
    --act_stochastic 

