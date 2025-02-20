#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=60m_spam_2-5e-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=./logs/60m_spam_2-5e-3.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
cd ../

for lr in 2e-3 3e-3 4e-3 5e-3
do
torchrun --standalone --nproc_per_node 2 main_pretrain.py \
    --model_config configs/llama_130m.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr $lr \
    --warmup_steps 2000 \
    --num_training_steps 10000 \
    --optimizer spam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project lrs_130m_stablespam \
    --restore_optimizer \
    --act_quant \
    --act_group_size 64 \
    --act_topk 2 \
    --act_stochastic 
done