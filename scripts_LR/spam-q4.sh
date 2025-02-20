#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=spam_60m_q4_lrs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --output=./logs_1b/spam_60m_q4_lrs.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
wandb login --relogin 0831d62c2353a87b23a963dcb28ecc86cef378ee
cd ..

for lr in 5e-3 4e-3 3e-3 2e-3 1e-3 5e-4   
do
torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_60m.json \
    --eval_every 1000 \
    --save_every 100000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr $lr \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --optimizer spam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-x \
    --name spam_60m_q4_lrs \
    --save_dir /scratch-shared/HTJ/llama_130m_adamw_qweight_simulate \
    --restore_optimizer \
    --update_proj_gap 500  \
    --act_quant \
    --act_group_size 64 \
    --act_topk 2 \
    --act_stochastic 
done