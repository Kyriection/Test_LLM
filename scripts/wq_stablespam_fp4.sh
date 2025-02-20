#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=130m_stablespam_fp_lr2-5e-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=./logs/130m_stablespam_fp_lr2-5e-3.out

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
    --save_every 100000 \
    --dtype bfloat16 \
    --batch_size 256 \
    --total_batch_size 512 \
    --lr 0.001 \
    --warmup_steps 2000 \
    --num_training_steps 10000 \
    --optimizer stablespam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-x \
    --name test-q-stable-spam \
    --save_dir llama_130m_adamw_qweight_simulate \
    --restore_optimizer \
    --gamma1 0.85 \
    --gamma2 0.999 \
    --theta 0.999 \
    --update_proj_gap 500 \
    --fp4
done