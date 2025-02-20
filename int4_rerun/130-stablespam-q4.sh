#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=stablespam_130_int4_1e-3_500_0.999_0.5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=15:00:00
#SBATCH --output=./logs_1b/stablespam_130_int4_1e-3_500_0.999_0.5.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
wandb login --relogin 0831d62c2353a87b23a963dcb28ecc86cef378ee
cd ..

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_130m.json \
    --eval_every 1000 \
    --save_every 100000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr 0.001 \
    --warmup_steps 2000 \
    --num_training_steps 20000 \
    --optimizer stablespam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project stablespam \
    --name 130-stablespam-int4_1e-3_500_0.999_0.5 \
    --save_dir llama_130m_adamw_qweight_simulate \
    --restore_optimizer \
    --act_quant \
    --act_group_size 64 \
    --act_topk 2 \
    --act_stochastic \
    --gamma1 0.5 \
    --gamma2 0.999 \
    --theta 0.0 \
    --update_proj_gap 500 

