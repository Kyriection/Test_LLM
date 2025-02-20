#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=stablespam_130_fp4_0.9_0.7_0.999_1e-3_500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --output=./logs_1b/stablespam_130_fp4_0.9_0.7_0.999_1e-3_500.out

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
    --lr 0.0008 \
    --warmup_steps 2000 \
    --num_training_steps 20000 \
    --optimizer stablespam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project stablespam \
    --name stablespam_130_fp4_0.9_0.7_0.999_1e-3_500 \
    --save_dir /scratch-shared/HTJ/llama_130m_adamw_qweight_simulate \
    --restore_optimizer \
    --gamma1 0.7 \
    --gamma2 0.9 \
    --theta 0.999 \
    --update_proj_gap 500 \
    --fp4