#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=60m_BF16_adam_momset1k_eta0.5_adaGN_adaCLIP
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=./logs_1b/60m_BF16_adam_momset1k_eta0.5_adaGN_adaCLIP.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
cd ../
wandb login --relogin 0831d62c2353a87b23a963dcb28ecc86cef378ee

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
    --optimizer stablespam1 \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-ablations \
    --name 60m-BF16_adam_momset1k_eta0.5_adaGN_adaClip \
    --save_dir /scratch-shared/HTJ/1b-model-stablespam-fp4 \
    --restore_optimizer \
    --gamma1 0.7 \
    --gamma2 0.9 \
    --theta 0.999 \
    --update_proj_gap 1000 \
    --total_T 10000 \
    --eta 0.5 



