#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=stablespam_ab_momset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=./logs_1b/stablespam_ab_fp4_momset.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
cd ../
for ga in 250 500 1000 2500 5000
do
torchrun --standalone --nproc_per_node 1 main_pretrain.py \
    --model_config configs/llama_60m.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr 0.005 \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --optimizer stablespam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-ab  \
    --name stablespam-abs_momenset \
    --save_dir /scratch-shared/HTJ/1b-model-stablespam-fp4 \
    --restore_optimizer \
    --fp4 \
    --gamma1 0.85 \
    --gamma2 0.999 \
    --theta 0.999 \
    --update_proj_gap $ga \
    --total_T 10000 \
    --eta 0.5 
done


