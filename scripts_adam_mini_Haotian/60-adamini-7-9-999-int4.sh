#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=adamini_our_testint4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00
#SBATCH --output=./logs_1b/adamini_our_testint4.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
wandb login --relogin 652c840880355982b883464461fb41868153b9a6

cd ..

torchrun --standalone --nproc_per_node 2 main_pretrain.py \
    --model_config configs/llama_60m.json \
    --eval_every 1000 \
    --save_every 100000 \
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
    --project galore-x \
    --name 60m-adamin-7-9-999-int4 \
    --save_dir /scratch-shared/HTJ/llama_130m_adamw_qweight_simulate \
    --restore_optimizer \
    --gamma1 0.7 \
    --gamma2 0.9 \
    --theta 0.999 \
    --act_quant \
    --act_group_size 64 \
    --act_topk 2 \
    --act_stochastic 
