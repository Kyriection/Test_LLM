#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=130m_top0.05_40_100prj_fp3Xonly_orignal_group_all_1e-3_20kT_groupfix1024_maxlen_512_stable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
#SBATCH --output=./logs_1b/130m_top0.05_40_100prj_fp3Xonly_original_groupall_1e-3_20kT_groupfix1024_maxlen512_stable.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
cd ../
wandb login --relogin 0831d62c2353a87b23a963dcb28ecc86cef378ee


for topk in 0.05
do 
torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_130m.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 8 \
    --total_batch_size 512 \
    --lr 0.001 \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --optimizer stablespam1 \
    --weight_quant \
    --simulation \
    --weight_bits 4 \
    --weight_decay 0 \
    --project amazing_LLM \
    --name 130m_top0.05_100prj_fp3Xonly_original_groupall_20kT_1e-3_groupsizefix1024_maxlen512_stable \
    --save_dir /scratch-shared/HTJ/1300m_groupsizeada_20K_1e-3_top0.05_maxlen512_fp3_groupsizefix512_stable  \
    --restore_optimizer \
    --gamma1 0.7 \
    --gamma2 0.9 \
    --theta 0.999 \
    --update_proj_gap 500 \
    --total_T 10000 \
    --eta 0.5 \
    --topk $topk \
    --fp4 \
    --max_length 512 \
    --weight_group_size 1024
done



