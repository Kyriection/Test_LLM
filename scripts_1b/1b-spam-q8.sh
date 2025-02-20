#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=spam_1b_q8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=./logs_1b/spam_1b_q8.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch
cd ../


torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_1b.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 64 \
    --total_batch_size 512 \
    --lr 0.0002 \
    --warmup_steps 2000 \
    --num_training_steps 20000 \
    --optimizer spam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 8 \
    --weight_decay 0 \
    --project 1bmodel \
    --name 1b-spam-q8 \
    --save_dir /scratch-shared/HTJ/1b_model-spam-int8 \
    --restore_optimizer \
    --act_quant \
    --act_group_size 64 \
    --act_topk 2 \
    --act_stochastic 

# ## vqa
#python mgm/eval_rs/batch_geochat_vqa.py \
#    --model-path $MODEL_PATH \
#    --model-base ../model_zoo/vicuna-7b-v1.5/ \
#    --question-file ../VRSBench/VRSBench_EVAL_vqa.json \
#    --image-folder $IMAGE_PATH \
#    --answers-file $OUTPUT/vrsbench_vqa_v2.jsonl
