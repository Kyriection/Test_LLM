#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=lion_130_fp42
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --output=./logs_1b/lion_130_fp42.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch

cd ..

torchrun --standalone --nproc_per_node 2 main_pretrain.py \
    --model_config configs/llama_130m.json \
    --eval_every 1000 \
    --save_every 100000 \
    --dtype bfloat16 \
    --batch_size 256 \
    --total_batch_size 512 \
    --lr 0.0002 \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --optimizer lion \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project galore-x \
    --name 130m-lion-2e-4 \
    --save_dir /scratch-shared/HTJ/llama_130m_adamw_qweight_simulate \
    --restore_optimizer \
    --fp4
# ## vqa
#python mgm/eval_rs/batch_geochat_vqa.py \
#    --model-path $MODEL_PATH \
#    --model-base ../model_zoo/vicuna-7b-v1.5/ \
#    --question-file ../VRSBench/VRSBench_EVAL_vqa.json \
#    --image-folder $IMAGE_PATH \
#    --answers-file $OUTPUT/vrsbench_vqa_v2.jsonl
