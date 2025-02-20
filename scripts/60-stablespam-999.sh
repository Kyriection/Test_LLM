export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1
torchrun --standalone --nproc_per_node 1 main_pretrain.py \
    --model_config configs/llama_60m.json \
    --eval_every 1000 \
    --save_every 100000 \
    --dtype bfloat16 \
    --batch_size 256 \
    --total_batch_size 512 \
    --lr 0.001 \
    --warmup_steps 2000 \
    --num_training_steps 20000 \
    --optimizer stablespam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 8 \
    --weight_decay 0 \
    --project galore-x \
    --name test-q-stable-spam \
    --save_dir llama_130m_adamw_qweight_simulate \
    --restore_optimizer \
    --act_quant \
    --act_group_size 64 \
    --act_topk 2 \
    --act_stochastic 
