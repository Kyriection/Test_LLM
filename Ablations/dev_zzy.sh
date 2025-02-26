for topk in 0.05
do 
torchrun --standalone --nproc_per_node 1 main_pretrain.py \
    --model_config configs/llama_60m.json \
    --eval_every 1000 \
    --save_every 4000 \
    --dtype bfloat16 \
    --batch_size 32 \
    --total_batch_size 512 \
    --lr 0.001 \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --optimizer stablespam1 \
    --weight_quant \
    --simulation \
    --weight_bits 4 \
    --weight_decay 0 \
    --project amazing_LLM_tokenwisenorm \
    --name 130m_top0.05_100prj_amazing_LLM_tokenwisenorm_256 \
    --save_dir /scratch-shared/HTJ/60m_top0.05_amazing_LLM_tokenwisenorm_256  \
    --restore_optimizer \
    --gamma1 0.7 \
    --gamma2 0.9 \
    --theta 0.999 \
    --update_proj_gap 500 \
    --total_T 10000 \
    --eta 0.5 \
    --topk $topk \
    --fp4 \
    --max_length 256 \
    --weight_group_size 1024
done



