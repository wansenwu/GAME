#!/bin/bash

# --data_path /ai/test/code/LLaVA/vg_ft.json
# --image_folder /ai/test/data/other/images/mscoco/images/train2014
# video_act.json
# vg_ft_cfr_new
# ivg
# vicuna-13b-v1.5
#    --lora_enable True \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /ai/test/pretrained_weights/vicuna-7b \
    --version v1 \
    --data_path /ai/test/code/LLaVA/scripts/vg_ft_cfr_new.json \
    --image_folder /ai/test/data/ \
    --vision_tower /ai/test/pretrained_weights/clip_weights/clip-vit-large-patch14-336 \
    --vision_tower_name clip \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --modality_sampler False \
    --bf16 True \
    --output_dir ./checkpoints/1220_image/llava-v1.5-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --pretrain_mm_mlp_adapter /ai/test/pretrained_weights/llava/mm_projector_7b.bin \
