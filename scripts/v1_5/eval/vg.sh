#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_name="1219_video_image"

#datasets=("unc/unc_val"
#         "unc/unc_testA"
#         "unc/unc_testB"
#         "unc+/unc+_val"
#         "unc+/unc+_testA"
#         "unc+/unc+_testB"
#         "gref/gref_val"
#         "gref_umd/gref_umd_val"
#         "gref_umd/gref_umd_test"
#         "flickr/flickr_test"
#         "referit/referit_test")

datasets=("unc/unc_val")

for split in "${datasets[@]}"
do
  for IDX in $(seq 0 $((CHUNKS-1)));
    do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vg \
        --model-path /ai/test/code/LLaVA/checkpoints/${model_name}/llava-v1.5-7b \
        --answers-file ./playground/data/eval/vg/answers_${model_name}/$split/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --split $split \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    done
  wait
    output_file=./playground/data/eval/vg/answers_${model_name}/$split/merge.jsonl
    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/data/eval/vg/answers_${model_name}/$split/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
done
