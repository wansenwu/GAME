#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_name="1219_video_image"

datasets=("TACoS")

#datasets=("TACoS" "Charades")

#datasets=("TACoS" "ActivityNet" "Charades")

for split in "${datasets[@]}"
do
#  for IDX in $(seq 0 $((CHUNKS-1)));
#    do
#    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_video_g \
#        --model-path /ai/test/code/LLaVA/checkpoints/${model_name}/llava-v1.5-7b \
#        --answers-file ./playground/data/eval/video_g/answers_${model_name}/$split/${CHUNKS}_${IDX}.jsonl \
#        --num-chunks $CHUNKS \
#        --chunk-idx $IDX \
#        --split $split \
#        --temperature 0 \
#        --conv-mode vicuna_v1 &
#    done
#  wait
    output_file=./playground/data/eval/video_g/answers_${model_name}/$split/merge.jsonl
    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/data/eval/video_g/answers_${model_name}/$split/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
done



## Evaluate
#python scripts/convert_seed_for_submission.py \
#    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
#    --result-file $output_file \
#    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/llava-v1.5-13b.jsonl

