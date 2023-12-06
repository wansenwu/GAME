import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import copy
import numpy as np
import torch.nn as nn

import pdb


def compute_temporal_iou_batch_paired(pred_windows, gt_windows):
    """ compute intersection-over-union along temporal axis for each pair of windows in pred_windows and gt_windows.
    Args:
        pred_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
    Returns:
        iou (float): np.ndarray, (N, )

    References:
        for np.divide with zeros, see https://stackoverflow.com/a/37977222
    """
    intersection = np.maximum(
        0, np.minimum(pred_windows[:, 1], gt_windows[:, 1]) - np.maximum(pred_windows[:, 0], gt_windows[:, 0])
    )
    union = np.maximum(pred_windows[:, 1], gt_windows[:, 1]) \
            - np.minimum(pred_windows[:, 0], gt_windows[:, 0])  # not the correct union though
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def video_evaluate(pred_windows, gt_windows):
    iou_thds = [0.3, 0.5, 0.7, 0.9]

    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    miou_at_one = float(f"{np.mean(pred_gt_iou) * 100:.2f}")
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    return miou_at_one, iou_thd2recall_at_one


def count_values_above_threshold(lst, threshold):
    count = 0
    for value in lst:
        if value[0] > threshold:
            count += 1
    return count


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    video_prompt = 'Given the video and a natural language query <{query}>, please identify the video moment described by the query and provide the start and end timestamps within the {duration}.'
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    split = args.splits
    root_dir = '/ai/test/data/VideoG/{}/'.format(split)
    # load eval split
    split_dir = '/ai/test/data/VideoG/{}/metadata/test.jsonl'.format(split)

    source_data = []
    with open(split_dir, 'r') as file:
        for line in file:
            data = json.loads(line)
            source_data.append(data)

    source_data = get_chunk(source_data, args.num_chunks, args.chunk_idx)
    result_list = []

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    # into splits
    pred_windows = []
    gt_windows = []
    adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=50)
    for item in tqdm(source_data):
        query = item['query']
        label = [round(x, 2) for x in item['relevant_windows'][0]]
        vid = item['vid']
        duration = round(item['duration'], 2)

        qs = video_prompt.replace('{query}', query).replace('{duration}', str(duration))

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()

        # get video feature
        sf_feature = np.load(root_dir+'vid_slowfast/'+vid+'.npz')
        clip_feature = np.load(root_dir+'vid_clip/'+vid+'.npz')
        # cat and normalize
        v_feat_list = [sf_feature['features'].astype(np.float32), clip_feature['features'].astype(np.float32)]
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)

        def l2_normalize_np_array(np_array, eps=1e-5):
            """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
            return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

        v_feat = l2_normalize_np_array(v_feat)

        video_feature = torch.from_numpy(v_feat).unsqueeze(0).transpose(1, 2)
        # pooling for equal length
        pool_feature = adaptive_avg_pool(video_feature)
        # transpose back
        image_tensor = pool_feature.transpose(1, 2)
        image_tensor = image_tensor.half().cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        def check_nan_parameters(model):
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f'Parameter {name} has NaN values at index {torch.where(torch.isnan(param))}.')

        # 假设已经定义了一个神经网络模型model
        check_nan_parameters(model)

        model.model.vision_tower.proj_layer.requires_grad_(False)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        filted_outputs = list(map(float, outputs.replace("[", "").replace("]", "").split(',')))
        # print(outputs, filted_outputs)
        gt_windows.append(label)
        pred_windows.append(filted_outputs)
        ans_file.write(json.dumps({"question_id": vid,
                                   "prompt": cur_prompt,
                                   "text": filted_outputs,
                                   "gt_box": label,
                                   "split": split,
                                   "metadata": {}}) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/ai/test/code/LLaVA/checkpoints/1128_video/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/ai/test/data/other/images/mscoco/images/train2014")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="./playground/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--splits", type=str, default='TACoS')
    args = parser.parse_args()

    eval_model(args)
