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


def bbox_converter(pil_img, bbox):

    width, height = pil_img.size
    ratio = float(336/float(max(width, height)))

    # no pad, only resize
    if width == height:
        result = pil_img
    # add y
    elif width > height:
        bbox[1] = bbox[1] + (width - height) // 2
        bbox[3] = bbox[3] + (width - height) // 2
    # add x
    else:
        bbox[0] = bbox[0] + (height - width) // 2
        bbox[2] = bbox[2] + (height - width) // 2

    resize_bbox = [round(b * ratio / 336, 3) for b in bbox]
    return resize_bbox


def xywh2xyxy(bbox):
    x, y, w, h = bbox.unbind(-1)
    b = [x, y, x+w, y+h]
    return torch.stack(b, dim=-1)


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def vg_acc(pred_bbox, gt_bbox):
    # pred_bbox = xywh2xyxy(pred_bbox)
    # gt_boxes = xywh2xyxy(gt_bbox)
    iou = bbox_iou(pred_bbox, gt_bbox)
    return iou


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


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img.resize((336, 336))
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result.resize((336, 336))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result.resize((336, 336))


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    dataset = '15'
    # load eval split
    split_dir = '/ai/test/data/MSA/msa_dataset_all/tw{}/test_data.json'.format(dataset)
    image_path = '/ai/test/data/MSA/IJCAI2019_data/twitter20{}_images/'.format(dataset)

    # for different splits
    split = args.splits
    source_data = json.load(open(split_dir))['annotations']

    print('evaluation for ' + dataset)

    source_data = get_chunk(source_data, args.num_chunks, args.chunk_idx)
    result_list = []
    # into splits
    acc = 0
    for item in tqdm(source_data):

        image_file = item['image_id'] + '.jpg'
        answer = item['answer']
        qs = '(This is very important to my career!) Given the image and sentence {}, what is the sentiment expressed?'.format(item['sentence'])
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

        image = Image.open(os.path.join(image_path, image_file)).convert('RGB')

        image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
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
        print(outputs, answer)
        if outputs == answer:
            acc = acc + 1
        result_list.append(outputs)

    with open("results_{}_new.json".format(dataset), "w") as f:
        json.dump(result_list, f, indent=2)
    accu = acc / len(result_list)
    print('Accuracy of split ' + dataset, accu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/ai/test/code/LLaVA/checkpoints/116_msa_tw15/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/ai/test/data/MSA/IJCAI2019_data/twitter2015_images/")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--splits", type=str, default='unc/unc_testA.pth')
    args = parser.parse_args()

    eval_model(args)
