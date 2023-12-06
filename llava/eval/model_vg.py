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
    size = 224   # 336 for clip
    if width == height:
        return pil_img.resize((size, size))
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result.resize((size, size))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result.resize((size, size))


def eval_model(args):
    # Model
    disable_torch_init()
    vg_prompt = 'Please provide the bounding box coordinate of the region this sentence describes: '
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # load eval split
    split_dir = '/ai/test/data/data/'
    datasets = ['unc/unc_val.pth',
                'unc/unc_testA.pth',
                'unc/unc_testB.pth',
                'unc+/unc+_val.pth',
                'unc+/unc+_testA.pth',
                'unc+/unc+_testB.pth',
                'gref/gref_val.pth',
                'gref_umd/gref_umd_val.pth',
                'gref_umd/gref_umd_test.pth',
                ]

    # for different splits
    split = args.splits
    source_data = torch.load(os.path.join(split_dir, split + '.pth'))
    dataset, result_name = split.split('/')
    if dataset == 'flickr':
        image_folder = "/ai/test/data/Flickr30k/flickr30k_images"
    elif dataset == 'referit':
        image_folder = "/ai/test/data/referit/images"
    else:
        image_folder = "/ai/test/data/other/images/mscoco/images/train2014"

    source_data = get_chunk(source_data, args.num_chunks, args.chunk_idx)
    result_list = []

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    # into splits
    for item in tqdm(source_data):
        if dataset == 'flickr':
            image_file, bbox, query = copy.deepcopy(item)
        else:
            image_file, _, bbox, query, _ = copy.deepcopy(item)

        if not (dataset == 'flickr' or dataset == 'referit'):
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]

        image_file = item[0]
        qs = vg_prompt + query
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

        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        gt_bbox = torch.tensor(bbox_converter(image, bbox))

        image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))

        # for clip encoder
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # for meta
        # image_tensor = image_processor.preprocess(image)

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
        # print(outputs, gt_bbox)
        outputs = torch.tensor(list(map(float, outputs.replace("[", "").replace("]", "").split(','))))

        iou = vg_acc(outputs.unsqueeze(0), gt_bbox.unsqueeze(0)).numpy().tolist()
        # print(iou)
        result_list.append(iou)
        ans_file.write(json.dumps({"question_id": image_file,
                                   "prompt": cur_prompt,
                                   "text": outputs.numpy().tolist(),
                                   "gt_box": gt_bbox.numpy().tolist(),
                                   "iou": iou[0],
                                   "split": split,
                                   "metadata": {}}) + "\n")
    ans_file.close()
    # with open("results_{}_new.json".format(result_name), "w") as f:
    #     json.dump(result_list, f, indent=2)
    # accu = count_values_above_threshold(result_list, 0.5)/len(result_list)
    # print('Accuracy of split ' + result_name, accu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/ai/test/code/LLaVA/checkpoints/1115_meta/llava-v1.5-13b")
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
    parser.add_argument("--splits", type=str, default='unc/unc_testA')
    args = parser.parse_args()

    eval_model(args)
