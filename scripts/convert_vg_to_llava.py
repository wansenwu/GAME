import json
import os
import re
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

clip_path = '/ai/test/pretrained_weights/clip_weights/clip-vit-large-patch14-336'
# load llava finetune data
data_path = '/ai/test/pretrained_weights/vicuna-13b-v1.5/llava_v1_5_mix665k.json'
split_indices = json.load(open(data_path))

with open("llava_subset.json", "w") as f:
    json.dump(split_indices[:100000], f, indent=2)


data_path2 = '/ai/test/code/LLaVA/vg_ft.json'
split_indices2 = json.load(open(data_path2))

# load refcoco data
root_dir = '/ai/test/data/other/images/mscoco/images/train2014'
coco_path = '/ai/test/data/data/unc/unc_train.pth'
images = torch.load(coco_path)


def expand2square(pil_img, background_color):
    width, height = pil_img.size

    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        pad = (0, (width - height) // 2)
        result.paste(pil_img, pad)

    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        pad = ((height - width) // 2, 0)
        result.paste(pil_img, pad)

    ratio = float(336/float(max(width, height)))
    return result.resize((336, 336)), [k*ratio for k in pad], ratio


def load_clip():
    image_processor = CLIPImageProcessor.from_pretrained(clip_path)
    vision_tower = CLIPVisionModel.from_pretrained(clip_path)
    return vision_tower, image_processor


ft_data = []
for item in images:
    image = item[0]
    input = item[3]
    output = item[2]

    model, processor = load_clip()
    image = Image.open(os.path.join(root_dir, image)).convert('RGB')

    plt.imshow(image)
    axe = plt.gca()
    rect = matplotlib.patches.Rectangle((output[0], output[1]), output[2], output[3], edgecolor='r',linewidth=2.0, facecolor='none')
    axe.add_patch(rect)
    plt.savefig("before.jpg")

    image2, pad, ratio = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    resize_output = [item * ratio for item in output]
    pro_image = processor.preprocess(image2, return_tensors='pt')['pixel_values'][0]

    image3 = Image.fromarray(pro_image.permute(1, 2, 0).byte().numpy())

    # 显示图像
    plt.imshow(image3)
    plt.savefig("tensor.jpg")

    width, height = image.size

    axe = plt.gca()

    # rect = matplotlib.patches.Rectangle((output[0], output[1]), output[2], output[3], edgecolor='r',linewidth=2.0, facecolor='none')
    rect2 = matplotlib.patches.Rectangle((resize_output[0] + pad[0], resize_output[1] + pad[1]), resize_output[2], resize_output[3], edgecolor='g', linewidth=2.0, facecolor='none')

    # axe.add_patch(rect)
    axe.add_patch(rect2)
    plt.savefig("after.jpg")

    ft_data.append({
                "id": image,
                "image": image,
                "conversations": [
                    {'from': 'human', 'value': f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes: {input}"},
                    {'from': 'gpt', 'value': f"{resize_output}"},
                ],
            })
print('kk')
