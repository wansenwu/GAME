import json
import os
import re
import torch
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from tqdm import tqdm
import copy


def is_value_present(dict_list, value):
    for i, dictionary in enumerate(dict_list):
        if value in dictionary.values():
            return i, True
    return 0, False


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

    resize_bbox = [round(b * ratio / 336, 2) for b in bbox]
    return resize_bbox


def demo_box(image, output, dataset):
    plt.imshow(image)
    axe = plt.gca()
    rect = matplotlib.patches.Rectangle((output[0], output[1]), output[2], output[3], edgecolor='r', linewidth=2.0,
                                        facecolor='none')
    axe.add_patch(rect)
    plt.savefig("{}.jpg".format(dataset))
    plt.clf()


if __name__ == '__main__':
    # load refcoco data
    root_dir = '/ai/test/data/'
    coco_path = '/ai/test/data/data/'

    # Use the REG for data augmentation
    REG = True

    datasets = ['unc', 'unc+', 'gref', 'gref_umd', 'flickr', 'referit']

    ft_data = []
    for dataset in datasets:
        if dataset == 'flickr':
            data_dir = 'Flickr30k/flickr30k_images'
        elif dataset == 'referit':
            data_dir = 'referit/images'
        else:
            data_dir = 'other/images/mscoco/images/train2014'

        imgsett_file = '{0}/{1}_train.pth'.format(dataset, dataset)
        split_path = osp.join(coco_path, imgsett_file)
        images = torch.load(split_path)
        for item in tqdm(images):
            if dataset == 'flickr':
                id, bbox, query = copy.deepcopy(item)
            else:
                id, _, bbox, query, _ = copy.deepcopy(item)

            if not (dataset == 'flickr' or dataset == 'referit'):
                bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
                # bbox[2], bbox[3] = bbox[2]-bbox[0], bbox[3]-bbox[1]

            image = Image.open(osp.join(root_dir, data_dir, id)).convert('RGB')

            # demo_box(image, bbox, dataset)

            final_bbox = bbox_converter(image, bbox)

            index, exist = is_value_present(ft_data, id)
            if not exist:
                ft_data.append({
                    "id": id,
                    "image": data_dir+"/"+id,
                    "conversations": [
                        {'from': 'human',
                         'value': f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes: {query}"},
                        {'from': 'gpt', 'value': "{}".format(final_bbox)},
                    ],
                })
                if REG:
                    ft_data[index]['conversations'].append(
                        {'from': 'human',
                         'value': f"Please describe the region within the given bounding box in the image: {final_bbox}"}
                    )

                    ft_data[index]['conversations'].append(
                        {'from': 'gpt', 'value': "{}".format(query)}
                    )
            else:
                ft_data[index]['conversations'].append(
                    {'from': 'human',
                     'value': f"Please provide the bounding box coordinate of the region this sentence describes: {query}"}
                )

                ft_data[index]['conversations'].append(
                    {'from': 'gpt', 'value': "{}".format(final_bbox)}
                )
                if REG:
                    ft_data[index]['conversations'].append(
                        {'from': 'human',
                         'value': f"Please describe the region within the given bounding box in the image: {final_bbox}"}
                    )

                    ft_data[index]['conversations'].append(
                        {'from': 'gpt', 'value': "{}".format(query)}
                    )
        print("split " + dataset + " finished", f'Number of samples: {len(ft_data)}')

    print(f'Number of samples: {len(ft_data)}')
    with open("vg_ft_reg.json", "w") as f:
        json.dump(ft_data, f, indent=2)


