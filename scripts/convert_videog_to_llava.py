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


if __name__ == '__main__':
    # load refcoco data
    root_dir = '/ai/test/data/VideoG/'

    datasets = ['TACoS', 'Charades', 'ActivityNet']
    # datasets = ['TACoS']

    ft_data = []
    for dataset in datasets:

        imgsett_file = 'metadata/train.jsonl'
        split_path = osp.join(root_dir, dataset, imgsett_file)

        all_data = []
        with open(split_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                all_data.append(data)

        for item in tqdm(all_data):
            query = item['query']
            label = [round(x, 2) for x in item['relevant_windows'][0]]
            vid = item['vid']
            duration = round(item['duration'], 2)

            ft_data.append({
                "id": vid,
                "video": [root_dir+dataset+'/vid_clip/'+vid+'.npz', root_dir+dataset+'/vid_slowfast/'+vid+'.npz'],
                "conversations": [
                    {'from': 'human',
                     'value': f"<image>\nGiven the video and a natural language query <{query}>, please identify the video moment described by the query and provide the start and end timestamps within the {duration}."},
                    {'from': 'gpt', 'value': "{}".format(label)},
                ],
            })
        print("split " + dataset + " finished", f'Number of samples: {len(ft_data)}')

    print(f'Total number of samples: {len(ft_data)}')
    with open("video_act.json", "w") as f:
        json.dump(ft_data, f, indent=2)


