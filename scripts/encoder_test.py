import torch

from llava.model.multimodal_encoder.clip_encoder import UniTower

import argparse
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Training Example')

parser.add_argument('--mm_vision_select_layer', default=-2)

args = parser.parse_args()

Net = UniTower('/ai/test/pretrained_weights/clip_weights/clip-vit-large-patch14-336', args)
Net.cuda()

# input devcie dtype
images = [torch.zeros([3, 336, 336]),
         torch.zeros([3, 336, 336]),
         torch.zeros([3, 336, 336]),
         torch.zeros([1, 576, 2816])]


# 定义比较函数，按照维度大小进行比较
def compare_dims(elem):
    if elem.shape == (1, 576, 2816):
        return 0
    elif elem.shape == (3, 336, 336):
        return 1
    else:
        return 2

# 按照维度排序
images.sort(key=compare_dims)

# 输出排序后的结果
print(images)

indices = list(range(len(images)))
processed_data = [None] * len(indices)

for index in indices:
    d = images[index].cuda()
    if d.shape == (3, 336, 336):
        processed_data[index] = Net(d.unsqueeze(0), mode='image')
        print('\n encoded image feature', processed_data[index].shape)
    elif d.shape == (1, 576, 2816):
        processed_data[index] = Net(d, mode='video')
        print('\n encoded video feature', processed_data[index].shape)
    elif d.shape == (1, 20, 400):
        processed_data[index] = Net.get_vision_tower()(d, mode='ptcd')
    else:
        raise ValueError("Invalid data shape:", d.shape)

image_features = torch.stack(processed_data).squeeze(1)
# image_features = self.get_model().mm_projector(image_features)
print(image_features.shape)
