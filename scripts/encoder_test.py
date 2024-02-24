# import torch
#
# from llava.model.multimodal_encoder.clip_encoder import UniTower
#
# import argparse
# import numpy as np
#
#
# parser = argparse.ArgumentParser(description='PyTorch Training Example')
#
# parser.add_argument('--mm_vision_select_layer', default=-2)
#
# args = parser.parse_args()
#
# Net = UniTower('/ai/test/pretrained_weights/clip_weights/clip-vit-large-patch14-336', args)
# Net.cuda()
#
# # input devcie dtype
# images = [torch.zeros([3, 336, 336]),
#          torch.zeros([3, 336, 336]),
#          torch.zeros([3, 336, 336]),
#          torch.zeros([1, 576, 2816])]
#
# test_tensor = torch.zeros([8, 1, 576, 5120])
#
# # 定义比较函数，按照维度大小进行比较
# def compare_dims(elem):
#     if elem.shape == (1, 576, 2816):
#         return 0
#     elif elem.shape == (3, 336, 336):
#         return 1
#     else:
#         return 2
#
# # 按照维度排序
# images.sort(key=compare_dims)
#
# indices = list(range(len(images)))
# processed_data = [None] * len(indices)
#
# for index in indices:
#     d = images[index].cuda()
#     if d.shape == (3, 336, 336):
#         processed_data[index] = Net(d.unsqueeze(0), mode='image')
#         print('\n encoded image feature', processed_data[index].shape)
#     elif d.shape == (1, 576, 2816):
#         processed_data[index] = Net(d, mode='video')
#         print('\n encoded video feature', processed_data[index].shape)
#     elif d.shape == (1, 20, 400):
#         processed_data[index] = Net.get_vision_tower()(d, mode='ptcd')
#     else:
#         raise ValueError("Invalid data shape:", d.shape)
#
# image_features = torch.stack(processed_data).squeeze(1)
# # image_features = self.get_model().mm_projector(image_features)
# print(image_features.shape)

"""Test script for multimodal sampler"""
# import random
# from torch.utils.data import Sampler
#
#
# class MultiModalSampler(Sampler):
#     def __init__(self, data_source, modalities, batch_size):
#         self.data_source = data_source
#         self.modalities = modalities
#         self.batch_size = batch_size
#         self.grouped_indices = self._group_indices()
#
#     def _group_indices(self):
#         grouped_indices = {modality: [] for modality in self.modalities}
#         for idx, sample in enumerate(self.data_source):
#             modality = sample['modality']  # 假设不同模态的数据包含在样本的'modality'字段中
#             grouped_indices[modality].append(idx)
#         return grouped_indices
#
#     def __iter__(self):
#         while True:
#             batch = []
#             for modality in self.modalities:
#                 indices = self.grouped_indices[modality]
#                 if len(indices) < self.batch_size:
#                     indices += random.choices(indices, k=self.batch_size - len(indices))
#                 batch.extend(random.sample(indices, self.batch_size))
#             yield batch
#
#     def __len__(self):
#         return len(self.grouped_indices[self.modalities[0]]) // self.batch_size  # 假设所有模态的数据量相同，取一个模态的数据量即可
#
# # 使用示例
# data_source = [
#     {'modality': 'modality1', 'data': 'Sample 1'},
#     {'modality': 'modality2', 'data': 'Sample 2'},
#     {'modality': 'modality1', 'data': 'Sample 3'},
#     {'modality': 'modality3', 'data': 'Sample 4'},
#     {'modality': 'modality2', 'data': 'Sample 5'},
#     {'modality': 'modality3', 'data': 'Sample 6'},
# ]
#
# modalities = ['modality1', 'modality2', 'modality3']
# batch_size = 1
#
# sampler = MultiModalSampler(data_source, modalities, batch_size)
# data_loader = iter(sampler)
#
# for _ in range(5):
#     batch_indices = next(data_loader)
#     batch_samples = [data_source[i] for i in batch_indices]
#     print("Batch:")
#     for sample in batch_samples:
#         print(f"Modality: {sample['modality']}, Data: {sample['data']}")
#     print()

import random
from torch.utils.data import Sampler


class ImageVideoSampler(Sampler):
    def __init__(self, data_source):
        self.image_indices = [i for i, sample in enumerate(data_source) if sample['modality'] == 'image']
        self.video_indices = [i for i, sample in enumerate(data_source) if sample['modality'] == 'video']
        self.num_samples = min(len(self.image_indices), len(self.video_indices))

    def __iter__(self):
        if len(self.image_indices) > len(self.video_indices):
            image_subset = random.sample(self.image_indices, self.num_samples)
            video_subset = self.video_indices * (self.num_samples // len(self.video_indices)) + random.sample(
                self.video_indices, self.num_samples % len(self.video_indices))
        else:
            video_subset = random.sample(self.video_indices, self.num_samples)
            image_subset = self.image_indices * (self.num_samples // len(self.image_indices)) + random.sample(
                self.image_indices, self.num_samples % len(self.image_indices))

        samples = []
        for i in range(self.num_samples):
            samples.append(image_subset[i])
            samples.append(video_subset[i])
        return iter(samples)

    def __len__(self):
        return 2 * self.num_samples


# 示例数据
data = [{'modality': 'image', 'data': 'image_data_1'},
        {'modality': 'image', 'data': 'image_data_2'},
        {'modality': 'video', 'data': 'video_data_1'},
        {'modality': 'video', 'data': 'video_data_2'},
        {'modality': 'image', 'data': 'image_data_3'},
        {'modality': 'video', 'data': 'video_data_3'},
        {'modality': 'image', 'data': 'image_data_4'}]

# 创建数据集和数据加载器
from torch.utils.data import DataLoader

dataset = data
sampler = ImageVideoSampler(dataset)
loader = DataLoader(dataset, batch_size=2, sampler=sampler)

# 打印每个小批量
for i, batch in enumerate(loader):
    print(f"batch {i}: {batch}")

'''Modify the dataset for sampler'''
# import torch
# import json
#
# file1 = '/ai/test/code/LLaVA/scripts/ivg_new.json'
# with open(file1, 'r', encoding='utf-8') as f:
#     data1 = json.load(f)
#
# for item in data1:
#     if 'video' in item:
#         item['modality'] = 'video'
#     elif 'image' in item:
#         item['modality'] = 'image'
#
# output_file = '/ai/test/code/LLaVA/scripts/ivg_new.json'
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(data1, f, ensure_ascii=False, indent=4)  # 将合并后的数据写入输出文件

