import torch
import torch.nn as nn

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import sys
sys.path.append('/ai/test/code/LLaVA/llava/model/multimodal_encoder')
from Data2Seq import Data2Seq
from timm.models.vision_transformer import Block
import torch.nn.init as init


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            print(images[0].shape, 'list')
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            print(images.shape, 'not list')
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class MetaTransformer(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.load_model()

    def load_model(self):
        self.image_processor = Data2Seq.Data2Seq(modality='image', dim=768)

        ckpt = torch.load("/ai/test/pretrained_weights/meta-transformer/Meta-Transformer_base_patch16_encoder")

        self.vision_tower = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
        self.vision_tower.load_state_dict(ckpt, strict=True)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):

        adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=50)

        # 将数据转换为需要的形状 (batch_size, hidden_dim, sequence_length)
        data = images.unsqueeze(0).transpose(1, 2)

        # 对数据进行平均池化，将 sequence_length 统一到 50
        pooled_data = adaptive_avg_pool(data)

        # 再次转换为原始形状 (batch_size, sequence_length, hidden_dim)
        pooled_data = pooled_data.transpose(1, 2)

        image_features = self.linear(pooled_data)
        if type(images) is list:
            image_features = []
            for image in images:
                image = image.to(device=self.device, dtype=self.dtype)
                image = self.image_processor(image)
                image_feature = self.vision_tower(image)
                image_features.append(image_feature)
        else:
            images = images.to(device=self.device, dtype=self.dtype)
            images = self.image_processor(images)
            image_features = self.vision_tower(images)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def processor_device(self):
        return next(self.vision_tower.image_processor.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 768

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class VideoProjector(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.vision_tower_name = vision_tower

        self.is_loaded = False

        self.load_model()

    def load_model(self):
        self.image_processor = 0
        modules = [nn.Linear(2816, 5120)]
        modules.append(nn.GELU())
        modules.append(nn.Linear(5120, 5120))
        modules.append(nn.GELU())
        modules.append(nn.Linear(5120, 1024))
        self.proj_layer = nn.Sequential(*modules)
        self.proj_layer.requires_grad_(True)

        self.is_loaded = True

    def forward(self, images):

        if type(images) is list:
            image_features = []
            for image in images:
                image = image.to(device=self.device, dtype=self.dtype)
                # Project to the hidden dim
                output = self.proj_layer(image.squeeze(1))

                image_features.append(output)
        else:
            images = images.to(device=self.device, dtype=self.dtype)
            image_features = self.proj_layer(images.squeeze(1))
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.proj_layer[0].weight.dtype

    @property
    def device(self):
        return self.proj_layer[0].weight.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


import pdb
import os
local_rank = int(os.environ["LOCAL_RANK"])


class UniTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.vision_tower_name = vision_tower

        self.is_loaded = False

        # for image
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.load_model()

    def load_model(self):

        # load image processor and model
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.image_tower.requires_grad_(False)

        # load video processor and model
        # modules = [nn.Linear(2816, 5120)]
        # modules.append(nn.ReLU())
        # modules.append(nn.Linear(5120, 5120))
        # modules.append(nn.ReLU())
        # modules.append(nn.Linear(5120, 1024))
        # self.video_tower = nn.Sequential(*modules)
        # init.kaiming_uniform_(self.video_tower[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        # init.kaiming_uniform_(self.video_tower[2].weight, a=0, mode='fan_in', nonlinearity='relu')
        # init.kaiming_uniform_(self.video_tower[4].weight, a=0, mode='fan_in', nonlinearity='relu')
        # self.video_tower = nn.Linear(2816, 1024)
        # self.video_tower.requires_grad_(True)
        self.video_tower = nn.Identity()

        # load pointcloud processor and model

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, data, mode):
        if mode == 'image':
            images = data
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                          output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype),
                                                       output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

            return image_features

        elif mode == 'video':
            print('encoding video now', local_rank)
            videos = data
            if type(videos) is list:
                video_features = []
                for video in videos:
                    video = video.to(device=self.video_device, dtype=self.video_dtype)
                    # Project to the hidden dim
                    output = self.video_tower(video)

                    video_features.append(output)
            else:
                videos = videos.to(device=self.device, dtype=self.dtype)

                #
                # if local_rank == 0:
                #     pdb.set_trace()
                video_features = self.video_tower(videos)

            print('encoding video finished', local_rank)
            return video_features

        elif mode == 'ptcd':
            return 0

    @property
    def dtype(self):
        return self.image_tower.dtype

    @property
    def device(self):
        return self.image_tower.device

    @property
    def video_dtype(self):
        return self.video_tower[0].weight.dtype

    @property
    def video_device(self):
        return self.video_tower[0].weight.device

    @property
    def hidden_size(self):
        return 1024

