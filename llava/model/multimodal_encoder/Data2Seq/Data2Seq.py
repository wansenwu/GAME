import sys
sys.path.append('/ai/test/code/LLaVA/llava/model/multimodal_encoder/Data2Seq')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import clip
import Hyper_Spectrum
import Text
import Time_Series
import Image
import Video
import Graph
from transformers.models.clip import CLIPTokenizer
from Text import zero_padding
from transformers import CLIPImageProcessor


class Data2Seq(nn.Module):
    def __init__(self, modality, dim):
        super().__init__()
        self.modality = modality
        self.preprocessor = CLIPImageProcessor.from_pretrained('/ai/test/pretrained_weights/clip_weights/clip-vit-large-patch14-336')
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.embed_dim = dim
        if self.modality == 'image' or self.modality == 'infrared' or self.modality == 'x-ray':
            self.embed = Image.PatchEmbed(embed_dim=self.embed_dim)
        elif self.modality == 'text':
            self.embed = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        elif self.modality == 'video':
            self.embed = Video.PatchEmbed(embed_dim=self.embed_dim)
        elif self.modality == 'graph':
            self.embed = Graph.GraphFeatureTokenizer(rand_node_id_dim = self.embed_dim, orf_node_id_dim = self.embed_dim)
        elif self.modality == 'hyper':
            self.embed =  Hyper_Spectrum.PatchEmbed(embed_dim=self.embed_dim)
        elif self.modality == 'time-series' or self.modality == 'imu':
            self.embed =  Time_Series.DataEmbedding(cin=1, d_model=self.embed_dim)

        self.embed.to(dtype=torch.bfloat16)

    def get_audio_embeddings(audio):
        waveform1, sr = torchaudio.load(audio)

        waveform1 = waveform1 - waveform1.mean()

        audio_embedding = torchaudio.compliance.kaldi.fbank(waveform1, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        return audio_embedding

    def preprocess(self, image):
        self.preprocessor.crop_size = {'height': 224, 'width': 224}
        preprocess_image = self.preprocessor(image, return_tensors='pt')['pixel_values'][0]
        return preprocess_image.unsqueeze(0)

    def forward(self, data):
        if self.modality in ['image', 'infrared', 'x-ray', 'video', 'graph', 'hyper', 'time-series', 'imu','text' ]:
            embeddings = self.embed(data)
        elif self.modality =='text':
            embeddings = self.embed(data)
            embeddings = zero_padding(text_tensor=embeddings, tar_dim = self.embed_dim)
        elif self.modality =='audio':
            embeddings = self.get_audio_embeddings(data)
        return embeddings