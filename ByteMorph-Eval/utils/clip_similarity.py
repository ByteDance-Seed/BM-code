# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AltCLIPModel, AltCLIPProcessor, AutoProcessor
from einops import rearrange
from scipy import spatial

class ClipSimilarity_new(nn.Module):
    def __init__(self, name: str = "ViT-B/32"):
        super().__init__()
        print(f"Using CLIP: {name}")
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)
        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.encode_text, self.encode_image = self.encode_text_clip, self.encode_image_clip
        self.model.eval().requires_grad_(False)
        self.device = next(self.parameters()).device
        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text_clip(self, text: list[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

 
    def encode_image_clip(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(self, 
                image_src: torch.Tensor,
                image_tgt: torch.Tensor,
                image_gen: torch.Tensor,
                return_cross_scores: bool = True,
                return_dict: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        image_features_src = self.encode_image(image_src).detach().cpu().float()
        image_features_tgt = self.encode_image(image_tgt).detach().cpu().float() 
        image_features_gen = self.encode_image(image_gen).detach().cpu().float()
        sim_direction_new = 1 - spatial.distance.cosine((image_features_tgt - image_features_src).view(image_features_src.shape[1]),
                                                    (image_features_gen - image_features_src).view(image_features_src.shape[1]))
        
        
        
        out_dict = {
            "clip_dir_img": sim_direction_new.item(),
        }
        if return_cross_scores:
            sim_direction_new_unnorm = torch.mm(image_features_tgt - image_features_src, (image_features_gen - image_features_src).T)
            out_dict.update({
                    "clip_dir_img_unnorm": sim_direction_new_unnorm.item(),
            })
            if return_dict:
                return out_dict
            else:
                raise NotImplementedError
        else:
            if return_dict:
                return out_dict
            else:
                raise NotImplementedError


class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-B/32"):
        super().__init__()
        print(f"Using CLIP: {name}")
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)
        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.encode_text, self.encode_image = self.encode_text_clip, self.encode_image_clip

        self.model.eval().requires_grad_(False)
        self.device = next(self.parameters()).device
        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text_clip(self, text: list[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features


    def encode_image_clip(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(self, 
                image_0: torch.Tensor, 
                image_1: torch.Tensor, 
                text_0: list[str], 
                text_1: list[str], 
                return_cross_scores: bool = False,
                return_dict: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        image_features_0 = self.encode_image(image_0).detach().cpu().float()
        image_features_1 = self.encode_image(image_1).detach().cpu().float()
        text_features_0 = self.encode_text(text_0).detach().cpu().float()
        text_features_1 = self.encode_text(text_1).detach().cpu().float() 

        sim_1 = 1 - spatial.distance.cosine(image_features_1.view(image_features_1.shape[1]),
                                                    text_features_1.view(text_features_1.shape[1]))
        sim_image = 1 - spatial.distance.cosine(image_features_0.view(image_features_0.shape[1]),
                                                    image_features_1.view(image_features_1.shape[1]))
        sim_direction = 1 - spatial.distance.cosine((image_features_1 - image_features_0).view(image_features_1.shape[1]),
                                                    (text_features_1 - text_features_0).view(text_features_0.shape[1]))
        out_dict = {
            "clip_sim_txt": sim_1.item(),
            "clip_sim_img": sim_image.item(),
            "clip_dir_txt": sim_direction.item(),
        }



        if return_cross_scores:
            sim_0_cross = F.cosine_similarity(image_features_0, text_features_1)
            sim_1_cross = F.cosine_similarity(image_features_1, text_features_0)
            sim_direction_unnorm = torch.mm(image_features_1 - image_features_0, (text_features_1 - text_features_0).T)
            sim_text = F.cosine_similarity(text_features_0, text_features_1)
            out_dict.update({
                    "clip_sim_0_cross": sim_0_cross.item(),
                    "clip_sim_1_cross": sim_1_cross.item(),
                    "clip_sim_dir_unnorm": sim_direction_unnorm.item(),
                    "clip_sim_text": sim_text.item(),
            })
            if return_dict:
                return out_dict
            else:
                raise NotImplementedError
        else:
            if return_dict:
                return out_dict
            else:
                raise NotImplementedError