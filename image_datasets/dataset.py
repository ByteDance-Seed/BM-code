# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 # Licensed under the FLUX.1 [dev] Non-Commercial License  
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at 
 #
 #  https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License. 

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import glob
import torch
import torchvision.transforms.functional as TF



def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        # self.images = glob.glob(img_dir +'**/*.jpg', recursive=True) + glob.glob(img_dir +'**/*.png', recursive=True) + glob.glob(img_dir +'**/*.jpeg', recursive=True)
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)
            img = image_resize(img, self.img_size)
            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            if self.caption_type == "json":
                prompt = json.load(open(json_path))['caption']
            else:
                prompt = open(json_path).read()
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)



class ImageEditPairDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False, grayscale_editing=False, zoom_camera=False):
        # self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images = glob.glob(img_dir +'**/*.jpg', recursive=True) + glob.glob(img_dir +'**/*.png', recursive=True) + glob.glob(img_dir +'**/*.jpeg', recursive=True)
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.grayscale_editing = grayscale_editing
        self.zoom_camera = zoom_camera 
        if "ByteMorph-Bench" or "InstructMove" in img_dir:
            self.eval = True
        else:
            self.eval = False
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            ori_width, ori_height = img.size
            left_half = (0, 0, ori_width // 2, ori_height)
            right_half = (ori_width // 2, 0, ori_width, ori_height)
            src_image = img.crop(left_half)  # Left half
            tgt_image = img.crop(right_half)  # Right half
            # print("ori_width, ori_height: ",ori_width, ori_height)
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    src_image = crop_to_aspect_ratio(src_image, ratio)
                    tgt_image = crop_to_aspect_ratio(tgt_image, ratio)
            src_image = image_resize(src_image, self.img_size)
            tgt_image = image_resize(tgt_image, self.img_size)
            w, h = src_image.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            # print("new_w, new_h: ",new_w, new_h)
            src_image = src_image.resize((new_w, new_h))
            src_image = torch.from_numpy((np.array(src_image) / 127.5) - 1)
            src_image = src_image.permute(2, 0, 1)
            tgt_image = tgt_image.resize((new_w, new_h))
            tgt_image = torch.from_numpy((np.array(tgt_image) / 127.5) - 1)
            tgt_image = tgt_image.permute(2, 0, 1)
            json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            if self.eval:
                image_name = self.images[idx].split('.')[0].split("/")[-1]
                edit_type = self.images[idx].split('.')[0].split("/")[-2]
            if self.caption_type == "json":
                if not self.eval:
                    prompt = None
                    edit_prompt = json.load(open(json_path))['edit']
                else:
                    prompt = None
                    edit_prompt = json.load(open(json_path))['edit']
            else:
                raise NotImplementedError
                # prompt = open(json_path).read()
            if (not self.grayscale_editing) and (not self.zoom_camera):
                if not self.eval:
                    return src_image, tgt_image, prompt, edit_prompt
                else:
                    return src_image, tgt_image, prompt, edit_prompt, image_name, edit_type
            if self.grayscale_editing and (not self.zoom_camera):
                # Grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
                grayscale_image = 0.2989 * src_image[0, :, :] + 0.5870 * src_image[1, :, :] + 0.1140 * src_image[2, :, :]
                tgt_image = grayscale_image.unsqueeze(0).repeat(3, 1, 1)
                edit_prompt = "Convert the input image to a black and white grayscale image while maintaining the original composition and details."
                if not self.eval:
                    return src_image, tgt_image, prompt, edit_prompt
                else:
                    return src_image, tgt_image, prompt, edit_prompt, image_name, edit_type
            if (not self.grayscale_editing) and self.zoom_camera:
                cropped = TF.center_crop(src_image, (256, 256))
                tgt_image = TF.resize(cropped, (512, 512))
                edit_prompt = "The central area of the input image is zoomed. The camera transitions from a wide shot to a closer position, narrowing its view."
                if not self.eval:
                    return src_image, tgt_image, prompt, edit_prompt
                else:
                    return src_image, tgt_image, prompt, edit_prompt, image_name, edit_type
            if self.grayscale_editing and self.zoom_camera:
                grayscale_image = 0.2989 * src_image[0, :, :] + 0.5870 * src_image[1, :, :] + 0.1140 * src_image[2, :, :]
                tgt_image = grayscale_image.unsqueeze(0).repeat(3, 1, 1)
                tgt_image = TF.center_crop(tgt_image, (256, 256))
                tgt_image = TF.resize(tgt_image, (512, 512))
                edit_prompt = "Convert the input image to a black and white grayscale image while maintaining the original composition and details. And the central area of the input image is zoomed, the camera transitions from a wide shot to a closer position, narrowing its view."
                if not self.eval:
                    return src_image, tgt_image, prompt, edit_prompt
                else:
                    return src_image, tgt_image, prompt, edit_prompt, image_name, edit_type
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def image_pair_loader(train_batch_size, num_workers, **args):
    dataset = ImageEditPairDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

def eval_image_pair_loader(eval_batch_size, num_workers, **args):
    dataset = ImageEditPairDataset(**args)
    return DataLoader(dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False)



if __name__ == "__main__":
    from src.flux.util import save_image
    example_dataset = ImageEditPairDataset(
        img_dir="", 
        img_size=512, 
        caption_type='json', 
        random_ratio=False,
        grayscale_editing=False,
        zoom_camera=False,
    )

    train_dataloader = DataLoader(
        example_dataset, 
        batch_size=1, 
        num_workers=4, 
        shuffle=False,
    )

    for step, batch in enumerate(train_dataloader):
        src_image, tgt_image, prompt, edit_prompt = batch
        os.makedirs("./debug", exist_ok=True)
        save_image(src_image, f"./debug/{step}-src_img.jpg")
        save_image(tgt_image, f"./debug/{step}-tgt_img.jpg")
        if step == 3:
            breakpoint()
