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

import os

from argparse import ArgumentParser

import numpy as np
import torch


from PIL import Image
from torchvision.utils import save_image

import json


from utils.clip_similarity import ClipSimilarity, ClipSimilarity_new
from tqdm import tqdm
from collections import defaultdict

# Assume ClipSimilarity, ClipSimilarity_new, compute_image_score, compute_new_score are imported

def merge_and_save(score, score_new, output_path):
    merged = {**score, **score_new}
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

def average_scores_in_folder(folder_path):
    avg_scores = defaultdict(float)
    count = 0
    for fname in os.listdir(folder_path):
        if fname.endswith("_clip.json"):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            for key, val in data.items():
                avg_scores[key] += val
            count += 1
    if count > 0:
        for key in avg_scores:
            avg_scores[key] /= count
    return dict(avg_scores)

def compute_new_score(clip_similarity_new, source_image_path, gen_image_path):
 

    # Split image
    image_src_tgt = Image.open(source_image_path).convert("RGB")
    image_src, image_tgt = image_src_tgt.crop((0, 0, image_src_tgt.width//2, image_src_tgt.height)), image_src_tgt.crop((image_src_tgt.width//2, 0, image_src_tgt.width, image_src_tgt.height))

    image_src_gen = Image.open(gen_image_path).convert("RGB")
    _, image_gen = image_src_gen.crop((0, 0, image_src_gen.width//2, image_src_gen.height)), image_src_gen.crop((image_src_gen.width//2, 0, image_src_gen.width, image_src_gen.height))
    if image_gen.size != image_tgt.size:
        image_gen = image_gen.resize(image_tgt.size, resample=Image.LANCZOS)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        image_src = torch.tensor(np.array(image_src)).float().permute(2,0,1)[None].div(255.).cuda()
        image_tgt = torch.tensor(np.array(image_tgt)).float().permute(2,0,1)[None].div(255.).cuda()
        image_gen = torch.tensor(np.array(image_gen)).float().permute(2,0,1)[None].div(255.).cuda()

        new_score = clip_similarity_new(
            image_src, image_tgt, image_gen,
            return_cross_scores=False, return_dict=True,
        )
        score.update(new_score)
    return score


def compute_image_score(clip_similarity, sample, image_path):



    image = Image.open(image_path).convert("RGB")
    image_0, image_1 = image.crop((0, 0, image.width//2, image.height)), image.crop((image.width//2, 0, image.width, image.height))
 

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        image_0 = torch.tensor(np.array(image_0)).float().permute(2,0,1)[None].div(255.).cuda()
        image_1 = torch.tensor(np.array(image_1)).float().permute(2,0,1)[None].div(255.).cuda()
        score = clip_similarity(
            image_0, image_1, sample["input"], sample["output"],
            return_cross_scores=False, return_dict=True,
        )
    return score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_data_root", required=False, type=str, default="/mnt/bn/dichang-bytenas/dichang-seed/xedit/data/evaluation_dataset/X-Edit-Bench_test/output_bench")
    parser.add_argument("--gen_data_root", required=False, type=str, default="/mnt/bn/dichang-bytenas/dichang-seed/xedit/data/evaluation_dataset/X-Edit-Bench_test/output_bench")
    parser.add_argument("--output_root", required=False, type=str, default="./")
    args = parser.parse_args()
    src_root = args.src_data_root
    gen_root = args.gen_data_root
    output_root = args.output_root
    error_log_path = os.path.join(output_root, "error_log.txt")
    
    clip_similarity = ClipSimilarity("ViT-B/32").cuda()
    clip_similarity_new = ClipSimilarity_new("ViT-B/32").cuda() 

    for subfolder in sorted(os.listdir(src_root)):
        subdir = os.path.join(src_root, subfolder)
        if not os.path.isdir(subdir):
            continue
        files = sorted(os.listdir(subdir))
        for fname in tqdm(files, desc=f"Processing {subfolder}"):
            if not fname.endswith(".png"):
                print(fname, " is not in png format.")
                continue
            
            src_tgt_img_path = os.path.join(subdir, fname)
            json_path = src_tgt_img_path.replace(".png", ".json")
            rel_path = os.path.relpath(subdir, src_root)
            output_dir = os.path.join(output_root, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, fname).replace(".png", "_clip.json")

            if os.path.exists(output_path):
                print(fname, " is processed already.")
                continue

            if not os.path.exists(json_path):
                with open(error_log_path, "a") as ef:
                    ef.write(f"[Missing JSON] {fname}\n")
                continue

            gen_image_path = src_tgt_img_path.replace(src_root, gen_root)
            if not os.path.exists(gen_image_path):
                with open(error_log_path, "a") as ef:
                    ef.write(f"[Missing Generated Image] {gen_image_path}\n")
                continue

            with open(json_path, "r") as f:
                sample = json.load(f)
            score = compute_image_score(clip_similarity, sample, gen_image_path)
            score_new = compute_new_score(clip_similarity_new, src_tgt_img_path, gen_image_path)
            merge_and_save(score, score_new, output_path)

        # After processing each subfolder
        avg_output_path = os.path.join(output_root, subfolder + "_avg.json")
        avg_scores = average_scores_in_folder(os.path.join(output_root, subfolder))
        with open(avg_output_path, "w") as f:
            json.dump(avg_scores, f, indent=2)
