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

import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5, save_image, tensor_to_pil_image, load_checkpoint)
from src.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor, IPDoubleStreamBlockProcessor, IPSingleStreamBlockProcessor, ImageProjModel
from src.flux.xflux_pipeline import XFluxSampler

from image_datasets.dataset import loader, eval_image_pair_loader, image_resize

from safetensors.torch import load_file
import json
logger = get_logger(__name__, log_level="INFO")

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config


def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    set_seed(args.seed)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            gpt_eval_path = os.path.join(args.output_dir,"Eval")
            os.makedirs(gpt_eval_path, exist_ok=True)

    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)
    if args.use_lora:
        lora_attn_procs = {}
    if args.use_ip:
        ip_attn_procs = {}
    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]

    if args.use_lora:
        for name, attn_processor in dit.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))

            if name.startswith("double_blocks") and layer_index in double_blocks_idx:
                if accelerator.is_main_process:
                    print("setting LoRA Processor for", name)
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                dim=3072, rank=args.rank
                )
            elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
                if accelerator.is_main_process:
                    print("setting LoRA Processor for", name)
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                dim=3072, rank=args.rank
                )
            else:
                lora_attn_procs[name] = attn_processor

        dit.set_attn_processor(lora_attn_procs)
    
    if args.use_ip:
        # unpack checkpoint
        checkpoint = load_checkpoint(args.ip_local_path, args.ip_repo_id, args.ip_name)
        prefix = "double_blocks."
        # blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            # if key.startswith(prefix):
            #     blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        ip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(os.getenv("CLIP_VIT")).to(
            accelerator.device, dtype=torch.bfloat16
        )
        ip_clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        ip_improj = ImageProjModel(4096, 768, 4)
        ip_improj.load_state_dict(proj)
        ip_improj = ip_improj.to(accelerator.device, dtype=torch.bfloat16)

        ip_attn_procs = {}

        for name, _ in dit.attn_processors.items():
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(accelerator.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = dit.attn_processors[name]
        dit.set_attn_processor(ip_attn_procs)


    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)



    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    print(f"Resuming from checkpoint {args.ckpt_dir}")
    dit_stat_dict = load_file(args.ckpt_dir)
    dit.load_state_dict(dit_stat_dict)
    dit = dit.to(weight_dtype)
    dit.eval()

    # test_dataloader = loader(**args.data_config)
    test_dataloader = eval_image_pair_loader(**args.data_config)



    # from deepspeed import initialize
    dit = accelerator.prepare(dit)

    # if accelerator.is_main_process:
    #     accelerator.init_trackers(args.tracker_project_name, {"test": None})

    logger.info("***** Running Evaluation *****")
    logger.info(f"  Instantaneous batch size = {args.eval_batch_size}")



    progress_bar = tqdm(
        range(0, len(test_dataloader)),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for step, batch in enumerate(test_dataloader):
        try:
            with accelerator.accumulate(dit):
                img, tgt_image, prompt, edit_prompt, img_name, edit_name = batch
                image_name = img_name[0]
                edit_type = edit_name[0]
                print(f"Sampling image {edit_type}/{image_name} for step {step}...")
                if args.use_ip:
                    sampler = XFluxSampler(clip=clip, t5=t5, ae=vae, model=dit, device=accelerator.device, ip_loaded=True,  spatial_condition=False, clip_image_processor=ip_clip_image_processor, image_encoder=ip_image_encoder, improj=ip_improj)
                elif args.use_spatial_condition:
                    sampler = XFluxSampler(clip=clip, t5=t5, ae=vae, model=dit, device=accelerator.device, ip_loaded=False,  spatial_condition=True, clip_image_processor=None, image_encoder=None, improj=None,share_position_embedding=args.share_position_embedding)
                else:
                    sampler = XFluxSampler(clip=clip, t5=t5, ae=vae, model=dit, device=accelerator.device, ip_loaded=False,  spatial_condition=False, clip_image_processor=None, image_encoder=None, improj=None)
                with torch.no_grad():
                    result = sampler(prompt=edit_prompt,
                                        width=args.sample_width,
                                        height=args.sample_height,
                                        num_steps=args.sample_steps,
                                        image_prompt=None, # ip_adapter
                                        true_gs=args.cfg_scale,
                                        seed=args.seed,
                                        ip_scale=args.ip_scale if args.use_ip else 1.0,
                                        source_image=img if args.use_spatial_condition else None,
                                        )
                print(f"Result for prompt #{step} is generated")
                # Save for FID 
                gen_save_path = os.path.join(args.output_dir, edit_type, "gen_images")
                inp_save_path = os.path.join(args.output_dir, edit_type, "ref_images")
                gt_save_path = os.path.join(args.output_dir, edit_type, "gt_images")
                os.makedirs(gen_save_path, exist_ok=True)
                os.makedirs(inp_save_path, exist_ok=True)
                os.makedirs(gt_save_path, exist_ok=True)
                result.save(os.path.join(gen_save_path, f"{image_name}.png"))
                save_image(img, os.path.join(inp_save_path, f"{image_name}.png"))
                save_image(tgt_image, os.path.join(gt_save_path, f"{image_name}.png"))
                
                # Save for GPT Eval
                gpt_save_path = os.path.join(gpt_eval_path, edit_type, image_name)
                os.makedirs(os.path.join(gpt_eval_path, edit_type), exist_ok=True)

                # Save for Vis
                ref_img = tensor_to_pil_image(img)
                gt_img = tensor_to_pil_image(tgt_image)
                gen_img = result
                assert ref_img.size == gt_img.size == gen_img.size == (512, 512)
                ref_gen_img = Image.new("RGB", (512 * 2, 512))
                ref_gen_gt_img = Image.new("RGB", (512 * 3, 512))  
                ref_gen_img.paste(ref_img, (0, 0))
                ref_gen_img.paste(gen_img, (512, 0))
                ref_gen_gt_img.paste(ref_img, (0, 0))
                ref_gen_gt_img.paste(gen_img, (512, 0))
                ref_gen_gt_img.paste(gt_img, (1024, 0))
                gpt_gen_img_path = gpt_save_path +".jpg"
                ref_gen_img.save(gpt_gen_img_path)


                progress_bar.update(1)
        except:
            print(f"Error for image {edit_type}/{image_name} ")
            with open(os.path.join(args.output_dir,"error_image_list.txt"), 'a', encoding='utf-8') as f:
                f.write(f"Error occured for image {edit_type}/{image_name}  during inference.\n")
            progress_bar.update(1)
            continue

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
